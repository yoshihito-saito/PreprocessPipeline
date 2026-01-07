import os
import re
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import savemat
from pathlib import Path

def extract_datetime(path):
    """Extract YYMMDD_HHMMSS from path."""
    m = re.search(r'(\d{6}_\d{6})', path)
    if m:
        return datetime.strptime(m.group(1), "%y%m%d_%H%M%S")
    return datetime.min 

def select_folder(initial_drive="S:\\"):
    """Open folder selection dialog."""
    # Create and hide root window
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    root.update()

    # Open dialog
    basePath = filedialog.askdirectory(
        title="Select data folder",
        initialdir=initial_drive if os.path.exists(initial_drive) else os.getcwd()
    )

    root.destroy()
    if basePath:
        print(f"Selected folder: {basePath}")
    return basePath


def get_sampling_rate(xml_path):
    """
    Extracts the sampling rate from an XML file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        sr_tag = root.find('.//sampleRate')
        if sr_tag is None:
            acq = root.find('.//acquisitionSystem')
            if acq is not None:
                sr_tag = acq.find('samplingRate')
        return float(sr_tag.text)
    except (AttributeError, ValueError):
        return None
 

def load_xml_metadata(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    anat_grps = []
    skipped_channels = []
    anat_desc = root.find('anatomicalDescription')
    if anat_desc is not None:
        ch_grps = anat_desc.find('channelGroups')
        if ch_grps is not None:
            for group in ch_grps.findall('group'):
                channels = []
                tags = group.findall('n')
                if not tags: tags = group.findall('channel')
                for ch in tags:
                    try:
                        if ch.text and ch.text.strip():
                            val = int(ch.text)
                            channels.append(val)
                            if ch.get('skip') == '1':
                                skipped_channels.append(val)
                    except (ValueError, TypeError):
                        continue
                if channels:
                    anat_grps.append(channels)

    spk_channels = []
    spk_desc = root.find('spikeDetection')
    has_spk_groups = False
    
    if spk_desc is not None:
        ch_grps = spk_desc.find('channelGroups')
        if ch_grps is not None:
            groups = ch_grps.findall('group')
            if groups:
                has_spk_groups = True
                for group in groups:
                    channels_container = group.find('channels')
                    tags = []
                    if channels_container is not None:
                        tags = channels_container.findall('n')
                        if not tags: tags = channels_container.findall('channel')
                    else:
                        tags = group.findall('n')
                        if not tags: tags = group.findall('channel')
                    for ch in tags:
                        try:
                            if ch.text and ch.text.strip():
                                spk_channels.append(int(ch.text))
                        except (ValueError, TypeError):
                            continue

    return anat_grps, spk_channels, has_spk_groups, skipped_channels, root

def create_channel_map(basepath, outputDir, basename=None, electrode_type=None, reject_channels=None, probe_assignments=None):
    if reject_channels is None:
        reject_channels = []
    
    base_path = Path(basepath)
    if basename is None:
        basename = base_path.name
        if basename.endswith('.xml'):
            basename = os.path.splitext(basename)[0]

    xml_path = base_path / f"{basename}.xml"

    if not xml_path.exists():
        print(f"Error: XML file {xml_path} not found.")
        return None

    print(f"Reading XML from: {xml_path}")
    anat_grps, spk_channels, has_spk_groups, skipped_channels, root = load_xml_metadata(xml_path)

    ngroups = len(anat_grps)
    if ngroups == 0:
        print("Warning: No anatomical groups found in XML.")
        return None
    print(f"Found {ngroups} anatomical groups (XML document order).")

    if electrode_type is None:
        electrode_type = 'staggered'
        desc_node = root.find('generalInfo/description')
        if desc_node is not None and desc_node.text:
            val = desc_node.text.strip().lower()
            if 'neuropixel' in val: electrode_type = 'NeuroPixel'
            elif 'staggered' in val: electrode_type = 'staggered'
            elif 'neurogrid' in val: electrode_type = 'neurogrid'
            elif 'grid' in val: electrode_type = 'neurogrid'
            elif 'poly3' in val: electrode_type = 'poly3'
            elif 'poly5' in val: electrode_type = 'poly5'
    
    print(f"Default electrode layout: {electrode_type}")

    if not probe_assignments:
        print("No probe assignments provided. Using default configuration (XML order).")
        probe_assignments = [
            {'type': electrode_type, 'groups': list(range(ngroups)), 'x_offset': 0}
        ]

    channel_coords = [] 

    for probe_idx, probe in enumerate(probe_assignments):
        p_type = probe.get('type', electrode_type)
        p_groups = probe.get('groups', [])
        p_x_offset = probe.get('x_offset', 0)
        
        print(f"Processing Probe {probe_idx+1}: Type={p_type}, Groups={p_groups}, X_Offset={p_x_offset}")

        for local_idx, g_idx in enumerate(p_groups):
            if g_idx >= ngroups:
                continue
            
            tchannels = anat_grps[g_idx]
            n_ch = len(tchannels)
            
            x = np.zeros(n_ch)
            y = np.zeros(n_ch)
            
            # --- Coordinate Calculation ---

            shank_id = local_idx + 1 
            
            if p_type == 'double_sided':
                # double_sided logic:
                # local_idx: 0(Back), 1(Front), 2(Back), 3(Front)...

                pair_idx = local_idx // 2
                is_front = (local_idx % 2 == 1)
                y = np.arange(1, n_ch + 1) * -20.0
                x[:] = 20.0
                x[::2] = -20.0
                # Pair 0 -> 200, Pair 1 -> 400 ...
                pair_origin = (pair_idx + 1) * 400.0
            
                # 60 um offset for front side
                intra_pair_offset = 80.0 if is_front else 0.0
                x = x + pair_origin + intra_pair_offset

            elif p_type == 'NeuroPixel':
                x_pat = [20, 60, 0, 40]
                x = np.tile(x_pat, (n_ch // 4) + 1)[:n_ch]
                y_base = (np.arange(n_ch) // 2) + 1
                y = y_base * -20.0
                x = x + shank_id * 200

            elif p_type == 'staggered':
                x[:] = 20.0
                y = np.arange(1, n_ch + 1) * -20.0
                x[::2] = -20.0
                x = x + shank_id * 200

            elif p_type == 'poly3':
                ext = n_ch % 3
                poly = (np.arange(1, n_ch - ext + 1)) % 3
                x[:] = 0 
                idx_p1 = np.where(poly == 1)[0] + ext
                idx_p2 = np.where(poly == 2)[0] + ext
                idx_p0 = np.where(poly == 0)[0] + ext
                x[idx_p1] = -18; x[idx_p2] = 0; x[idx_p0] = 18
                x[:ext] = 0
                mask_18 = (x == 18); y[mask_18] = np.arange(1, np.sum(mask_18) + 1) * -20
                mask_0 = (x == 0); y[mask_0] = np.arange(1, np.sum(mask_0) + 1) * -20 - 10 + ext * 20
                mask_m18 = (x == -18); y[mask_m18] = np.arange(1, np.sum(mask_m18) + 1) * -20
                x = x + shank_id * 200

            elif p_type == 'poly5':
                ext = n_ch % 5
                poly = (np.arange(1, n_ch - ext + 1)) % 5
                x[:] = np.nan
                x[np.where(poly == 1)[0] + ext] = -36
                x[np.where(poly == 2)[0] + ext] = -18
                x[np.where(poly == 3)[0] + ext] = 0
                x[np.where(poly == 4)[0] + ext] = 18
                x[np.where(poly == 0)[0] + ext] = 36
                if ext > 0: x[:ext] = 18 * ((-1.0)**np.arange(1, ext + 1))
                for val, y_off in [(36, 0), (18, -14), (0, 0), (-18, -14), (-36, 0)]:
                    mask = (x == val)
                    if np.any(mask):
                        y[mask] = np.arange(1, np.sum(mask) + 1) * -28 + y_off
                x = x + shank_id * 200

            elif p_type == 'neurogrid':
                for i in range(n_ch):
                    x[i] = n_ch - (i + 1)
                    y[i] = -(i + 1) * 30
                x = x + shank_id * 30
            
            x = x + p_x_offset
            
            if p_type == 'neurogrid':
                k_val = (g_idx // 4) + 1
            else:
                k_val = g_idx + 1

            for i in range(n_ch):
                channel_coords.append({
                    'id': tchannels[i],
                    'x': x[i],
                    'y': y[i],
                    'k': k_val
                })

    sorted_coords = sorted(channel_coords, key=lambda d: d['id'])
    
    if not sorted_coords:
        return None

    Nchannels = len(sorted_coords)
    xcoords = np.array([d['x'] for d in sorted_coords])
    ycoords = np.array([d['y'] for d in sorted_coords])
    kcoords = np.array([d['k'] for d in sorted_coords])
    real_channels = np.array([d['id'] for d in sorted_coords])

    connected = np.ones(Nchannels, dtype=bool)

    if reject_channels:
        for rc in reject_channels:
            matches = np.where(real_channels == rc)[0]
            if len(matches) > 0:
                connected[matches] = False
    
    if skipped_channels:
        for sc in skipped_channels:
            matches = np.where(real_channels == sc)[0]
            if len(matches) > 0:
                connected[matches] = False
    
    if has_spk_groups:
        spk_set = set(spk_channels)
        for i, ch_id in enumerate(real_channels):
            if ch_id not in spk_set:
                connected[i] = False
    
    chanMap = np.arange(1, Nchannels + 1).reshape(1, -1)
    chanMap0ind = real_channels.reshape(1, -1)
    
    save_dict = {
        'chanMap': chanMap.astype(float),
        'chanMap0ind': chanMap0ind.astype(float),
        'connected': connected.reshape(-1, 1).astype(float),
        'xcoords': xcoords.reshape(-1, 1).astype(float),
        'ycoords': ycoords.reshape(-1, 1).astype(float),
        'kcoords': kcoords.reshape(-1, 1).astype(float)
    }

    out_file = Path(outputDir) / 'chanMap.mat'
    savemat(out_file, save_dict)
    print(f"Successfully saved chanMap.mat to {out_file}")
    
    return out_file

