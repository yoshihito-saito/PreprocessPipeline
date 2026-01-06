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
    m = re.search(r'(\d{6}_\d{6})', path)
    if m:
        return datetime.strptime(m.group(1), "%y%m%d_%H%M%S")
    return datetime.min 

def select_folder(initial_drive="S:\\"):
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    root.lift()  # Bring window to front
    root.attributes('-topmost', True)  # Keep on top
    root.update()  # Process pending events

    # Open folder selection dialog with initial directory
    basePath = filedialog.askdirectory(
        title="Select data folder",
        initialdir=initial_drive if os.path.exists(initial_drive) else os.getcwd()
    )

    root.destroy()  # Destroy root window
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


def create_channel_map(basepath, basename=None, electrode_type=None, reject_channels=None, probe_assignments=None):
    """
    Python implementation of createChannelMapFile_KSW with multiple probe support.
    Generates a chanMap.mat file based on XML configuration and electrode type.
    
    probe_assignments: List of dicts, e.g. :
    [
        {'type': 'staggered', 'groups': [0,1,2,3], 'x_offset': 0},
        {'type': 'poly3', 'groups': [4,5,6,7], 'x_offset': 2000}
    ]
    If None, uses electrode_type for all groups.
    """
    base_path = Path(basepath)
    if basename is None:
        basename = base_path.name
    if reject_channels is None:
        reject_channels = []

    xml_path = base_path / f"{basename}.xml"
                
    if not xml_path.exists():
        print(f"Error: Could not find XML file at {xml_path}")
        return None

    print(f"Reading XML from: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract AnatGrps and Skips
    anat_grps = []
    anat_skips = []
    anat_desc = root.find('.//anatomicalDescription')
    if anat_desc is not None:
        ch_grps = anat_desc.find('channelGroups')
        if ch_grps is not None:
            for group in ch_grps.findall('group'):
                channels = []
                skips = []
                tags = group.findall('channel') if group.find('channel') is not None else group.findall('n')
                for ch_tag in tags:
                    channels.append(int(ch_tag.text))
                    skips.append(int(ch_tag.attrib.get('skip', '0')))
                if channels:
                    anat_grps.append(channels)
                    anat_skips.append(skips)
    
    ngroups = len(anat_grps)
    if ngroups == 0:
        print("Warning: No anatomical groups found.")
        return None

    # Default probe assignment if not provided
    if probe_assignments is None:
        if electrode_type is None:
            electrode_type = 'staggered'
            desc = root.find('.//description')
            if desc is not None and desc.text:
                if 'NeuroPixel' in desc.text: electrode_type = 'NeuroPixel'
                elif 'poly3' in desc.text: electrode_type = 'poly3'
                elif 'poly5' in desc.text: electrode_type = 'poly5'
        probe_assignments = [{'type': electrode_type, 'groups': list(range(ngroups)), 'x_offset': 0}]

    # Extract SpkGrps
    spk_channels = []
    spk_desc = root.find('.//spikeDetection')
    if spk_desc is not None:
        ch_grps = spk_desc.find('channelGroups')
        if ch_grps is not None:
            for group in ch_grps.findall('group'):
                channels_wrapper = group.find('channels')
                tags = (channels_wrapper.findall('channel') if channels_wrapper is not None else group.findall('channel')) or \
                       (channels_wrapper.findall('n') if channels_wrapper is not None else group.findall('n'))
                spk_channels.extend([int(t.text) for t in tags])

    xcoords_list = []
    ycoords_list = []
    kcoords_list = []
    side_list = []
    probe_id_list = []
    ordered_channels = []

    last_shank_id = 0

    for i_p, pa in enumerate(probe_assignments):
        p_type = pa.get('type', 'staggered')
        p_groups_indices = pa.get('groups', [])
        p_x_offset = pa.get('x_offset', 0)
        
        p_anat_grps = [anat_grps[i] for i in p_groups_indices]
        p_ngroups = len(p_anat_grps)
        
        if p_type == 'NeuroPixel':
            for a in range(p_ngroups):
                x_pattern = np.array([20, 60, 0, 40])
                tchannels = p_anat_grps[a]
                n_ch = len(tchannels)
                x = (np.tile(x_pattern, (n_ch // 4) + 1)[:n_ch]) + p_x_offset
                y = np.repeat(np.arange(n_ch // 2) * -20, 2)[:n_ch]
                xcoords_list.extend(x.tolist())
                ycoords_list.extend(y.tolist())
                kcoords_list.extend([last_shank_id + a + 1] * n_ch)
                probe_id_list.extend([i_p + 1] * n_ch)
                side_list.extend([''] * n_ch)
                ordered_channels.extend(tchannels)
                
        elif p_type == 'staggered':
            for a in range(p_ngroups):
                tchannels = p_anat_grps[a]
                n_ch = len(tchannels)
                x = np.full(n_ch, 20.0)
                y = np.arange(1, n_ch + 1) * -20.0
                x[::2] = -20.0 
                x = x + (a + 1) * 200 + p_x_offset
                xcoords_list.extend(x.tolist())
                ycoords_list.extend(y.tolist())
                kcoords_list.extend([last_shank_id + a + 1] * n_ch)
                probe_id_list.extend([i_p + 1] * n_ch)
                side_list.extend([''] * n_ch)
                ordered_channels.extend(tchannels)

        elif p_type == 'double_sided':
            for ix in range(p_ngroups // 2):
                idx_a = ix * 2
                idx_b = ix * 2 + 1
                chan_a = p_anat_grps[idx_a]
                chan_b = p_anat_grps[idx_b]
                
                n_a = len(chan_a)
                xa = np.full(n_a, 20.0)
                ya = np.arange(1, n_a + 1) * -20.0
                xa[::2] = -20.0
                xa = xa + (ix + 1) * 200 + p_x_offset
                
                n_b = len(chan_b)
                xb = np.zeros(n_b)
                yb = np.arange(1, n_b + 1) * -20.0
                xb[::2] = xa[:n_b:2] + 40
                xb[1::2] = xa[1:n_b:2] - 40
                
                xcoords_list.extend(xa.tolist()); ycoords_list.extend(ya.tolist())
                kcoords_list.extend([last_shank_id + ix + 1] * n_a)
                probe_id_list.extend([i_p + 1] * n_a)
                side_list.extend(['Back'] * n_a); ordered_channels.extend(chan_a)
                
                xcoords_list.extend(xb.tolist()); ycoords_list.extend(yb.tolist())
                kcoords_list.extend([last_shank_id + ix + 1] * n_b)
                probe_id_list.extend([i_p + 1] * n_b)
                side_list.extend(['Front'] * n_b); ordered_channels.extend(chan_b)
                
        elif p_type == 'poly3':
            for a in range(p_ngroups):
                tchannels = p_anat_grps[a]
                n_ch = len(tchannels)
                x = np.zeros(n_ch); y = np.zeros(n_ch)
                extrachannels = n_ch % 3
                polyline = np.arange(n_ch - extrachannels) % 3
                x[np.where(polyline == 1)[0] + extrachannels] = -18
                x[np.where(polyline == 2)[0] + extrachannels] = 0
                x[np.where(polyline == 0)[0] + extrachannels] = 18
                for val in [18, 0, -18]:
                    mask = (x == val); count = np.sum(mask)
                    if count > 0:
                        y[mask] = (np.arange(1, count + 1) * -20) - (10 if val == 0 else 0) + (extrachannels * 20 if val == 0 else 0)
                x = x + (a + 1) * 200 + p_x_offset
                xcoords_list.extend(x.tolist()); ycoords_list.extend(y.tolist())
                kcoords_list.extend([last_shank_id + a + 1] * n_ch)
                probe_id_list.extend([i_p + 1] * n_ch)
                side_list.extend([''] * n_ch); ordered_channels.extend(tchannels)

        elif p_type == 'poly5':
            for a in range(p_ngroups):
                tchannels = p_anat_grps[a]
                n_ch = len(tchannels)
                x = np.zeros(n_ch); y = np.zeros(n_ch)
                extrachannels = n_ch % 5
                polyline = np.arange(n_ch - extrachannels) % 5
                x[np.where(polyline == 1)[0] + extrachannels] = -2 * 18
                x[np.where(polyline == 2)[0] + extrachannels] = -18
                x[np.where(polyline == 3)[0] + extrachannels] = 0
                x[np.where(polyline == 4)[0] + extrachannels] = 18
                x[np.where(polyline == 0)[0] + extrachannels] = 2 * 18
                if extrachannels > 0: x[:extrachannels] = 18 * ((-1) ** np.arange(1, extrachannels + 1))
                for val in [36, 18, 0, -18, -36]:
                    mask = (x == val); count = np.sum(mask)
                    if count > 0:
                        y[mask] = np.arange(1, count + 1) * -28 - (14 if val in [18, -18] else 0)
                x = x + (a + 1) * 200 + p_x_offset
                xcoords_list.extend(x.tolist()); ycoords_list.extend(y.tolist())
                kcoords_list.extend([last_shank_id + a + 1] * n_ch)
                probe_id_list.extend([i_p + 1] * n_ch)
                side_list.extend([''] * n_ch); ordered_channels.extend(tchannels)

        # Update last_shank_id to ensure shanks are unique across probes
        shanks_in_probe = (p_ngroups // 2) if p_type == 'double_sided' else p_ngroups
        last_shank_id += shanks_in_probe

    # Map back to absolute indices
    all_channels_flat = np.concatenate(anat_grps)
    max_ch = np.max(all_channels_flat)
    connected = np.zeros(max_ch + 1, dtype=bool)
    side = np.empty((max_ch + 1, 1), dtype=object); side[:] = ''
    
    for grp, skips in zip(anat_grps, anat_skips):
        for ch, s in zip(grp, skips):
            connected[ch] = (s == 0)

    if reject_channels:
        for rc in reject_channels:
            if 0 <= rc < len(connected): connected[rc] = False
    
    if spk_channels:
        for ch in range(len(connected)):
            if connected[ch] and ch not in spk_channels: connected[ch] = False

    xcoords_sorted = np.zeros(max_ch + 1); ycoords_sorted = np.zeros(max_ch + 1)
    kcoords_final = np.zeros(max_ch + 1)
    probe_ids_final = np.zeros(max_ch + 1)
    
    for i, ch_idx in enumerate(ordered_channels):
        xcoords_sorted[ch_idx] = xcoords_list[i]
        ycoords_sorted[ch_idx] = ycoords_list[i]
        kcoords_final[ch_idx] = kcoords_list[i]
        probe_ids_final[ch_idx] = probe_id_list[i]
        side[ch_idx] = side_list[i]

    save_dict = {
        'chanMap': np.arange(1, max_ch + 2).astype(float),
        'connected': connected.astype(float).reshape(-1, 1),
        'xcoords': xcoords_sorted.astype(float).reshape(-1, 1),
        'ycoords': ycoords_sorted.astype(float).reshape(-1, 1),
        'kcoords': kcoords_final.astype(float).reshape(-1, 1),
        'probe_ids': probe_ids_final.astype(float).reshape(-1, 1),
        'side': side,
        'chanMap0ind': np.arange(max_ch + 1).astype(float).reshape(-1, 1)
    }
    
    out_path = base_path / 'chanMap.mat'
    savemat(out_path, save_dict)
    print(f"Successfully saved multi-probe channel map to: {out_path}")
    return out_path
