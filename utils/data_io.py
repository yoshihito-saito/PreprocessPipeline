import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import savemat
from pathlib import Path
import h5py 

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
                    'k': k_val,
                    'p': probe_idx + 1
                })

    sorted_coords = sorted(channel_coords, key=lambda d: d['id'])
    
    if not sorted_coords:
        return None

    Nchannels = len(sorted_coords)
    xcoords = np.array([d['x'] for d in sorted_coords])
    ycoords = np.array([d['y'] for d in sorted_coords])
    kcoords = np.array([d['k'] for d in sorted_coords])
    pcoords = np.array([d['p'] for d in sorted_coords])
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
        'kcoords': kcoords.reshape(-1, 1).astype(float),
        'probe_ids': pcoords.reshape(-1, 1).astype(float)
    }

    out_file = Path(outputDir) / 'chanMap.mat'
    savemat(out_file, save_dict)
    print(f"Successfully saved chanMap.mat to {out_file}")
    
    return out_file

def convert_dual_side_map(chan_map_file, x_shift=6.0, pairs_to_merge=None, custom_shank_positions=None):
    """
    Converts a channel map to merge pairs of shanks (representing front/back)
    into single shanks with a small lateral offset.

    Parameters
    ----------
    chan_map_file : str or Path
        Path to the .mat file.
    x_shift : float, optional
        Microns to shift the 'front' side (2nd in pair). Default is 6.0.
    pairs_to_merge : list of tuples, optional
        List of (back_shank_id, front_shank_id) to merge.
    custom_shank_positions : dict, optional
        Dictionary {shank_id: x_um} to explicitly set positions.
    """
    from scipy.io import loadmat
    
    p = Path(chan_map_file)
    if not p.exists():
        print(f"File not found: {p}")
        return

    data = loadmat(p)
    
    x = data['xcoords'].flatten()
    y = data['ycoords'].flatten()
    k = data['kcoords'].flatten()
    
    x_shape = data['xcoords'].shape
    k_shape = data['kcoords'].shape
    
    unique_shanks = np.unique(k)
    unique_shanks.sort()
    
    if pairs_to_merge is None:
        n_pairs = len(unique_shanks) // 2
        pairs_to_merge = []
        for i in range(n_pairs):
            pairs_to_merge.append((unique_shanks[2*i], unique_shanks[2*i+1]))
        print(f"Auto-detected {len(pairs_to_merge)} pairs to merge.")
    else:
        print(f"Using provided list of {len(pairs_to_merge)} pairs to merge.")
    
    for (s_back, s_front) in pairs_to_merge:
        idx_back = (k == s_back)
        idx_front = (k == s_front)
        
        if not np.any(idx_back) or not np.any(idx_front):
            print(f"Warning: Missing channels for pair ({s_back}, {s_front}).")
            continue
        
        k[idx_front] = s_back
        
        mean_x_back = np.mean(x[idx_back])
        mean_x_front = np.mean(x[idx_front])
        current_offset = mean_x_front - mean_x_back
        
        x[idx_front] = x[idx_front] - current_offset + x_shift
        print(f"Merged Shank {s_front} into {s_back}. Shifted X by {-current_offset + x_shift:.2f} um.")

    # Custom positions override
    if custom_shank_positions:
        print(f"Applying custom positions to {len(custom_shank_positions)} shanks.")
        for s_id, target_x in custom_shank_positions.items():
            idx_group = (k == s_id)
            if np.any(idx_group):
                start_mean = np.mean(x[idx_group])
                x[idx_group] += (target_x - start_mean)
                print(f"  Shank {s_id}: Moved to {target_x:.1f} (shift {target_x - start_mean:.1f})")

    data['xcoords'] = x.reshape(x_shape)
    data['kcoords'] = k.reshape(k_shape)
    
    savemat(p, data)
    print(f"Updated {p} with dual-side conversion.")

def load_rez(rez_path):
    """
    Load MATLAB v7.3 rez.mat file using h5py and return as a dictionary.
    Properly handles axis reversal for multi-dimensional arrays (Fortran vs C order).
    """
    def h5_to_dict(obj):
        if isinstance(obj, h5py.Group):
            # MATLAB structs are stored as Groups
            return {k: h5_to_dict(obj[k]) for k in obj.keys()}
        elif isinstance(obj, h5py.Dataset):
            data = obj[()]
            if isinstance(data, np.ndarray):
                # MATLAB is Fortran-order (column-major), Python/HDF5 is C-order (row-major)
                # For N-dimensional arrays, we need to reverse ALL axes
                if data.ndim > 1:
                    # Reverse axis order: (A,B,C) in MATLAB -> (C,B,A) in h5py -> reverse to (A,B,C)
                    data = np.ascontiguousarray(np.transpose(data, axes=range(data.ndim-1, -1, -1)))
                # Squeeze singleton dimensions to match MATLAB's behavior
                data = np.squeeze(data)
            return data
        return obj

    with h5py.File(rez_path, 'r') as f:
        if 'rez' in f:
            return h5_to_dict(f['rez'])
        else:
            return h5_to_dict(f)

def rezToPhy(rez, save_path):
    """
    Extract and save data for Phy from Kilosort rez structure (Python dictionary format).
    Handles proper array shapes and 0-based indexing for Phy compatibility.
    """
    
    # Create save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Cleanup: delete existing related files
    outputs = [
        'amplitudes.npy', 'channel_map.npy', 'channel_positions.npy', 'pc_features.npy',
        'pc_feature_ind.npy', 'similar_templates.npy', 'spike_clusters.npy', 'spike_templates.npy',
        'spike_times.npy', 'templates.npy', 'templates_ind.npy', 'template_features.npy',
        'template_feature_ind.npy', 'whitening_mat.npy', 'whitening_mat_inv.npy'
    ]
    
    for filename in os.listdir(save_path):
        if filename in outputs:
            os.remove(os.path.join(save_path, filename))
            
    phy_dir = os.path.join(save_path, '.phy')
    if os.path.exists(phy_dir):
        shutil.rmtree(phy_dir)

    # 2. Data extraction (rez.st3)
    st3 = rez['st3']
    n_spikes = st3.shape[0]
    
    # Spike times (samples)
    spike_times = st3[:, 0].astype(np.uint64)
    # Template IDs (convert to 0-based)
    spike_templates = (st3[:, 1] - 1).astype(np.uint32)
    
    # Cluster IDs (convert to 0-based)
    if st3.shape[1] > 4:
        spike_clusters = (st3[:, 4] - 1).astype(np.int32)
    else:
        spike_clusters = spike_templates.copy().astype(np.int32)

    amplitudes = st3[:, 2]

    # 3. Get Ops information
    ops = rez['ops']
    chan_map = np.atleast_1d(ops['chanMap']).flatten()
    chan_map_0ind = (chan_map - 1).astype(np.int32)  # Convert to 0-based
    
    connected = np.atleast_1d(rez['connected']).flatten().astype(bool)
    xcoords = np.atleast_1d(rez['xcoords']).flatten()
    ycoords = np.atleast_1d(rez['ycoords']).flatten()
    
    # 4. Reconstruct templates
    # After load_rez axis reversal: U should be (n_chan, n_filt, rank), W should be (nt, n_filt, rank)
    U = rez['U']
    W = rez['W']
    
    # U: (n_chan, n_filt, rank), W: (nt, n_filt, rank)
    # Output: templates (n_filt, nt, n_chan) for Phy
    templates = np.einsum('cfr, tfr -> ftc', U, W)
    n_templates = templates.shape[0]
    n_chan = U.shape[0]
    
    # templates_ind: (n_templates, n_channels)
    templates_inds = np.tile(np.arange(n_chan), (n_templates, 1)).astype(np.int32)

    # 5. Features
    # pc_features: (n_spikes, n_pc_features, n_channels_loc)
    pc_features = rez['cProjPC']
    # pc_feature_ind: (n_templates, n_channels_loc) - 0-based
    pc_feature_inds = (np.atleast_2d(rez['iNeighPC']) - 1).astype(np.int32)
    
    # template_features: (n_spikes, n_template_features)
    template_features = rez['cProj']
    # template_feature_ind: (n_templates, n_template_features) - 0-based
    template_feature_inds = (np.atleast_2d(rez['iNeigh']) - 1).astype(np.int32)
    
    # pc_feature_ind must be (n_templates, n_channels_loc)
    if pc_feature_inds.shape[0] != n_templates:
        pc_feature_inds = pc_feature_inds.T
    
    # template_feature_ind must be (n_templates, n_template_features)
    if template_feature_inds.shape[0] != n_templates:
        template_feature_inds = template_feature_inds.T

    # 6. Save as .npy files (ensure contiguous arrays)
    np.save(os.path.join(save_path, 'spike_times.npy'), np.ascontiguousarray(spike_times))
    np.save(os.path.join(save_path, 'spike_templates.npy'), np.ascontiguousarray(spike_templates))
    np.save(os.path.join(save_path, 'spike_clusters.npy'), np.ascontiguousarray(spike_clusters))
    
    np.save(os.path.join(save_path, 'amplitudes.npy'), np.ascontiguousarray(amplitudes))
    np.save(os.path.join(save_path, 'templates.npy'), np.ascontiguousarray(templates.astype(np.float32)))
    np.save(os.path.join(save_path, 'templates_ind.npy'), np.ascontiguousarray(templates_inds))
    
    np.save(os.path.join(save_path, 'channel_map.npy'), np.ascontiguousarray(chan_map_0ind[connected]))
    coords = np.column_stack((xcoords[connected], ycoords[connected]))
    np.save(os.path.join(save_path, 'channel_positions.npy'), np.ascontiguousarray(coords))
    
    np.save(os.path.join(save_path, 'template_features.npy'), np.ascontiguousarray(template_features.astype(np.float32)))
    np.save(os.path.join(save_path, 'template_feature_ind.npy'), np.ascontiguousarray(template_feature_inds))
    np.save(os.path.join(save_path, 'pc_features.npy'), np.ascontiguousarray(pc_features.astype(np.float32)))
    np.save(os.path.join(save_path, 'pc_feature_ind.npy'), np.ascontiguousarray(pc_feature_inds))
    
    whitening_matrix = rez['Wrot'] / 200
    whitening_matrix_inv = np.linalg.pinv(whitening_matrix)  # Use pinv for numerical stability
    np.save(os.path.join(save_path, 'whitening_mat.npy'), np.ascontiguousarray(whitening_matrix.astype(np.float32)))
    np.save(os.path.join(save_path, 'whitening_mat_inv.npy'), np.ascontiguousarray(whitening_matrix_inv.astype(np.float32)))
    
    if 'simScore' in rez:
        np.save(os.path.join(save_path, 'similar_templates.npy'), np.ascontiguousarray(rez['simScore'].astype(np.float32)))

    # 7. Create params.py
    params_path = os.path.join(save_path, 'params.py')
    fb_val = ops.get('fbinary', 'recording.dat')
    # Handle HDF5 string encoding (array of ints)
    if isinstance(fb_val, np.ndarray) and fb_val.dtype.kind in 'ui':
        fbinary = "".join([chr(int(c)) for c in fb_val.flatten()])
    else:
        fbinary = str(fb_val)
        
    dat_path = '../' + os.path.basename(fbinary)
    n_chan_tot = int(np.atleast_1d(ops['NchanTOT']).flatten()[0])
    fs = float(np.atleast_1d(ops['fs']).flatten()[0])
    
    with open(params_path, 'w') as f:
        f.write(f"dat_path = r'{dat_path}'\n")
        f.write(f"n_channels_dat = {n_chan_tot}\n")
        f.write("dtype = 'int16'\n")
        f.write("offset = 0\n")
        f.write(f"sample_rate = {fs}\n")
        f.write("hp_filtered = False\n")

    print(f"Done! Phy files saved to {save_path}")