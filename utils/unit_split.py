import numpy as np
import spikeinterface as si
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

def autosplit_outliers_pca(
    analyzer,
    # ---- Distance Gate Parameters (Main) ----
    contamination: float = 0.05,
    threshold_mode: str = "empirical",
    random_state: int = 42,
    min_clean_frac: float = 0.50,
    relax_factor: float = 0.5,
    # ---- Waveform similarity (Rescue) ----
    use_waveform_gate: bool = True,
    wf_threshold: float = 0.9,
    wf_cand_max: int | None = 3000,
    wf_ms_before: float = 1.0,
    wf_ms_after: float = 2.0,
    wf_n_chans: int = 20,
    wf_center: str = "demean",
    # ---- Output Control ----
    squeeze_all_outlier_to_new: bool = True,
    return_details: bool = False,
    verbose: bool = True,
):
    """
    Split per-unit outliers using PCA features and Mahalanobis distance with an optional waveform rescue gate.
    Returns a new Sorting object with 0-indexed integer IDs.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        SpikeInterface SortingAnalyzer object with 'principal_components' extension computed.
    contamination : float, default 0.05
        The estimated fraction of outliers in each unit. Determines the distance threshold.
    threshold_mode : str, default "empirical"
        Method to calculate the distance threshold:
        - "empirical": Uses the (1 - contamination) quantile of the calculated distances.
        - "chi2": Uses the theoretical Chi-squared distribution (conservative).
        - "calibrated_chi2": Theoretical Chi-squared threshold with a small dimension-based correction.
    random_state : int, default 42
        Seed for the random number generator used in waveform subsampling.
    min_clean_frac : float, default 0.50
        The minimum fraction of spikes that must remain in the "clean" cluster.
        If the threshold would remove more than this, the threshold is relaxed.
    relax_factor : float, default 0.5
        Factor used to reduce the contamination rate when relaxing the threshold.
    use_waveform_gate : bool, default True
        If True, spikes flagged as outliers by PCA distance are checked against the unit's
        average template. If they are sufficiently similar, they are "rescued" (not split).
    wf_threshold : float, default 0.9
        Cosine similarity threshold for rescuing spikes. Values closer to 1.0 are stricter.
    wf_cand_max : int | None, default 3000
        Maximum number of outlier candidates to check with the waveform gate to save time.
    wf_ms_before : float, default 1.0
        Time (ms) before the peak to include in the waveform comparison window.
    wf_ms_after : float, default 2.0
        Time (ms) after the peak to include in the waveform comparison window.
    wf_n_chans : int, default 20
        Number of best channels (based on PTP) to use for the waveform gate.
    wf_center : str, default "demean"
        Waveform centering method ("demean" or None).

    squeeze_all_outlier_to_new : bool, default True
        If True, the outliers are moved to a new unit. If False, they are effectively discarded from the sorting.
    return_details : bool, default False
        If True, returns a dictionary containing action details for each original unit.
    verbose : bool, default True
        If True, prints progress and split information to the console.


    Returns
    -------
    sorting_out : si.NumpySorting
        A new sorting object with split units.
    details : dict, optional
        Only returned if return_details is True. Maps original UIDs to split information.
    """

    rng = np.random.default_rng(random_state)

    # ==========================================================================
    #  1. Preparation
    # ==========================================================================
    sorting = analyzer.sorting
    fs = sorting.get_sampling_frequency()

    if not analyzer.has_extension("principal_components"):
        raise RuntimeError("Missing 'principal_components' extension.")
    
    pc_ext = analyzer.get_extension("principal_components")
    
    # Waveform rescue prep
    recording = getattr(analyzer, "recording", None)
    templates = None
    uid_to_uindex = None

    if use_waveform_gate:
        if recording is None: raise RuntimeError("Analyzer needs recording for waveform gate.")
        if not analyzer.has_extension("templates"):
            analyzer.compute("templates", operators=["average"])
        
        ext_tmpl = analyzer.get_extension("templates")
        try: templates = ext_tmpl.get_data(operator="average")
        except: templates = ext_tmpl.get_data()
        
        unit_ids_order = sorting.get_unit_ids()
        uid_to_uindex = {uid: i for i, uid in enumerate(unit_ids_order)}

    # ==========================================================================
    #  2. Helper Functions
    # ==========================================================================
    def _compute_stable_dist(features):
        scaler = StandardScaler()
        try:
            X_scaled = scaler.fit_transform(features)
        except ValueError:
            return None

        cov = np.cov(X_scaled, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6 

        try:
            U, s, Vt = np.linalg.svd(cov)
            s[s < 1e-8] = 1e-8 
            inv_cov = (U / s) @ Vt
        except np.linalg.LinAlgError:
            inv_cov = np.diag(1.0 / np.diag(cov))

        mu = np.mean(X_scaled, axis=0)
        diff = X_scaled - mu
        d2 = np.sum(diff @ inv_cov * diff, axis=1)
        return np.sqrt(d2)

    def _extract_waveforms(rec, frs, ch_ids, nb, na):
        frs = frs.astype(np.int64)
        ns = nb + na
        nc = len(ch_ids)
        W = np.empty((frs.size, ns, nc), dtype=np.float32)
        for i, f in enumerate(frs):
            if f - nb < 0 or f + na > rec.get_total_samples():
                W[i] = 0
                continue
            tr = rec.get_traces(start_frame=f-nb, end_frame=f+na, channel_ids=ch_ids, return_in_uV=True)
            if tr.shape[0] == ns: W[i] = tr
            else: W[i] = 0
        return W

    def _cosine_scores(W, tmpl_win, c_mode):
        X = W.reshape(W.shape[0], -1)
        t = tmpl_win.reshape(-1)
        if c_mode == "demean":
            X = X - X.mean(axis=1, keepdims=True)
            t = t - t.mean()
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        tn = t / (np.linalg.norm(t) + 1e-12)
        return (Xn @ tn)

    # ==========================================================================
    #  3. Main Loop
    # ==========================================================================
    orig_unit_ids = sorting.get_unit_ids()
    final_spike_trains = [] 
    details = {}

    for uid in orig_unit_ids:
        # --- Load Data ---
        try:
            if hasattr(pc_ext, "get_projections_one_unit"):
                pca_data = pc_ext.get_projections_one_unit(unit_id=uid) 
            else:
                pca_data = pc_ext.get_data(unit_id=uid)
            spike_frames = sorting.get_unit_spike_train(unit_id=uid)
        except Exception:
            if verbose: print(f"unit {uid}: Data load error")
            continue

        n_total = spike_frames.size
        
        # --- Pass-through (Small units) ---
        if n_total < 50 or pca_data is None:
            new_id = len(final_spike_trains)
            final_spike_trains.append(spike_frames)
            if verbose:
                print(f"unit {uid}: keep all ({n_total} spikes) -> unit {new_id} (small/no-pca)")
            details[uid] = {"new_clean_id": new_id, "action": "passthrough"}
            continue

        # Flatten PCA
        features_flat = pca_data.reshape(n_total, -1)
        n_dims = features_flat.shape[1]

        # --- A. Distance Gate ---
        d_dist = _compute_stable_dist(features_flat)
        
        if d_dist is None:
            new_id = len(final_spike_trains)
            final_spike_trains.append(spike_frames)
            details[uid] = {"new_clean_id": new_id, "action": "error_math"}
            continue

        # Threshold
        if threshold_mode == "empirical":
            thr_val = np.quantile(d_dist, 1.0 - contamination)
        elif threshold_mode == "chi2":
            thr_val = np.sqrt(chi2.ppf(1.0 - contamination, df=n_dims))
        elif threshold_mode == "calibrated_chi2":
            # Theoretical threshold with a small correction for finite samples/dimensions
            thr_val = np.sqrt(chi2.ppf(1.0 - contamination, df=n_dims) * (1.0 + 2.0 / n_dims))
        else:
            thr_val = np.quantile(d_dist, 1.0 - contamination)

        out_mask = d_dist > thr_val

        # Safety Relax
        n_out = out_mask.sum()
        if (n_total - n_out) / n_total < min_clean_frac:
            c_relax = max(1e-4, contamination * relax_factor)
            if threshold_mode == "chi2":
                 thr_val = np.sqrt(chi2.ppf(1.0 - c_relax, df=n_dims))
            elif threshold_mode == "calibrated_chi2":
                 thr_val = np.sqrt(chi2.ppf(1.0 - c_relax, df=n_dims) * (1.0 + 2.0 / n_dims))
            else:
                 thr_val = np.quantile(d_dist, 1.0 - c_relax)
            out_mask = d_dist > thr_val

        # --- B. Waveform Refinement (Rescue) ---
        wf_used = False
        if use_waveform_gate and templates is not None and uid in uid_to_uindex:
            cand_idx = np.flatnonzero(out_mask)
            if cand_idx.size > 0:
                nb = int(round(wf_ms_before * fs / 1000.0))
                na = int(round(wf_ms_after * fs / 1000.0))
                tmpl_full = templates[uid_to_uindex[uid]]
                ptp = np.ptp(tmpl_full, axis=0)
                ch_inds = np.argsort(ptp)[-wf_n_chans:][::-1] if wf_n_chans < ptp.size else np.arange(ptp.size)
                ch_ids_rec = recording.get_channel_ids()
                sel_ch_ids = [ch_ids_rec[i] for i in ch_inds]
                
                peak_idx = np.argmax(np.max(np.abs(tmpl_full[:, ch_inds]), axis=1))
                t_s, t_e = peak_idx - nb, peak_idx + na
                
                if t_s >= 0 and t_e <= tmpl_full.shape[0]:
                    tmpl_win = tmpl_full[t_s:t_e][:, ch_inds]
                    if wf_cand_max and cand_idx.size > wf_cand_max:
                        c_pick = rng.choice(cand_idx, size=int(wf_cand_max), replace=False)
                    else:
                        c_pick = cand_idx
                    
                    n_samp = recording.get_num_samples(segment_index=0)
                    valid_fr = (spike_frames[c_pick] - nb >= 0) & (spike_frames[c_pick] + na <= n_samp)
                    c_pick = c_pick[valid_fr]

                    if c_pick.size > 0:
                        W_cand = _extract_waveforms(recording, spike_frames[c_pick], sel_ch_ids, nb, na)
                        s_cand = _cosine_scores(W_cand, tmpl_win, wf_center)
                        rescued = c_pick[s_cand >= wf_threshold]
                        out_mask[rescued] = False
                        wf_used = True

        # --- C. Output Collection & Logging (Updated Format) ---
        cln_fr = spike_frames[~out_mask]
        nz_fr = spike_frames[out_mask]
        
        # Calculate next ID
        next_id = len(final_spike_trains)
        
        assigned_clean_id = None
        assigned_split_id = None
        
        # 1. Main Unit (Clean)
        if cln_fr.size > 0:
            final_spike_trains.append(cln_fr)
            assigned_clean_id = next_id
            next_id += 1 
        
        # 2. Split Unit (Outliers)
        if nz_fr.size > 0 and squeeze_all_outlier_to_new:
            final_spike_trains.append(nz_fr)
            assigned_split_id = next_id
            
        # --- Verbose Output ---
        if verbose:
            # Case 1: Split to new unit
            if assigned_split_id is not None:
                if assigned_clean_id is not None:
                    # unit 2 (N spikes) -> split to unit 516 (M spikes)
                    print(f"unit {assigned_clean_id} ({n_total} spikes) -> split to unit {assigned_split_id} ({nz_fr.size} spikes)")
                else:
                    # All moved
                    print(f"unit {assigned_split_id} ({n_total} spikes) -> all outliers from original")
            
            # Case 2: No split (Keep all or Discarded)
            else:
                if assigned_clean_id is not None:
                    if nz_fr.size == 0:
                        # unit 1 keep all
                        print(f"unit {assigned_clean_id} keep all")
                    else:
                        # Noise discarded (if squeeze_all_outlier_to_new=False)
                        print(f"unit {assigned_clean_id} ({n_total} spikes) -> kept ({cln_fr.size} spikes) | discarded {nz_fr.size} outliers")
                else:
                    # All spikes discarded
                    print(f"Original unit spikes ({n_total}) all discarded")

        details[uid] = {
            "clean_new_id": assigned_clean_id,
            "split_new_id": assigned_split_id,
            "n_clean": cln_fr.size, 
            "n_outlier": nz_fr.size
        }

    # ==========================================================================
    #  4. Construct Output Sorting
    # ==========================================================================
    spikes_list = []
    labels_list = []
    
    for new_id, times in enumerate(final_spike_trains):
        if times.size > 0:
            spikes_list.append(times)
            labels_list.append(np.full(times.size, new_id, dtype=np.int64))
            
    if spikes_list:
        times_concat = np.concatenate(spikes_list)
        labels_concat = np.concatenate(labels_list)
        order = np.argsort(times_concat)
        sorting_out = si.NumpySorting.from_samples_and_labels(
            times_concat[order], labels_concat[order], fs
        )
    else:
        sorting_out = si.NumpySorting.from_samples_and_labels(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64), fs
        )

    return (sorting_out, details) if return_details else sorting_out