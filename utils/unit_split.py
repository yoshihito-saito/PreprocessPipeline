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
    wf_template_max: int | None = 1000,
    wf_n_chans: int = 10,
    wf_center: str = "demean",
    # ---- Output Control ----
    squeeze_all_outlier_to_new: bool = True,
    min_spikes: int = 10,
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
        - "adaptive_chi2": Uses Chi-squared theory with data-driven scale correction based on median distance.
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
    wf_template_max : int | None, default 1000
        Maximum number of clean spikes to use for computing the reference template.
        Clean spikes are randomly sampled up to this number for efficiency.
        If None, all clean spikes are used (may be slow for large units).
    wf_n_chans : int, default 20
        Number of best channels (based on PTP) to use for the waveform gate.
    wf_center : str, default "demean"
        Waveform centering method ("demean" or None).

    squeeze_all_outlier_to_new : bool, default True
        If True, the outliers are moved to a new unit. If False, they are effectively discarded from the sorting.
    min_spikes : int, default 50
        Minimum number of spikes required to create a new outlier unit.
        If the number of outliers is less than this threshold, they are discarded instead of being assigned a new ID.
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
    analyzer_ms_before = None
    analyzer_ms_after = None

    if use_waveform_gate:
        if recording is None: raise RuntimeError("Analyzer needs recording for waveform gate.")
        if not analyzer.has_extension("templates"):
            analyzer.compute("templates", operators=["average"])
        
        ext_tmpl = analyzer.get_extension("templates")
        try: templates = ext_tmpl.get_data(operator="average")
        except: templates = ext_tmpl.get_data()
        
        # Get ms_before/after from analyzer's waveform extension
        if analyzer.has_extension("waveforms"):
            wf_ext = analyzer.get_extension("waveforms")
            # Get parameters from extension's params attribute
            if hasattr(wf_ext, 'params'):
                analyzer_ms_before = wf_ext.params.get('ms_before', 1.0)
                analyzer_ms_after = wf_ext.params.get('ms_after', 2.0)
            else:
                # Fallback: use default values
                analyzer_ms_before = 1.0
                analyzer_ms_after = 2.0
        else:
            # Fallback to default SpikeInterface values
            analyzer_ms_before = 1.0
            analyzer_ms_after = 2.0
        
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
        n_zero = 0
        for i, f in enumerate(frs):
            if f - nb < 0 or f + na > rec.get_total_samples():
                W[i] = 0
                n_zero += 1
                continue
            tr = rec.get_traces(start_frame=f-nb, end_frame=f+na, channel_ids=ch_ids, return_in_uV=True)
            if tr.shape[0] == ns: 
                W[i] = tr
            else: 
                W[i] = 0
                n_zero += 1
        return W, n_zero

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
        # --- Load spike train first (cheap), so we can early-discard tiny units ---
        try:
            spike_frames = sorting.get_unit_spike_train(unit_id=uid)
        except Exception:
            if verbose:
                print(f"unit {uid}: spike_train load error")
            details[uid] = {"clean_new_id": None, "action": "error_load_spike_train"}
            continue

        n_total = int(spike_frames.size)

        # --- Early drop (Small original units) ---
        # Rationale:
        # - Units with extremely few spikes can produce all-zero templates (e.g., spikes near segment edges -> no waveforms).
        # - Those all-zero templates crash Phy when it tries to find the best channel.
        # So, if the ORIGINAL unit has < min_spikes, we drop it entirely (no new ID).
        if n_total < min_spikes:
            if verbose:
                print(f"unit {uid} (orig): discarded ({n_total} spikes) [below min_spikes={min_spikes}]")
            details[uid] = {"clean_new_id": None, "action": "discard_small", "n_total": int(n_total)}
            continue

        # --- Load PCA only for sufficiently large units ---
        try:
            if hasattr(pc_ext, "get_projections_one_unit"):
                pca_data = pc_ext.get_projections_one_unit(unit_id=uid)
            else:
                pca_data = pc_ext.get_data(unit_id=uid)
        except Exception:
            pca_data = None

        # If PCA data is missing but unit is large enough, keep it as-is.
        if pca_data is None:
            new_id = len(final_spike_trains)
            final_spike_trains.append(spike_frames)
            if verbose:
                print(f"unit {uid} (orig) -> unit {new_id} (new): keep all ({n_total} spikes) [no-pca]")
            details[uid] = {"clean_new_id": new_id, "action": "passthrough_no_pca", "n_total": int(n_total)}
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
        elif threshold_mode == "adaptive_chi2":
            # Theoretical threshold with data-driven scale correction
            theo_thr_sq = chi2.ppf(1.0 - contamination, df=n_dims)
            
            # Estimate scale factor using median (robust to outliers)
            d2 = d_dist ** 2
            median_d2 = np.median(d2)
            expected_median = chi2.median(df=n_dims)
            scale_factor = median_d2 / expected_median
            
            # Apply scale correction
            thr_val = np.sqrt(theo_thr_sq * scale_factor)
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

        out_mask = d_dist > thr_val

        # Safety Relax
        n_out = out_mask.sum()
        if (n_total - n_out) / n_total < min_clean_frac:
            c_relax = max(1e-4, contamination * relax_factor)
            if threshold_mode == "adaptive_chi2":
                theo_thr_sq = chi2.ppf(1.0 - c_relax, df=n_dims)
                d2 = d_dist ** 2
                median_d2 = np.median(d2)
                expected_median = chi2.median(df=n_dims)
                scale_factor = median_d2 / expected_median
                thr_val = np.sqrt(theo_thr_sq * scale_factor)
            else:
                thr_val = np.quantile(d_dist, 1.0 - c_relax)
            out_mask = d_dist > thr_val

        # --- B. Waveform Refinement (Rescue) ---
        wf_used = False
        if use_waveform_gate and recording is not None:
            cand_idx = np.flatnonzero(out_mask)      # outlier candidates
            clean_idx = np.flatnonzero(~out_mask)    # clean spikes
            
            if cand_idx.size > 0 and clean_idx.size > 0:
                # Use analyzer's ms_before/after
                nb = int(round(analyzer_ms_before * fs / 1000.0))
                na = int(round(analyzer_ms_after * fs / 1000.0))
                
                # Select best channels based on pre-computed template PTP
                if templates is not None and uid in uid_to_uindex:
                    tmpl_full = templates[uid_to_uindex[uid]]
                    ptp = np.ptp(tmpl_full, axis=0)
                else:
                    # Fallback: use all channels
                    ptp = np.ones(recording.get_num_channels())
                ch_inds = np.argsort(ptp)[-wf_n_chans:][::-1] if wf_n_chans < ptp.size else np.arange(ptp.size)
                ch_ids_rec = recording.get_channel_ids()
                sel_ch_ids = [ch_ids_rec[i] for i in ch_inds]
                
                n_samp = recording.get_num_samples(segment_index=0)
                
                # --- Step 1: Build clean template from clean spikes ---
                # Sample clean spikes for template computation
                if wf_template_max and clean_idx.size > wf_template_max:
                    tmpl_pick = rng.choice(clean_idx, size=int(wf_template_max), replace=False)
                else:
                    tmpl_pick = clean_idx
                
                # Filter valid frames
                valid_tmpl = (spike_frames[tmpl_pick] - nb >= 0) & (spike_frames[tmpl_pick] + na <= n_samp)
                tmpl_pick = tmpl_pick[valid_tmpl]
                
                if tmpl_pick.size > 10:  # Need minimum spikes for reliable template
                    W_clean, _ = _extract_waveforms(recording, spike_frames[tmpl_pick], sel_ch_ids, nb, na)
                    
                    # Filter out zero-filled
                    valid_clean = np.any(W_clean != 0, axis=(1, 2))
                    W_clean = W_clean[valid_clean]
                    
                    if W_clean.shape[0] > 10:
                        # Compute mean template from clean spikes
                        clean_template = np.mean(W_clean, axis=0)
                        
                        if verbose:
                            print(f"  [Waveform Gate] Built template from {W_clean.shape[0]} clean spikes")
                        
                        # --- Step 2: Check ALL outlier candidates ---
                        valid_cand = (spike_frames[cand_idx] - nb >= 0) & (spike_frames[cand_idx] + na <= n_samp)
                        c_valid_idx = cand_idx[valid_cand]
                        
                        if c_valid_idx.size > 0:
                            W_cand, n_zero = _extract_waveforms(recording, spike_frames[c_valid_idx], sel_ch_ids, nb, na)
                            
                            # Filter out zero-filled
                            valid_wf = np.any(W_cand != 0, axis=(1, 2))
                            W_valid = W_cand[valid_wf]
                            c_valid = c_valid_idx[valid_wf]
                            
                            if verbose:
                                print(f"  [Waveform Gate] template shape: {clean_template.shape}, checking {c_valid.size} outliers")
                                print(f"  [Waveform Gate] zero-filled waveforms: {n_zero}/{c_valid_idx.size}")
                            
                            if c_valid.size > 0:
                                s_cand = _cosine_scores(W_valid, clean_template, wf_center)
                                rescued_mask = s_cand >= wf_threshold
                                rescued = c_valid[rescued_mask]
                                out_mask[rescued] = False
                                wf_used = True
                                
                                if verbose:
                                    n_rescued = rescued.size
                                    avg_sim = s_cand.mean()
                                    max_sim = s_cand.max()
                                    min_sim = s_cand.min()
                                    pct_rescued = (n_rescued / c_valid.size * 100) if c_valid.size > 0 else 0
                                    # Histogram of similarity scores
                                    n_low = (s_cand < 0.5).sum()
                                    n_mid = ((s_cand >= 0.5) & (s_cand < 0.8)).sum()
                                    n_high = (s_cand >= 0.8).sum()
                                    print(f"  [Waveform Gate] rescued {n_rescued}/{c_valid.size} ({pct_rescued:.1f}%) outliers")
                                    print(f"  [Waveform Gate] similarity: avg={avg_sim:.3f}, range=[{min_sim:.3f}, {max_sim:.3f}]")
                                    print(f"  [Waveform Gate] distribution: <0.5:{n_low}, 0.5-0.8:{n_mid}, >=0.8:{n_high}")
                            elif verbose:
                                print(f"  [Waveform Gate] no valid outlier waveforms to process")
                    elif verbose:
                        print(f"  [Waveform Gate] insufficient clean waveforms ({W_clean.shape[0]}) for template")
                elif verbose:
                    print(f"  [Waveform Gate] insufficient clean spikes ({tmpl_pick.size}) for template")

        # --- C. Output Collection & Logging (Updated Format) ---
        cln_fr = spike_frames[~out_mask]
        nz_fr = spike_frames[out_mask]
        
        assigned_clean_id = None
        assigned_split_id = None
        
        # 1. Main Unit (Clean)
        if cln_fr.size > 0:
            assigned_clean_id = len(final_spike_trains)
            final_spike_trains.append(cln_fr)
        
        # 2. Split Unit (Outliers) - only if meets minimum spike threshold
        if nz_fr.size >= min_spikes and squeeze_all_outlier_to_new:
            assigned_split_id = len(final_spike_trains)
            final_spike_trains.append(nz_fr)
            
        # --- Verbose Output ---
        if verbose:
            # Case 1: Split to new unit
            if assigned_split_id is not None:
                if assigned_clean_id is not None:
                    # unit 2 (orig) -> unit 2 (clean) | split to unit 516 (outliers)
                    print(f"unit {uid} (orig) -> unit {assigned_clean_id} (clean, {cln_fr.size} spikes) | split to unit {assigned_split_id} (outliers, {nz_fr.size} spikes)")
                else:
                    # All moved
                    print(f"unit {uid} (orig) -> unit {assigned_split_id} (all outliers, {n_total} spikes)")
            
            # Case 2: No split (Keep all or Discarded)
            else:
                if assigned_clean_id is not None:
                    if nz_fr.size == 0:
                        # unit 1 (orig) -> unit 1 (new): keep all
                        print(f"unit {uid} (orig) -> unit {assigned_clean_id} (new): keep all")
                    elif nz_fr.size < min_spikes:
                        # Outliers below min_spikes threshold
                        print(f"unit {uid} (orig) -> unit {assigned_clean_id} (new): kept ({cln_fr.size} spikes) | discarded {nz_fr.size} outliers (below min_spikes={min_spikes})")
                    else:
                        # Noise discarded (if squeeze_all_outlier_to_new=False)
                        print(f"unit {uid} (orig) -> unit {assigned_clean_id} (new): kept ({cln_fr.size} spikes) | discarded {nz_fr.size} outliers")
                else:
                    # All spikes discarded
                    print(f"unit {uid} (orig): all {n_total} spikes discarded")

        details[uid] = {
            "clean_new_id": assigned_clean_id,
            "split_new_id": assigned_split_id,
            "n_clean": cln_fr.size, 
            "n_outlier": nz_fr.size
        }

    # ==========================================================================
    #  4. Construct Output Sorting
    # ==========================================================================
    # Ensure contiguous unit IDs (0, 1, 2, ...) with no gaps
    # This is critical for Phy compatibility
    spikes_list = []
    labels_list = []
    
    contiguous_id = 0
    for times in final_spike_trains:
        if times.size > 0:
            spikes_list.append(times)
            labels_list.append(np.full(times.size, contiguous_id, dtype=np.int64))
            contiguous_id += 1
            
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
    
    if verbose:
        print(f"\n[Summary] Output: {contiguous_id} units with contiguous IDs (0 to {contiguous_id-1})")

    return (sorting_out, details) if return_details else sorting_out