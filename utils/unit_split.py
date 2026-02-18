import numpy as np
import scipy.linalg
import spikeinterface as si
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from joblib import Parallel, delayed

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
    # ---- Parallelism ----
    n_jobs: int = -1,
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
    n_jobs : int, default -1
        Number of parallel jobs for unit processing. -1 uses all available CPUs.
        Uses thread-based parallelism (safe for SpikeInterface recordings).

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

        mu = np.mean(X_scaled, axis=0)
        diff = X_scaled - mu  # (n_spikes, n_dims)

        try:
            # Cholesky: ~3x faster than SVD for positive-definite Î£.
            # dÂ² = ||Lâ»Â¹(x-Î¼)||Â²  avoids forming Î£â»Â¹ explicitly.
            L, lower = scipy.linalg.cho_factor(cov, lower=True, check_finite=False)
            z = scipy.linalg.solve_triangular(L, diff.T, lower=True, check_finite=False)  # (n_dims, n_spikes)
            d2 = np.einsum('ij,ij->j', z, z)  # sum of squares per spike
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            # Fallback: diagonal approximation
            diag_var = np.clip(np.diag(cov), 1e-8, None)
            d2 = np.sum((diff ** 2) / diag_var, axis=1)

        return np.sqrt(d2)

    def _extract_waveforms(rec, frs, ch_ids, nb, na):
        """
        Batch waveform extraction: sort frames, read minimal contiguous chunks,
        then slice. Much faster than per-spike get_traces calls.
        """
        frs = frs.astype(np.int64)
        n_spikes = frs.size
        ns = nb + na
        nc = len(ch_ids)
        n_samp = rec.get_total_samples()

        valid = (frs - nb >= 0) & (frs + na <= n_samp)
        W = np.zeros((n_spikes, ns, nc), dtype=np.float32)
        valid_frs = frs[valid]
        if valid_frs.size == 0:
            return W, n_spikes

        # Sort to enable sequential (cache-friendly) reads
        sort_order = np.argsort(valid_frs)
        sorted_frs = valid_frs[sort_order]

        # Merge nearby windows into larger chunks (gap < 2x window = read together)
        gap_thresh = ns * 2
        chunk_starts = [int(sorted_frs[0]) - nb]
        chunk_ends = [int(sorted_frs[0]) + na]
        for f in sorted_frs[1:]:
            start = int(f) - nb
            if start <= chunk_ends[-1] + gap_thresh:
                chunk_ends[-1] = max(chunk_ends[-1], int(f) + na)
            else:
                chunk_starts.append(start)
                chunk_ends.append(int(f) + na)

        # Read each chunk once
        chunk_cache: dict[int, np.ndarray] = {}
        for cs, ce in zip(chunk_starts, chunk_ends):
            try:
                chunk_cache[cs] = rec.get_traces(
                    start_frame=cs, end_frame=ce, channel_ids=ch_ids, return_in_uV=True
                )
            except Exception:
                pass

        # Slice each spike from its chunk using binary search (O(log n_chunks) per spike)
        valid_indices = np.flatnonzero(valid)
        inv_order = np.empty_like(sort_order)
        inv_order[sort_order] = np.arange(sort_order.size)

        chunk_starts_arr = np.array(chunk_starts, dtype=np.int64)
        for sorted_i in range(sorted_frs.size):
            f = int(sorted_frs[sorted_i])
            spike_start = f - nb
            # Binary search: rightmost chunk_start <= spike_start
            ci = int(np.searchsorted(chunk_starts_arr, spike_start, side='right')) - 1
            if ci < 0:
                continue
            cs = chunk_starts[ci]
            chunk = chunk_cache.get(cs)
            if chunk is not None:
                local_start = spike_start - cs
                local_end = local_start + ns
                if local_end <= chunk.shape[0]:
                    W[valid_indices[inv_order[sorted_i]]] = chunk[local_start:local_end]

        n_zero = int((~valid).sum())
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
    #  3. Per-unit processing task (parallelizable)
    # ==========================================================================
    def _process_unit(uid, rng_seed):
        rng_u = np.random.default_rng(rng_seed)
        log_lines = []

        # --- Load spike train first (cheap) ---
        try:
            spike_frames = sorting.get_unit_spike_train(unit_id=uid)
        except Exception:
            log_lines.append(f"unit {uid}: spike_train load error")
            return uid, None, None, {"action": "error_load_spike_train", "clean_new_id": None}, log_lines

        n_total = int(spike_frames.size)

        # --- Early drop (small original units) ---
        if n_total < min_spikes:
            log_lines.append(
                f"unit {uid} (orig): discarded ({n_total} spikes) [below min_spikes={min_spikes}]"
            )
            return uid, None, None, {"clean_new_id": None, "action": "discard_small", "n_total": n_total}, log_lines

        # --- Load PCA ---
        try:
            if hasattr(pc_ext, "get_projections_one_unit"):
                pca_data = pc_ext.get_projections_one_unit(unit_id=uid)
            else:
                pca_data = pc_ext.get_data(unit_id=uid)
        except Exception:
            pca_data = None

        if pca_data is None:
            log_lines.append(f"unit {uid} (orig): keep all ({n_total} spikes) [no-pca]")
            return uid, spike_frames, None, {"action": "passthrough_no_pca", "n_total": n_total}, log_lines

        features_flat = pca_data.reshape(n_total, -1)
        n_dims = features_flat.shape[1]

        # --- A. Distance Gate ---
        d_dist = _compute_stable_dist(features_flat)
        if d_dist is None:
            return uid, spike_frames, None, {"action": "error_math"}, log_lines

        if threshold_mode == "empirical":
            thr_val = np.quantile(d_dist, 1.0 - contamination)
        elif threshold_mode == "adaptive_chi2":
            theo_thr_sq = chi2.ppf(1.0 - contamination, df=n_dims)
            d2 = d_dist ** 2
            median_d2 = np.median(d2)
            expected_median = chi2.median(df=n_dims)
            scale_factor = median_d2 / expected_median
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
        if use_waveform_gate and recording is not None:
            cand_idx = np.flatnonzero(out_mask)
            clean_idx = np.flatnonzero(~out_mask)

            if cand_idx.size > 0 and clean_idx.size > 0:
                nb = int(round(analyzer_ms_before * fs / 1000.0))
                na = int(round(analyzer_ms_after * fs / 1000.0))

                if templates is not None and uid in uid_to_uindex:
                    tmpl_full = templates[uid_to_uindex[uid]]
                    ptp = np.ptp(tmpl_full, axis=0)
                else:
                    ptp = np.ones(recording.get_num_channels())
                ch_inds = np.argsort(ptp)[-wf_n_chans:][::-1] if wf_n_chans < ptp.size else np.arange(ptp.size)
                ch_ids_rec = recording.get_channel_ids()
                sel_ch_ids = [ch_ids_rec[i] for i in ch_inds]

                n_samp = recording.get_num_samples(segment_index=0)

                if wf_template_max and clean_idx.size > wf_template_max:
                    tmpl_pick = rng_u.choice(clean_idx, size=int(wf_template_max), replace=False)
                else:
                    tmpl_pick = clean_idx

                valid_tmpl = (spike_frames[tmpl_pick] - nb >= 0) & (spike_frames[tmpl_pick] + na <= n_samp)
                tmpl_pick = tmpl_pick[valid_tmpl]

                if tmpl_pick.size > 10:
                    W_clean, _ = _extract_waveforms(recording, spike_frames[tmpl_pick], sel_ch_ids, nb, na)
                    valid_clean = np.any(W_clean != 0, axis=(1, 2))
                    W_clean = W_clean[valid_clean]

                    if W_clean.shape[0] > 10:
                        clean_template = np.mean(W_clean, axis=0)

                        valid_cand = (spike_frames[cand_idx] - nb >= 0) & (spike_frames[cand_idx] + na <= n_samp)
                        c_valid_idx = cand_idx[valid_cand]

                        if c_valid_idx.size > 0:
                            W_cand, _ = _extract_waveforms(
                                recording, spike_frames[c_valid_idx], sel_ch_ids, nb, na
                            )
                            valid_wf = np.any(W_cand != 0, axis=(1, 2))
                            W_valid = W_cand[valid_wf]
                            c_valid = c_valid_idx[valid_wf]

                            if c_valid.size > 0:
                                s_cand = _cosine_scores(W_valid, clean_template, wf_center)
                                rescued_mask = s_cand >= wf_threshold
                                out_mask[c_valid[rescued_mask]] = False

        # --- C. Build results ---
        cln_fr = spike_frames[~out_mask]
        nz_fr = spike_frames[out_mask]
        outlier_fr = nz_fr if (nz_fr.size >= min_spikes and squeeze_all_outlier_to_new) else None

        if nz_fr.size == 0:
            log_lines.append(f"unit {uid} (orig): keep all ({n_total} spikes)")
        elif outlier_fr is not None:
            log_lines.append(
                f"unit {uid} (orig): keep {cln_fr.size} clean | split {nz_fr.size} outliers"
            )
        else:
            log_lines.append(
                f"unit {uid} (orig): keep {cln_fr.size} clean | discard {nz_fr.size} outliers"
            )

        detail = {"n_clean": int(cln_fr.size), "n_outlier": int(nz_fr.size)}
        return uid, cln_fr if cln_fr.size > 0 else None, outlier_fr, detail, log_lines

    # ==========================================================================
    #  4. Run in parallel (threads â€” safe for SI recordings via mmap)
    # ==========================================================================
    orig_unit_ids = sorting.get_unit_ids()
    seeds = rng.integers(0, 2**31, size=len(orig_unit_ids)).tolist()

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_unit)(uid, seed)
        for uid, seed in zip(orig_unit_ids, seeds)
    )

    # ==========================================================================
    #  5. Reconstruct output in original unit order
    # ==========================================================================
    final_spike_trains = []
    details = {}

    for uid, cln_fr, nz_fr, detail, log_lines in results:
        if verbose:
            for line in log_lines:
                print(line)

        assigned_clean_id = None
        assigned_split_id = None

        if cln_fr is not None:
            assigned_clean_id = len(final_spike_trains)
            final_spike_trains.append(cln_fr)

        if nz_fr is not None:
            assigned_split_id = len(final_spike_trains)
            final_spike_trains.append(nz_fr)

        detail["clean_new_id"] = assigned_clean_id
        detail["split_new_id"] = assigned_split_id
        details[uid] = detail

    # ==========================================================================
    #  6. Construct Output Sorting
    # ==========================================================================
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
        print(f"\n[Summary] Output: {contiguous_id} units (0 to {contiguous_id - 1})")

    return (sorting_out, details) if return_details else sorting_out

