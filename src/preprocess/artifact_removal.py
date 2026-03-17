import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
from joblib import Parallel, delayed, effective_n_jobs
import matplotlib.pyplot as plt
import spikeinterface.widgets as sw

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    _tqdm = None


class _NullProgress:
    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def _make_progress(*, total: int, desc: str, disable: bool):
    if disable or _tqdm is None:
        return _NullProgress()
    return _tqdm(total=total, desc=desc, unit="task", leave=False)



def detect_high_amplitude_artifacts(
    recording, 
    by_group: bool = True,             # If False, treat all channels as a single group
    estimate_windows: int = 50,        # Number of random windows used to estimate per-group noise scale
    estimate_window_s: float = 1.0,    # Duration (s) of each estimation window
    threshold_sigma: float = 20,       # Multiplier on noise scale (MAD→σ) to set amplitude threshold
    seed: int = 0,                     # RNG seed for reproducible window sampling
    chunk_s: float = 5.0,              # Chunk duration (s) for scanning the full recording
    dead_time_ms: float = 50.0,       # Refractory (ms) to merge nearby triggers from the same group
    n_jobs: int = -1                   # Number of parallel workers for threshold estimation and chunk scanning
):
    """
    Detect large-amplitude movement/lick artifacts and return trigger frames per group.
    All steps (threshold estimation, worker definition, parallel scanning, aggregation) are
    performed inside this single function.

    Parameters
    ----------
    recording : si.BaseRecording (or compatible)
        Single-segment recording extractor.
    by_group : bool, default True
        If True, detect artifacts independently for each group.
        If False, treat all channels as a single group.
    estimate_windows : int, default 50
        Number of random windows to estimate noise scale.
    estimate_window_s : float, default 1.0
        Duration (s) of each estimation window.
    threshold_sigma : float, default 20
        Multiplier on noise scale (σ estimated from MAD) to set absolute threshold.
    seed : int, default 0
        RNG seed for estimation window sampling.
    chunk_s : float, default 5.0
        Chunk size (s) for processing the full recording.
    dead_time_ms : float, default 100.0
        Refractory period (ms) to merge multiple detection triggers into one.
    n_jobs : int, default -1
        Number of parallel jobs (joblib) used for threshold estimation and chunk scanning.
        Uses backend="threading" internally.

    Returns
    -------
    final_triggers : dict
        Mapping {group_id: [frame_indices]}.
    """
    
    def _estimate_threshold_for_window_inner(gid, ch_inds, start, channel_ids):
        if ch_inds.size == 0:
            return int(gid), np.empty((0,), dtype=np.float32)

        sub = recording.select_channels(channel_ids=channel_ids[ch_inds])
        start_frame = int(start)
        end_frame = int(start_frame + W_est)
        # Load window and compute time-wise median across channels
        X_est = sub.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            return_in_uV=True,
        ).astype(np.float32).T
        m = np.median(np.abs(X_est), axis=1)
        return int(gid), np.asarray(m, dtype=np.float32)

    def _expand_run_ranges_with_dead_time(run_ranges, last_trig):
        if len(run_ranges) == 0:
            return np.empty((0,), dtype=np.int64), last_trig

        accepted_chunks = []
        step = int(dead_time_samp) + 1
        current_last = int(last_trig)
        for start, end in np.asarray(run_ranges, dtype=np.int64):
            first = max(int(start), current_last + step)
            if first > int(end):
                continue
            accepted = np.arange(first, int(end) + 1, step, dtype=np.int64)
            if accepted.size == 0:
                continue
            accepted_chunks.append(accepted)
            current_last = int(accepted[-1])
        if not accepted_chunks:
            return np.empty((0,), dtype=np.int64), current_last
        return np.concatenate(accepted_chunks), current_last

    def _build_cluster_summaries_from_run_ranges(run_ranges):
        run_ranges = np.asarray(run_ranges, dtype=np.int64)
        if run_ranges.size == 0:
            return []

        cluster_summaries = []
        cluster_start_idx = 0
        cluster_end = int(run_ranges[0, 1])
        step = int(dead_time_samp) + 1

        for idx in range(1, run_ranges.shape[0]):
            next_start = int(run_ranges[idx, 0])
            if next_start - cluster_end > dead_time_samp:
                cluster_runs = run_ranges[cluster_start_idx:idx]
                cluster_accepted, _ = _expand_run_ranges_with_dead_time(
                    cluster_runs,
                    int(cluster_runs[0, 0]) - step,
                )
                cluster_summaries.append(
                    (
                        int(cluster_runs[0, 0]),
                        cluster_runs.copy(),
                        cluster_accepted,
                    )
                )
                cluster_start_idx = idx
            cluster_end = max(cluster_end, int(run_ranges[idx, 1]))

        cluster_runs = run_ranges[cluster_start_idx:]
        cluster_accepted, _ = _expand_run_ranges_with_dead_time(
            cluster_runs,
            int(cluster_runs[0, 0]) - step,
        )
        cluster_summaries.append(
            (
                int(cluster_runs[0, 0]),
                cluster_runs.copy(),
                cluster_accepted,
            )
        )
        return cluster_summaries

    def _worker_scan_chunk_inner(core_start_frame, core_end_frame, thresholds_dict, group_to_inds, halo_samp):
        read_start_frame = max(0, int(core_start_frame) - int(halo_samp))
        read_end_frame = min(T, int(core_end_frame) + int(halo_samp))

        # Load traces in microvolts for the expanded time window [read_start_frame, read_end_frame)
        X = recording.get_traces(
            start_frame=read_start_frame,
            end_frame=read_end_frame,
            return_in_uV=True
        ).astype(np.float32)
        np.abs(X, out=X)

        # For each group, test the absolute median across its channels against the group threshold
        candidates = {}
        for gid, ch_inds in group_to_inds.items():
            if ch_inds.size == 0 or gid not in thresholds_dict:
                continue

            th_amp = thresholds_dict[gid]

            # X has already been converted to absolute amplitudes once per chunk.
            m = np.median(X[:, ch_inds], axis=1)

            violation_mask = m > th_amp
            if not np.any(violation_mask):
                continue

            # Collapse contiguous supra-threshold samples into runs in the worker.
            # This keeps scan output compact even when long noisy intervals exceed threshold.
            run_start_mask = violation_mask.copy()
            run_start_mask[1:] &= ~violation_mask[:-1]
            run_end_mask = violation_mask.copy()
            run_end_mask[:-1] &= ~violation_mask[1:]

            run_starts = np.flatnonzero(run_start_mask) + read_start_frame
            run_ends = np.flatnonzero(run_end_mask) + read_start_frame
            in_core = (run_ends >= core_start_frame) & (run_starts < core_end_frame)
            if np.any(in_core):
                run_starts = np.maximum(run_starts[in_core], core_start_frame)
                run_ends = np.minimum(run_ends[in_core], core_end_frame - 1)
                run_ranges = np.column_stack((run_starts, run_ends)).astype(np.int64, copy=False)
                candidates[gid] = _build_cluster_summaries_from_run_ranges(run_ranges)
        return candidates

    # ----------------------------------------------------
    
    assert recording.get_num_segments() == 1, "Single segment only supported"
    
    sf = recording.get_sampling_frequency()
    T = recording.get_total_samples()
    dead_time_samp = int(dead_time_ms * 1e-3 * sf)

    ch_ids = recording.get_channel_ids()
    
    if by_group:
        try:
            # Per-channel group ids (same order/length as ch_ids)
            group_ids = np.asarray(recording.get_property("group"))
        except Exception:
            # Fallback: all channels belong to group 0
            group_ids = np.zeros(len(ch_ids), dtype=int)
    else:
        # Treat all channels as a single group
        group_ids = np.zeros(len(ch_ids), dtype=int)
    
    unique_groups = np.unique(group_ids)
    group_to_inds = {int(g): np.where(group_ids == g)[0] for g in unique_groups}

    print(f"Detecting artifacts on {len(unique_groups)} groups (by_group={by_group})...")

    # -------------------- threshold estimation --------------------
    print(f"Estimating thresholds using backend='threading' (n_jobs={n_jobs})...")
    rng = np.random.default_rng(seed)
    W_est = int(estimate_window_s * sf)
    group_estimation_starts = {
        int(gid): rng.integers(0, max(1, T - W_est - 1), size=estimate_windows)
        for gid in unique_groups
    }
    threshold_tasks = [
        (int(gid), int(start))
        for gid in unique_groups
        for start in group_estimation_starts[int(gid)]
    ]
    threshold_batch_size = max(1, int(effective_n_jobs(n_jobs)))
    threshold_window_items = []
    threshold_progress = _make_progress(
        total=len(threshold_tasks),
        desc="Estimating thresholds",
        disable=False,
    )
    for batch_start in range(0, len(threshold_tasks), threshold_batch_size):
        task_batch = threshold_tasks[batch_start: batch_start + threshold_batch_size]
        batch_items = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_estimate_threshold_for_window_inner)(
                gid,
                group_to_inds[gid],
                start,
                ch_ids,
            )
            for gid, start in task_batch
        )
        threshold_window_items.extend(batch_items)
        threshold_progress.update(len(task_batch))
    threshold_progress.close()
    threshold_samples_by_group = {int(gid): [] for gid in unique_groups}
    for gid, abs_vals in threshold_window_items:
        if abs_vals.size > 0:
            threshold_samples_by_group[int(gid)].append(abs_vals)
    thresholds = {}
    for gid in unique_groups:
        pool_abs = threshold_samples_by_group[int(gid)]
        if not pool_abs:
            thresholds[int(gid)] = 1e6
            continue
        abs_vals = np.concatenate(pool_abs)
        sigma = 1.4826 * np.median(abs_vals)   # MAD → σ
        thresholds[int(gid)] = float(sigma * threshold_sigma)

    # -------------------- scanning --------------------
    scan_halo_samp = max(0, dead_time_samp)
    scan_halo_ms = (1000.0 * scan_halo_samp / float(sf)) if sf else 0.0
    print(
        "Scanning recording using "
        f"backend='threading' (n_jobs={n_jobs}, halo_ms={scan_halo_ms:.3f})..."
    )
    chunk_len = int(chunk_s * sf)
    chunks = []
    beg = 0
    while beg < T:
        end = min(T, beg + chunk_len)
        chunks.append((beg, end))
        beg = end

    # -------------------- aggregation & dead-time merge --------------------
    final_triggers = {int(g): [] for g in unique_groups}
    last_trig_by_group = {int(g): -dead_time_samp * 2 for g in unique_groups}
    scan_batch_size = max(1, int(effective_n_jobs(n_jobs)))
    scan_progress = _make_progress(
        total=len(chunks),
        desc="Scanning chunks",
        disable=False,
    )

    # Process chunk batches in chronological order so dead-time merging is preserved
    # even when detections straddle chunk boundaries.
    for batch_start in range(0, len(chunks), scan_batch_size):
        chunk_batch = chunks[batch_start: batch_start + scan_batch_size]
        batch_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_worker_scan_chunk_inner)(
                start, end, thresholds, group_to_inds, scan_halo_samp
            )
            for start, end in chunk_batch
        )
        scan_progress.update(len(chunk_batch))

        for res in batch_results:
            for gid, cluster_summaries in res.items():
                if len(cluster_summaries) == 0:
                    continue
                filtered = final_triggers[int(gid)]
                last_trig = last_trig_by_group[int(gid)]
                for cluster_idx, (cluster_start, cluster_runs, cluster_accepted) in enumerate(cluster_summaries):
                    if cluster_idx == 0 and int(cluster_start) - int(last_trig) <= dead_time_samp:
                        accepted, last_trig = _expand_run_ranges_with_dead_time(cluster_runs, last_trig)
                    else:
                        accepted = cluster_accepted
                        if accepted.size > 0:
                            last_trig = int(accepted[-1])
                    if accepted.size > 0:
                        filtered.extend(accepted.tolist())
                last_trig_by_group[int(gid)] = int(last_trig)
    scan_progress.close()

    total_artifacts = 0
    for gid in unique_groups:
        n_art = len(final_triggers[int(gid)])
        total_artifacts += n_art
        print(f"  Group {gid}: {n_art} artifacts detected")

    print(f"Done. Total artifacts detected: {total_artifacts}")
    return final_triggers


def remove_artifacts(
    recording_in: si.BaseRecording,
    artifact_per_group: dict[int, list[int]] or list[int],
    by_group: bool = True,         # If False, treat all channels as a single group
    ms_before: float = 0.5,        # Window start relative to trigger (ms; positive values look back in time)
    ms_after: float = 3.0,         # Window end relative to trigger (ms; positive values look forward in time)
    mode: str = "cubic",           # Interpolation strategy: 'cubic' | 'linear' | 'mean' (SI >= 0.98)
):
    """
    Apply `spre.remove_artifacts` independently per group (or globally), then stitch channels back together.

    Parameters
    ----------
    recording_in : si.BaseRecording
        Single-segment input recording extractor.
    artifact_per_group : dict[int, list[int]]
        Mapping group_id -> list of trigger frames (global timeline) to remove.
    by_group : bool, default True
        If True, apply removal independently for each group.
        If False, apply removal to all channels simultaneously using triggers from key 0.
    ms_before : float, default 0.5
        Duration (ms) before trigger to remove.
    ms_after : float, default 3.0
        Duration (ms) after trigger to remove.
    mode : str, default 'cubic'
        SI strategy for trace interpolation ('cubic', 'linear', or 'mean').

    Returns
    -------
    recording_clean : si.BaseRecording
        Cleaned sub-recording / aggregate recording.
    details : dict
        Execution summary per group.
    """
    assert recording_in.get_num_segments() == 1, "single-segment only"

    sf = recording_in.get_sampling_frequency()
    ch_ids = recording_in.get_channel_ids()

    # --- group assignment ---
    try:
        original_group_ids = np.asarray(recording_in.get_property("group"))
        if original_group_ids.shape[0] != len(ch_ids):
            raise ValueError("group length mismatch")
    except Exception:
        original_group_ids = np.zeros(len(ch_ids), dtype=int)

    if by_group:
        process_group_ids = original_group_ids
    else:
        process_group_ids = np.zeros(len(ch_ids), dtype=int)

    # Build group -> channel indices (preserve original channel order)
    group_to_inds = {int(g): np.where(process_group_ids == g)[0] for g in np.unique(process_group_ids)}
    # Determine deterministic group order as they appear along channels
    ordered_gids = sorted(group_to_inds.keys(), key=lambda g: (group_to_inds[g][0] if group_to_inds[g].size else 1e18))

    cleaned_subs = []
    details = {}

    for gid in ordered_gids:
        idx = group_to_inds[gid]
        if idx.size == 0:
            continue

        # Sub-recording for this group (keeps properties)
        gr_ch_ids = ch_ids[idx]
        sub_rec = recording_in.select_channels(channel_ids=gr_ch_ids)

        # Triggers for this group (frames). Deduplicate & sort.
        # If by_group=False, we expect triggers in key 0.
        if type(artifact_per_group) is list:
            trig = artifact_per_group
        else:
            trig = artifact_per_group.get(int(gid), [])
        if trig:
            trig = np.unique(np.asarray(trig, dtype=np.int64)).tolist()

        if trig:
            # Apply artifact removal on this group only
            sub_clean = spre.remove_artifacts(
                sub_rec,
                list_triggers=trig,     # frames on the full recording timeline
                ms_before=ms_before,
                ms_after=ms_after,
                mode=mode,
            )
            details[int(gid)] = {
                "n_triggers": len(trig),
                "ms_before": ms_before,
                "ms_after": ms_after,
                "mode": mode,
                "channel_ids": gr_ch_ids.tolist(),
            }
        else:
            # No triggers -> passthrough
            sub_clean = sub_rec
            details[int(gid)] = {
                "n_triggers": 0,
                "ms_before": ms_before,
                "ms_after": ms_after,
                "mode": mode,
                "channel_ids": gr_ch_ids.tolist(),
            }

        cleaned_subs.append(sub_clean)

    # Aggregate channels back (keeps original order by our group ordering + in-group order)
    recording_clean = si.aggregate_channels(cleaned_subs)
    original_ids = recording_in.get_channel_ids()
    recording_clean = recording_clean.select_channels(original_ids)
    # Re-apply original group property on the aggregated recording
    try:
        agg_ch_ids = recording_clean.get_channel_ids()
        gid_map = {int(ch_ids[i]): int(original_group_ids[i]) for i in range(len(ch_ids))}
        agg_groups = [gid_map[int(ch)] for ch in agg_ch_ids]
        if hasattr(recording_clean, "set_channel_property"):
            recording_clean.set_channel_property(agg_ch_ids, "group", agg_groups)
        elif hasattr(recording_clean, "set_property"):
            try:
                recording_clean.set_property("group", agg_groups, ids=agg_ch_ids)
            except TypeError:
                recording_clean.set_property("group", agg_groups)
    except Exception:
        pass

    return recording_clean, details

def plot_traces_around_artifact(
    recording_before,
    recording_after,
    ttl_times,
    *,
    before_ms: float = 5.0,
    after_ms: float = 5.0,
    units: str = "s",
    segment_index: int = 0,
    channel_ids=None,
    order_channel_by_depth: bool = True,
    show_channel_ids: bool = True,
    color_groups: bool = False,
    figsize: tuple = (12, 6),
    n_max: int | None = None,
):
    """
    Plots 'Before' and 'After' traces side-by-side for each TTL event.
    Ensures only common channels are plotted to avoid mismatches.
    """
    # 1. Basic validation
    assert recording_before.get_num_segments() > segment_index
    assert recording_after.get_num_segments() > segment_index

    sf = recording_before.get_sampling_frequency()
    total_samples = recording_after.get_num_frames(segment_index=segment_index)
    total_time = total_samples / sf

    # 2. Convert TTL times to seconds
    ttl_times = np.asarray(ttl_times, dtype=float)
    if units == "frames":
        ttl_times = ttl_times / sf

    # 3. Define time window
    t_before = before_ms / 1000.0
    t_after = after_ms / 1000.0

    # 4. Handle Channel IDs (Intersection logic)
    # Get IDs from both recordings to ensure existence
    ids_bef = recording_before.get_channel_ids()
    ids_aft = recording_after.get_channel_ids()
    
    # Find common channels
    common_ids = np.intersect1d(ids_bef, ids_aft)

    # Filter by user selection if provided
    if channel_ids is not None:
        target_ids = np.asarray(channel_ids)
        common_ids = np.intersect1d(common_ids, target_ids)
    
    if len(common_ids) == 0:
        raise ValueError("No common channels found between recordings.")

    # Sort IDs to ensure consistent plotting order
    # (Try numerical sort first, fallback to string sort)
    try:
        sorted_indices = np.argsort(common_ids.astype(int))
        final_channel_ids = common_ids[sorted_indices]
    except ValueError:
        final_channel_ids = np.sort(common_ids)

    # 5. Subsample TTLs if limit is set
    if n_max is not None and len(ttl_times) > n_max:
        idxs = np.linspace(0, len(ttl_times) - 1, n_max, dtype=int)
        ttl_times = ttl_times[idxs]

    # 6. Plotting Loop
    for t0 in ttl_times:
        tmin = max(0.0, t0 - t_before)
        tmax = min(total_time, t0 + t_after)
        
        if tmax <= tmin:
            continue

        # Create figure with 2 columns. sharey=False prevents scaling issues.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)

        plot_kwargs = dict(
            channel_ids=final_channel_ids,
            order_channel_by_depth=order_channel_by_depth,
            show_channel_ids=show_channel_ids,
            time_range=[tmin, tmax],
            segment_index=segment_index,
            color_groups=color_groups,
            backend='matplotlib',
        )

        # Plot Before (Left)
        sw.plot_traces(recording_before, ax=ax1, **plot_kwargs)
        ax1.set_title(f"Before (t={t0:.4f}s)")
        ax1.axvline(t0, color='r', linestyle='--', alpha=0.7, lw=1)

        # Plot After (Right)
        sw.plot_traces(recording_after, ax=ax2, **plot_kwargs)
        ax2.set_title(f"After (t={t0:.4f}s)")
        ax2.axvline(t0, color='r', linestyle='--', alpha=0.7, lw=1)

        plt.tight_layout()
        plt.show()
