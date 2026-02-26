"""State scoring pipeline compatible with neurocode SleepScoreMaster flow.

Algorithm overview (high level)
1) Inputs and prerequisites
   - Load `basename.lfp`/`.eeg` and session metadata.
   - Optional `pulses.intsPeriods` is used as `ignoretime`.
   - Bad channels are read from `session.channelTags.Bad.channels`.

2) EMG surrogate from LFP (`basename.EMGFromLFP.LFP.mat`)
   - Select EMG channels from spike groups (excluding bad channels).
   - Band-pass filter selected channels (high-frequency LFP range).
   - Compute pairwise correlations in sliding windows and average across pairs.
   - Save timestamps, EMG trace, channels, detector name, and sampling frequency.

3) SW/TH channel selection (`basename.SleepScoreLFP.LFP.mat`)
   - Evaluate candidate channels on downsampled LFP (MATLAB-style downsample).
   - SW metric: power-spectrum-slope (PSS) bimodality score.
   - TH metric: theta prominence above 1/f-like residual background.
   - Select best SW/TH channels and save raw selected traces (thLFP/swLFP).
   - Save channel-selection figure (`StateScoreFigures/*_SWTHChannels.jpg`).

4) Feature extraction for state clustering
   - Recompute SW/TH spectral features on selected channels.
   - Build:
     - `broadbandSlowWave` (NREM-related axis)
     - `thratio` (REM-related theta axis)
     - `EMG` (movement axis, interpolated to clustering timeline)
   - Apply transient masking and smoothing.
   - Remove `ignoretime` after smoothing to match MATLAB ordering.

5) Thresholding (histogram dip logic)
   - Estimate `swthresh`, `EMGthresh`, `THthresh` from histogram valleys.
   - Initial state assignment:
     - NREM: SW above threshold
     - REM: not NREM, low EMG, high theta
     - WAKE: remaining points

6) State post-processing (minimum interruptions)
   - Apply minimum-duration conversions (default 6 s; configurable).
   - Optional WAKE->REM suppression rule can be enabled/disabled.
   - Export `SleepState.states.mat` (`ints`, `idx`, `detectorinfo`).

7) Episodes and theta append
   - Build `SleepStateEpisodes.states.mat`.
   - Append THETA/nonTHETA intervals to SleepState.
   - Save summary figures (`*_SSResults.jpg`, `*_SSCluster2D.jpg`, `*_SSCluster3D.jpg`).

Outputs
- `basename.EMGFromLFP.LFP.mat`
- `basename.SleepScoreLFP.LFP.mat`
- `basename.SleepState.states.mat`
- `basename.SleepStateEpisodes.states.mat`
- `StateScoreFigures/*.jpg`
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal

from .metafile import PreprocessConfig


@dataclass
class StateScoreResult:
    emg_mat_path: Path
    sleepscore_lfp_mat_path: Path
    sleep_state_mat_path: Path
    sleep_state_episodes_mat_path: Path
    figure_paths: list[Path]


def _basename_from_basepath(basepath: Path) -> str:
    return basepath.name


def _safe_uint_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _resolve_parallel_jobs(job_kwargs: dict[str, Any] | None) -> int:
    if not isinstance(job_kwargs, dict):
        return 1
    raw = job_kwargs.get("n_jobs", 1)
    try:
        n_jobs = int(raw)
    except Exception:
        return 1
    cpu = max(1, int(os.cpu_count() or 1))
    if n_jobs == -1:
        return cpu
    if n_jobs < -1:
        return max(1, cpu + 1 + n_jobs)
    return max(1, n_jobs)


def _to_uint_vector(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values).reshape(-1)
    if vals.size == 0:
        return np.asarray([], dtype=np.uint8)
    vmax = int(np.max(vals))
    dt = _safe_uint_dtype(vmax)
    return np.rint(vals).astype(dt, copy=False).reshape(-1)


def _to_uint_intervals(intervals: np.ndarray) -> np.ndarray:
    arr = np.asarray(intervals, dtype=np.float64).reshape(-1, 2) if np.asarray(intervals).size else np.empty((0, 2))
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.uint8)
    vmax = int(np.nanmax(arr))
    dt = _safe_uint_dtype(vmax)
    return np.rint(arr).astype(dt, copy=False)


def _norm_to_range(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.full_like(arr, lo)
    return lo + (arr - mn) * (hi - lo) / (mx - mn)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0 or win <= 1:
        return arr
    win = max(1, min(win, arr.size))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(arr, kernel, mode="same")


def _find_bimodal_threshold(values: np.ndarray, start_bins: int = 12, max_bins: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.asarray([0.0]), np.asarray([0.0]), 0.5

    bins = int(start_bins)
    chosen_hist: np.ndarray | None = None
    chosen_centers: np.ndarray | None = None
    chosen_thr = float(np.median(x))
    while bins <= max_bins:
        hist, edges = np.histogram(x, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        peaks, props = signal.find_peaks(hist)
        if peaks.size >= 2:
            heights = props.get("peak_heights")
            if heights is None:
                heights = hist[peaks]
            order = np.argsort(heights)[-2:]
            sel = np.sort(peaks[order])
            left, right = int(sel[0]), int(sel[1])
            if right > left:
                valley = left + int(np.argmin(hist[left:right + 1]))
                chosen_hist = hist.astype(np.float64)
                chosen_centers = centers.astype(np.float64)
                chosen_thr = float(centers[valley])
                break
        chosen_hist = hist.astype(np.float64)
        chosen_centers = centers.astype(np.float64)
        bins += 1
    if chosen_hist is None or chosen_centers is None:
        return np.asarray([0.0]), np.asarray([0.0]), chosen_thr
    return chosen_hist, chosen_centers, chosen_thr


def _in_intervals(timestamps: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    ints = np.asarray(intervals, dtype=np.float64).reshape(-1, 2) if np.asarray(intervals).size else np.empty((0, 2))
    if ints.size == 0:
        return np.zeros(ts.shape, dtype=bool)
    out = np.zeros(ts.shape, dtype=bool)
    for start, stop in ints:
        out |= (ts >= float(start)) & (ts <= float(stop))
    return out


def _intervals_from_mask(mask: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if m.size == 0 or ts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    edges = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size - 1]
    if starts.size == 0 or ends.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    n = min(starts.size, ends.size)
    starts = starts[:n]
    ends = ends[:n]
    return np.column_stack((ts[starts], ts[ends])).astype(np.float64)


def _idx_to_int(states: np.ndarray, timestamps: np.ndarray, statenames: list[str]) -> dict[str, np.ndarray]:
    idx = np.asarray(states).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    out: dict[str, np.ndarray] = {}
    if idx.size == 0:
        for name in statenames:
            if name:
                out[f"{name}state"] = np.empty((0, 2), dtype=np.float64)
        return out

    for state_id, name in enumerate(statenames, start=1):
        if not name:
            continue
        mask = idx == state_id
        out[f"{name}state"] = _intervals_from_mask(mask, ts)
    return out


def _int_to_idx(ints: dict[str, np.ndarray], statenames: list[str], sf: float = 1.0) -> dict[str, Any]:
    all_ints: list[np.ndarray] = []
    for name in statenames:
        if not name:
            continue
        arr = np.asarray(ints.get(f"{name}state", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
        if arr.size:
            all_ints.append(arr)
    if all_ints:
        max_t = int(np.ceil(np.max(np.concatenate(all_ints, axis=0)[:, 1]) * sf))
    else:
        max_t = 0
    if max_t <= 0:
        return {
            "states": np.asarray([], dtype=np.uint8),
            "timestamps": np.asarray([], dtype=np.float64),
            "statenames": np.asarray(statenames, dtype=object).reshape(1, -1),
        }

    states = np.zeros((max_t,), dtype=np.uint8)
    for state_id, name in enumerate(statenames, start=1):
        if not name:
            continue
        arr = np.asarray(ints.get(f"{name}state", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
        for start, stop in arr:
            # MATLAB bz_INTtoIDX behavior:
            # stateints = round(INT*sf); stateints(stateints==0)=1; IDX(stateints(1):stateints(2))=state_id
            s1 = max(1, int(np.rint(start * sf)))
            e1 = min(max_t, int(np.rint(stop * sf)))
            if e1 >= s1:
                states[s1 - 1 : e1] = np.uint8(state_id)
    timestamps = np.arange(1, max_t + 1, dtype=np.float64) / float(sf)
    return {
        "states": states,
        "timestamps": timestamps,
        "statenames": np.asarray(statenames, dtype=object).reshape(1, -1),
    }


def _merge_intervals(intervals: np.ndarray, max_gap: float, min_duration: float) -> np.ndarray:
    arr = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    merged: list[list[float]] = [[float(arr[0, 0]), float(arr[0, 1])]]
    for start, stop in arr[1:]:
        if float(start) - merged[-1][1] <= float(max_gap):
            merged[-1][1] = max(merged[-1][1], float(stop))
        else:
            merged.append([float(start), float(stop)])
    out = np.asarray(merged, dtype=np.float64)
    keep = (out[:, 1] - out[:, 0]) >= float(min_duration)
    return out[keep, :]


def _subtract_intervals(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if aa.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if bb.size == 0:
        return aa

    out: list[list[float]] = []
    for start, stop in aa:
        cur = [(float(start), float(stop))]
        for bs, be in bb:
            next_cur: list[tuple[float, float]] = []
            for cs, ce in cur:
                if be <= cs or bs >= ce:
                    next_cur.append((cs, ce))
                    continue
                if bs > cs:
                    next_cur.append((cs, float(bs)))
                if be < ce:
                    next_cur.append((float(be), ce))
            cur = next_cur
        for cs, ce in cur:
            if ce > cs:
                out.append([cs, ce])
    if not out:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _matlab_datetime_like(date_text: str) -> np.ndarray:
    out = np.empty((1,), dtype=[("s0", "O"), ("s1", "O"), ("s2", "O"), ("arr", "O")])
    out["s0"][0] = np.asarray([], dtype=np.uint8)
    out["s1"][0] = np.asarray([], dtype=np.uint8)
    out["s2"][0] = np.asarray([], dtype=np.uint8)
    out["arr"][0] = np.asarray([date_text], dtype=object)
    return out


def _extract_bad_channels_1based(session_struct: dict[str, Any]) -> np.ndarray:
    try:
        bad = np.asarray(session_struct["channelTags"]["Bad"]["channels"]).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.uint16)
    bad = bad[np.isfinite(bad)]
    if bad.size == 0:
        return np.asarray([], dtype=np.uint16)
    return np.rint(bad).astype(_safe_uint_dtype(int(np.max(bad))), copy=False)


def _extract_spike_groups_1based(session_struct: dict[str, Any]) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    try:
        chans = session_struct["extracellular"]["spikeGroups"]["channels"]
    except Exception:
        return groups

    arr = np.asarray(chans, dtype=object)
    for item in arr.reshape(-1):
        vals = np.asarray(item).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        groups.append(np.rint(vals).astype(np.int64, copy=False))
    return groups


def _select_emg_channels(
    spike_groups_1based: list[np.ndarray],
    reject_channels_1based: set[int],
    n_channels: int,
) -> np.ndarray:
    usable_groups: list[np.ndarray] = []
    for grp in spike_groups_1based:
        keep = np.asarray([ch for ch in grp if int(ch) not in reject_channels_1based and 1 <= int(ch) <= n_channels])
        if keep.size > 0:
            usable_groups.append(keep.astype(np.int64, copy=False))

    if usable_groups:
        if len(usable_groups) > 1:
            picks = [int(grp[0]) for grp in usable_groups]
        else:
            grp = usable_groups[0]
            n_pick = min(5, grp.size)
            picks = [int(v) for v in grp[:n_pick]]
    else:
        picks = [ch for ch in range(1, n_channels + 1) if ch not in reject_channels_1based]
        picks = picks[: min(8, len(picks))]

    if not picks:
        picks = [1]
    return np.asarray(sorted(set(picks)), dtype=np.int64)


def _load_binary_lfp(path: Path, n_channels: int) -> np.ndarray:
    n_channels = int(n_channels)
    if n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {n_channels}")
    itemsize = np.dtype(np.int16).itemsize
    file_bytes = int(Path(path).stat().st_size)
    frame_bytes = n_channels * itemsize
    if file_bytes % frame_bytes != 0:
        raise ValueError(
            f"LFP file size {file_bytes} bytes not divisible by frame size {frame_bytes} bytes: {path}"
        )
    n_samples = file_bytes // frame_bytes
    return np.memmap(path, dtype=np.int16, mode="r", shape=(n_samples, n_channels), order="C")


def _load_binary_lfp_channels(path: Path, n_channels: int, channels_1based: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    ch = np.asarray(channels_1based, dtype=np.int64).reshape(-1)
    if ch.size == 0:
        lfp_mm = _load_binary_lfp(path, n_channels=n_channels)
        return np.empty((lfp_mm.shape[0], 0), dtype=np.int16), ch, {}
    ch = np.unique(ch)
    if np.any(ch < 1) or np.any(ch > int(n_channels)):
        raise ValueError(
            f"Requested channel out of range 1..{int(n_channels)}: {ch.tolist()}"
        )
    lfp_mm = _load_binary_lfp(path, n_channels=n_channels)
    ch0 = (ch - 1).astype(np.int64, copy=False)
    lfp_subset = np.asarray(lfp_mm[:, ch0], dtype=np.int16)
    channel_to_col = {int(c): i for i, c in enumerate(ch.tolist())}
    return lfp_subset, ch, channel_to_col


def _nearest_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    xn = np.asarray(x_new, dtype=np.float64).reshape(-1)
    if xx.size == 0 or yy.size == 0:
        return np.zeros_like(xn)
    idx = np.searchsorted(xx, xn, side="left")
    idx = np.clip(idx, 0, xx.size - 1)
    left = np.clip(idx - 1, 0, xx.size - 1)
    choose_left = np.abs(xx[left] - xn) < np.abs(xx[idx] - xn)
    out_idx = np.where(choose_left, left, idx)
    return yy[out_idx]


def _matlab_downsample(x: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(x)
    f = max(1, int(factor))
    if arr.ndim == 1:
        return arr[::f]
    return arr[::f, ...]


def _modzscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return 0.6745 * (arr - med) / mad


def _normtoint_modz(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return _modzscore(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    for c in range(arr.shape[1]):
        out[:, c] = _modzscore(arr[:, c])
    return out


def _matlab_smooth(x: np.ndarray, window: float) -> np.ndarray:
    win = max(1, int(np.rint(float(window))))
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0 or win <= 1:
        return arr
    win = max(1, min(win, arr.size))
    kernel = np.ones(win, dtype=np.float64)
    finite = np.isfinite(arr)
    num = np.convolve(np.where(finite, arr, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(np.float64), kernel, mode="same")
    out = np.full_like(arr, np.nan, dtype=np.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def _hist_counts_centers(x: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(x, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        centers = np.linspace(0.0, 1.0, int(max(1, bins)), dtype=np.float64)
        return np.zeros_like(centers), centers
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(max(1, bins)) + 1, dtype=np.float64)
    counts, _ = np.histogram(vals, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return counts.astype(np.float64), centers.astype(np.float64)


def _hartigans_dip_statistic(xpdf: np.ndarray) -> float:
    x = np.sort(np.asarray(xpdf, dtype=np.float64).reshape(-1))
    n = int(x.size)
    if n < 4:
        return 0.0
    if not np.isfinite(x[0]) or not np.isfinite(x[-1]) or x[-1] <= x[0]:
        return 0.0

    xsign = -np.sign(np.diff(np.diff(x)))
    posi = np.flatnonzero(xsign > 0)
    negi = np.flatnonzero(xsign < 0)
    if posi.size == 0 or negi.size == 0 or np.all(posi < np.min(negi)):
        return 0.0

    # 1-based indexing translation of Hartigan's AS217 algorithm
    mn = np.zeros((n + 1,), dtype=np.int64)
    mj = np.zeros((n + 1,), dtype=np.int64)
    gcm = np.zeros((n + 1,), dtype=np.int64)
    lcm = np.zeros((n + 1,), dtype=np.int64)

    fn = float(n)
    low = 1
    high = n
    dip = 1.0 / fn

    mn[1] = 1
    for j in range(2, n + 1):
        mn[j] = j - 1
        mnj = int(mn[j])
        mnmnj = int(mn[mnj])
        a = mnj - mnmnj
        b = j - mnj
        while not (mnj == 1 or ((x[j - 1] - x[mnj - 1]) * a < (x[mnj - 1] - x[mnmnj - 1]) * b)):
            mn[j] = mnmnj
            mnj = int(mn[j])
            mnmnj = int(mn[mnj])
            a = mnj - mnmnj
            b = j - mnj

    mj[n] = n
    for jk in range(1, n):
        k = n - jk
        mj[k] = k + 1
        mjk = int(mj[k])
        mjmjk = int(mj[mjk])
        a = mjk - mjmjk
        b = k - mjk
        while not (mjk == n or ((x[k - 1] - x[mjk - 1]) * a < (x[mjk - 1] - x[mjmjk - 1]) * b)):
            mj[k] = mjmjk
            mjk = int(mj[k])
            mjmjk = int(mj[mjk])
            a = mjk - mjmjk
            b = k - mjk

    iterate = True
    while iterate:
        ic = 1
        gcm[ic] = high
        igcm1 = int(gcm[ic])
        ic += 1
        gcm[ic] = int(mn[igcm1])
        while gcm[ic] > low:
            igcm1 = int(gcm[ic])
            ic += 1
            gcm[ic] = int(mn[igcm1])
        icx = ic

        ic = 1
        lcm[ic] = low
        lcm1 = int(lcm[ic])
        ic += 1
        lcm[ic] = int(mj[lcm1])
        while lcm[ic] < high:
            lcm1 = int(lcm[ic])
            ic += 1
            lcm[ic] = int(mj[lcm1])
        icv = ic

        ig = icx
        ih = icv
        ix = icx - 1
        iv = 2
        d = 0.0

        if icx != 2 or icv != 2:
            iterate_bp50 = True
            while iterate_bp50:
                igcmx = int(gcm[ix])
                lcmiv = int(lcm[iv])
                if not (igcmx > lcmiv):
                    lcmiv1 = int(lcm[iv - 1])
                    a = lcmiv - lcmiv1
                    b = igcmx - lcmiv1 - 1
                    den = fn * (x[lcmiv - 1] - x[lcmiv1 - 1])
                    dx = ((x[igcmx - 1] - x[lcmiv1 - 1]) * a / den - b / fn) if den != 0 else 0.0
                    ix -= 1
                    if not (dx < d):
                        d = dx
                        ig = ix + 1
                        ih = iv
                else:
                    lcmiv = int(lcm[iv])
                    igcm = int(gcm[ix])
                    igcm1 = int(gcm[ix + 1])
                    a = lcmiv - igcm1 + 1
                    b = igcm - igcm1
                    den = fn * (x[igcm - 1] - x[igcm1 - 1])
                    dx = (a / fn - ((x[lcmiv - 1] - x[igcm1 - 1]) * b) / den) if den != 0 else 0.0
                    iv += 1
                    if not (dx < d):
                        d = dx
                        ig = ix + 1
                        ih = iv - 1

                if ix < 1:
                    ix = 1
                if iv > icv:
                    iv = icv
                iterate_bp50 = gcm[ix] != lcm[iv]
        else:
            d = 1.0 / fn

        iterate = not (d < dip)
        if iterate:
            dl = 0.0
            if ig != icx:
                for j in range(ig, icx):
                    temp = 1.0 / fn
                    jb = int(gcm[j + 1])
                    je = int(gcm[j])
                    if je - jb > 1 and x[je - 1] != x[jb - 1]:
                        a = je - jb
                        const = a / (fn * (x[je - 1] - x[jb - 1]))
                        for jr in range(jb, je + 1):
                            b = jr - jb + 1
                            t = b / fn - (x[jr - 1] - x[jb - 1]) * const
                            if t > temp:
                                temp = t
                    if dl < temp:
                        dl = temp

            du = 0.0
            if ih != icv:
                for k in range(ih, icv):
                    temp = 1.0 / fn
                    kb = int(lcm[k])
                    ke = int(lcm[k + 1])
                    if ke - kb > 1 and x[ke - 1] != x[kb - 1]:
                        a = ke - kb
                        const = a / (fn * (x[ke - 1] - x[kb - 1]))
                        for kr in range(kb, ke + 1):
                            b = kr - kb - 1
                            t = (x[kr - 1] - x[kb - 1]) * const - b / fn
                            if t > temp:
                                temp = t
                    if du < temp:
                        du = temp

            dipnew = dl if dl >= du else du
            if dip < dipnew:
                dip = dipnew
            low = int(gcm[ig])
            high = int(lcm[ih])

    return 0.5 * float(dip)


def _find_peak_locs_for_threshold(
    hist: np.ndarray,
    *,
    mode: str,
    prepend_zero: bool = False,
) -> np.ndarray:
    hh = np.asarray(hist, dtype=np.float64).reshape(-1)
    if hh.size == 0:
        return np.asarray([], dtype=np.int64)
    src = np.r_[0.0, hh] if prepend_zero else hh
    pks, props = signal.find_peaks(src)
    if pks.size == 0:
        return np.asarray([], dtype=np.int64)
    if prepend_zero:
        pks = pks - 1
        pks = pks[(pks >= 0) & (pks < hh.size)]
        if pks.size == 0:
            return np.asarray([], dtype=np.int64)
    if mode == "leftmost2":
        return np.sort(pks[:2].astype(np.int64, copy=False))
    # MATLAB SW/TH behavior: NPeaks=2 + SortStr='descend'
    heights = props.get("peak_heights")
    if heights is None:
        heights = src[pks]
    order = np.argsort(heights)[-2:]
    return np.sort(pks[order].astype(np.int64, copy=False))


def _find_hist_dip_threshold_between(
    hist: np.ndarray,
    centers: np.ndarray,
    peak_locs: np.ndarray,
) -> float:
    hh = np.asarray(hist, dtype=np.float64).reshape(-1)
    cc = np.asarray(centers, dtype=np.float64).reshape(-1)
    locs = np.asarray(peak_locs, dtype=np.int64).reshape(-1)
    if hh.size == 0 or cc.size == 0:
        return 0.5
    if locs.size < 2:
        return float(np.nanmedian(cc))
    left = int(np.clip(locs[0], 0, hh.size - 1))
    right = int(np.clip(locs[1], 0, hh.size - 1))
    if right <= left:
        return float(cc[left])
    seg = -hh[left : right + 1]
    # MATLAB findpeaks_SleepScore on -hist with NPeaks=1, SortStr='descend':
    # for flat valleys, choose the earliest max location.
    if seg.size == 0 or not np.any(np.isfinite(seg)):
        dip_rel = int(np.argmin(hh[left : right + 1]))
    else:
        seg_max = np.nanmax(seg)
        candidates = np.flatnonzero(seg == seg_max)
        if candidates.size == 0:
            dip_rel = int(np.argmin(hh[left : right + 1]))
        else:
            dip_rel = int(candidates[0])
    dip_idx = int(np.clip(left + dip_rel, 0, cc.size - 1))
    return float(cc[dip_idx])


def _spectrogram_with_freq_vector(
    signal_1d: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    *,
    window_sec: float,
    dt_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal_1d, dtype=np.float64).reshape(-1)
    fvec = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if x.size == 0 or fvec.size == 0:
        return np.asarray([], dtype=np.float64), np.empty((fvec.size, 0), dtype=np.float64)
    nperseg = max(32, int(np.rint(float(window_sec) * float(fs))))
    nperseg = min(nperseg, x.size)
    if nperseg < 8:
        nperseg = x.size
    step = max(1, int(np.rint(float(dt_sec) * float(fs))))
    noverlap = int(np.clip(nperseg - step, 0, nperseg - 1))
    nfft = int(2 ** np.ceil(np.log2(max(256, nperseg * 2))))
    f, t, spec = signal.spectrogram(
        x,
        fs=float(fs),
        window=np.hamming(nperseg),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="spectrum",
        mode="complex",
    )
    if spec.size == 0 or t.size == 0:
        return np.asarray([], dtype=np.float64), np.empty((fvec.size, 0), dtype=np.float64)
    amp = np.empty((fvec.size, t.size), dtype=np.float64)
    mag = np.abs(spec)
    for i in range(t.size):
        amp[:, i] = np.interp(fvec, f, mag[:, i], left=np.nan, right=np.nan)
    return t.astype(np.float64), amp


def _compute_power_spectrum_slope_cached(
    *,
    cache: dict[tuple[Any, ...], dict[str, np.ndarray]] | None,
    cache_key: tuple[Any, ...] | None,
    lfp_1d: np.ndarray,
    timestamps: np.ndarray,
    fs: float,
    winsize: float,
    dt: float,
    frange: tuple[float, float],
    nfreqs: int,
    irasa: bool,
) -> dict[str, np.ndarray]:
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]
    pss = _compute_power_spectrum_slope(
        lfp_1d=lfp_1d,
        timestamps=timestamps,
        fs=fs,
        winsize=winsize,
        dt=dt,
        frange=frange,
        nfreqs=nfreqs,
        irasa=irasa,
    )
    if cache is not None and cache_key is not None:
        cache[cache_key] = pss
    return pss


def _compute_power_spectrum_slope(
    lfp_1d: np.ndarray,
    timestamps: np.ndarray,
    fs: float,
    *,
    winsize: float,
    dt: float,
    frange: tuple[float, float],
    nfreqs: int = 200,
    irasa: bool = True,
) -> dict[str, np.ndarray]:
    x = np.asarray(lfp_1d, dtype=np.float64).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if x.size == 0 or ts.size == 0:
        return {
            "slope": np.asarray([], dtype=np.float64),
            "intercept": np.asarray([], dtype=np.float64),
            "timestamps": np.asarray([], dtype=np.float64),
            "freqs": np.asarray([], dtype=np.float64),
            "amp": np.empty((0, 0), dtype=np.float64),
            "resid": np.empty((0, 0), dtype=np.float64),
            "irasa_smooth": np.empty((0, 0), dtype=np.float64),
            "sampling_rate": np.asarray([[0.0]], dtype=np.float64),
        }

    fmin = float(frange[0])
    fmax = float(frange[1])
    nfreqs_target = int(max(8, nfreqs))
    padding = 0
    freqs_work = np.logspace(np.log10(fmin), np.log10(fmax), nfreqs_target)
    valid = slice(0, nfreqs_target)
    if irasa:
        max_rescale = 2.9
        padding = int(np.floor((nfreqs_target / 2.0) * np.log10(max_rescale**2) / np.log10(fmax / fmin)))
        nfreqs_work = nfreqs_target + 2 * padding
        actual = 10 ** ((np.log10(fmax / fmin) * padding) / (nfreqs_target - 1))
        freqs_work = np.logspace(np.log10(fmin / actual), np.log10(fmax * actual), nfreqs_work)
        valid = slice(padding, nfreqs_work - padding)

    t_rel, amp_mag = _spectrogram_with_freq_vector(
        signal_1d=x,
        fs=fs,
        freqs=freqs_work,
        window_sec=winsize,
        dt_sec=dt,
    )
    if amp_mag.size == 0 or t_rel.size == 0:
        return {
            "slope": np.asarray([], dtype=np.float64),
            "intercept": np.asarray([], dtype=np.float64),
            "timestamps": np.asarray([], dtype=np.float64),
            "freqs": np.asarray([], dtype=np.float64),
            "amp": np.empty((0, 0), dtype=np.float64),
            "resid": np.empty((0, 0), dtype=np.float64),
            "irasa_smooth": np.empty((0, 0), dtype=np.float64),
            "sampling_rate": np.asarray([[0.0]], dtype=np.float64),
        }

    amp_log = np.log10(np.maximum(amp_mag, np.finfo(np.float64).eps)).T  # time x freq
    assumed_ts = np.arange(x.size, dtype=np.float64) / float(fs)
    t_abs = _nearest_interp(assumed_ts, ts, t_rel)

    if irasa:
        irasa_smooth = np.full_like(amp_log, np.nan)
        pad = int(max(1, padding))
        for j in range(int(valid.start), int(valid.stop)):
            left = np.arange(j - pad, j, dtype=np.int64)
            right = np.arange(j + 1, j + pad + 1, dtype=np.int64)
            inds = np.concatenate([left, right])
            inds = inds[(inds >= 0) & (inds < amp_log.shape[1])]
            if inds.size == 0:
                irasa_smooth[:, j] = amp_log[:, j]
            else:
                irasa_smooth[:, j] = np.nanmedian(amp_log[:, inds], axis=1)
        amp_use = amp_log[:, valid]
        fit_use = irasa_smooth[:, valid]
        resid = amp_use - fit_use
        freqs = freqs_work[valid]
        irasa_keep = fit_use
    else:
        amp_use = amp_log
        fit_use = amp_log
        resid = np.zeros_like(amp_use)
        freqs = freqs_work
        irasa_keep = np.empty((amp_use.shape[0], 0), dtype=np.float64)

    x_fit = np.log10(freqs)
    slope = np.empty((amp_use.shape[0],), dtype=np.float64)
    intercept = np.empty((amp_use.shape[0],), dtype=np.float64)
    for i in range(amp_use.shape[0]):
        y = fit_use[i, :]
        p = np.polyfit(x_fit, y, 1)
        slope[i] = float(p[0])
        intercept[i] = float(p[1])

    return {
        "slope": slope,
        "intercept": intercept,
        "timestamps": t_abs,
        "freqs": freqs,
        "amp": amp_use,
        "resid": resid,
        "irasa_smooth": irasa_keep,
        "sampling_rate": np.asarray([[1.0 / max(dt, 1e-12)]], dtype=np.float64),
    }


def _compute_emg_from_lfp(
    *,
    basepath: Path,
    basename: str,
    lfp_path: Path,
    session_struct: dict[str, Any],
    reject_channels_1based: np.ndarray,
    overwrite: bool,
    sampling_frequency: float = 2.0,
) -> tuple[dict[str, Any], Path]:
    out_path = basepath / f"{basename}.EMGFromLFP.LFP.mat"
    if out_path.exists() and not overwrite:
        loaded = loadmat(out_path, simplify_cells=True)
        if "EMGFromLFP" in loaded:
            return loaded["EMGFromLFP"], out_path

    n_channels = int(session_struct["extracellular"]["nChannels"])
    fs = float(session_struct["extracellular"]["srLfp"])
    reject_set = set(int(v) for v in np.asarray(reject_channels_1based).reshape(-1))
    spk_groups = _extract_spike_groups_1based(session_struct)
    xcorr_chs_1based = _select_emg_channels(spk_groups, reject_set, n_channels=n_channels)
    lfp_subset, _, _ = _load_binary_lfp_channels(
        lfp_path,
        n_channels=n_channels,
        channels_1based=xcorr_chs_1based,
    )
    sig = np.asarray(lfp_subset, dtype=np.float64)

    nyq = max(625.0, fs / 2.0)
    lo, hi = 300.0, max(301.0, nyq - 25.0)
    b, a = signal.butter(3, [lo / (fs / 2.0), min(0.999, hi / (fs / 2.0))], btype="bandpass")
    sig = signal.filtfilt(b, a, sig, axis=0)

    bin_scoot_s = 1.0 / float(sampling_frequency)
    step = max(1, int(np.rint(fs * bin_scoot_s)))
    half_window = step
    centers = np.arange(half_window, sig.shape[0] - half_window, step, dtype=np.int64)
    if centers.size == 0:
        centers = np.asarray([sig.shape[0] // 2], dtype=np.int64)

    n_bins = int(centers.size)
    emg_corr = np.zeros((n_bins,), dtype=np.float64)
    n_pairs = 0
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    sample_index = centers[:, None] + offsets[None, :]

    for j in range(sig.shape[1]):
        for k in range(j + 1, sig.shape[1]):
            x_seg = sig[sample_index, j]
            y_seg = sig[sample_index, k]
            x_seg = x_seg - x_seg.mean(axis=1, keepdims=True)
            y_seg = y_seg - y_seg.mean(axis=1, keepdims=True)
            den = np.sqrt(np.sum(x_seg * x_seg, axis=1) * np.sum(y_seg * y_seg, axis=1))
            num = np.sum(x_seg * y_seg, axis=1)
            corr = np.where(den > 0, num / den, 0.0)
            emg_corr += corr
            n_pairs += 1
    if n_pairs > 0:
        emg_corr /= float(n_pairs)

    emg = {
        "timestamps": (centers.astype(np.float64) / fs).reshape(-1, 1),
        "data": emg_corr.reshape(-1, 1),
        "channels": xcorr_chs_1based.astype(_safe_uint_dtype(int(np.max(xcorr_chs_1based)))).reshape(1, -1),
        "detectorName": "getEMGFromLFP",
        "samplingFrequency": np.asarray([[int(np.rint(sampling_frequency))]], dtype=np.uint8),
    }
    savemat(out_path, {"EMGFromLFP": emg}, do_compression=True)
    return emg, out_path


def _score_sw_candidate(
    lfp_ch: np.ndarray,
    fs: float,
    window_sec: float,
    smoothfact: float,
    ignoretime: np.ndarray,
    *,
    pss_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] | None = None,
    cache_key: tuple[Any, ...] | None = None,
) -> tuple[float, np.ndarray]:
    ts = np.arange(np.asarray(lfp_ch).size, dtype=np.float64) / float(fs)
    pss = _compute_power_spectrum_slope_cached(
        cache=pss_cache,
        cache_key=cache_key,
        lfp_1d=lfp_ch,
        timestamps=ts,
        fs=fs,
        winsize=float(window_sec),
        dt=float(window_sec),  # PickSWTH uses noverlap=0
        frange=(4.0, 90.0),
        nfreqs=100,
        irasa=True,
    )
    metric = np.asarray(pss["slope"], dtype=np.float64).reshape(-1)
    specdt = 1.0 / max(float(np.asarray(pss["sampling_rate"]).reshape(-1)[0]), 1e-12) if metric.size else 1.0
    metric = _matlab_smooth(metric, float(smoothfact) / max(specdt, 1e-12))
    metric = _norm_to_range(metric, 0.0, 1.0)
    if ignoretime.size:
        keep = ~_in_intervals(np.asarray(pss["timestamps"], dtype=np.float64).reshape(-1), ignoretime)
        metric = metric[keep]
    if metric.size < 5:
        return 0.0, metric
    metric_sorted = np.sort(metric[np.isfinite(metric)])
    score = _hartigans_dip_statistic(metric_sorted) if metric_sorted.size >= 4 else 0.0
    return score, metric


def _score_theta_candidate(
    lfp_ch: np.ndarray,
    fs: float,
    window_sec: float,
    smoothfact: float,
    ignoretime: np.ndarray,
    *,
    pss_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] | None = None,
    cache_key: tuple[Any, ...] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    ts = np.arange(np.asarray(lfp_ch).size, dtype=np.float64) / float(fs)
    pss = _compute_power_spectrum_slope_cached(
        cache=pss_cache,
        cache_key=cache_key,
        lfp_1d=lfp_ch,
        timestamps=ts,
        fs=fs,
        winsize=float(window_sec),
        dt=float(window_sec),  # PickSWTH uses noverlap=0
        frange=(2.0, 20.0),
        nfreqs=100,
        irasa=True,
    )
    freqs = np.asarray(pss["freqs"], dtype=np.float64).reshape(-1)
    resid = np.asarray(pss["resid"], dtype=np.float64)
    t = np.asarray(pss["timestamps"], dtype=np.float64).reshape(-1)
    if resid.size == 0 or freqs.size == 0:
        return 0.0, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    resid = np.maximum(resid, 0.0)
    th_mask = (freqs >= 5.0) & (freqs <= 10.0)
    if not np.any(th_mask):
        return 0.0, np.asarray([], dtype=np.float64), np.nanmean(resid, axis=0), freqs
    thratio = np.max(resid[:, th_mask], axis=1)
    specdt = 1.0 / max(float(np.asarray(pss["sampling_rate"]).reshape(-1)[0]), 1e-12)
    thratio = _matlab_smooth(thratio, float(smoothfact) / max(specdt, 1e-12))
    if ignoretime.size:
        keep = ~_in_intervals(t, ignoretime)
        thratio = thratio[keep]
    thratio = _norm_to_range(thratio, 0.0, 1.0)
    meanspec = np.nanmean(resid, axis=0)
    denom = float(np.nansum(meanspec))
    score = float(np.nansum(meanspec[th_mask]) / denom) if denom > 0 else 0.0
    return score, thratio, meanspec, freqs


def _spectrogram_to_target_freqs(
    lfp_ch: np.ndarray,
    fs: float,
    target_freqs: np.ndarray,
    window_sec: float,
    *,
    noverlap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = max(1e-6, float(window_sec) - float(noverlap_sec))
    return _spectrogram_with_freq_vector(
        signal_1d=np.asarray(lfp_ch, dtype=np.float64).reshape(-1),
        fs=float(fs),
        freqs=np.asarray(target_freqs, dtype=np.float64).reshape(-1),
        window_sec=float(window_sec),
        dt_sec=dt,
    )


def _build_sw_plot_payload(
    lfp_ch: np.ndarray,
    fs: float,
    window_sec: float,
    smoothfact: float,
    ignoretime: np.ndarray,
    *,
    pss_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] | None = None,
    cache_key: tuple[Any, ...] | None = None,
) -> dict[str, np.ndarray]:
    ts = np.arange(np.asarray(lfp_ch).size, dtype=np.float64) / float(fs)
    pss = _compute_power_spectrum_slope_cached(
        cache=pss_cache,
        cache_key=cache_key,
        lfp_1d=lfp_ch,
        timestamps=ts,
        fs=fs,
        winsize=float(window_sec),
        dt=float(window_sec),
        frange=(4.0, 90.0),
        nfreqs=200,
        irasa=True,
    )
    sw_freqs = np.asarray(pss["freqs"], dtype=np.float64).reshape(-1)
    t_spec = np.asarray(pss["timestamps"], dtype=np.float64).reshape(-1)
    fftspec = np.asarray(pss["amp"], dtype=np.float64).T  # freq x time (log-power)
    if t_spec.size == 0 or sw_freqs.size == 0 or fftspec.size == 0:
        sw_freqs = np.logspace(0.0, 2.0, 100, dtype=np.float64)
        return {
            "freqs": sw_freqs,
            "t_spec": np.asarray([], dtype=np.float64),
            "fftspec": np.empty((sw_freqs.size, 0), dtype=np.float64),
            "t_metric": np.asarray([], dtype=np.float64),
            "metric_plot": np.asarray([], dtype=np.float64),
            "mu": 0.0,
            "sig": 1.0,
        }
    mu = float(np.nanmean(fftspec))
    sig = float(np.nanstd(fftspec))
    broadband = np.asarray(pss["slope"], dtype=np.float64).reshape(-1)
    spec_dt = 1.0 / max(float(np.asarray(pss["sampling_rate"]).reshape(-1)[0]), 1e-12)
    broadband = _matlab_smooth(broadband, float(smoothfact) / max(spec_dt, 1e-12))
    t_metric = t_spec.copy()
    if ignoretime.size:
        keep = ~_in_intervals(t_metric, ignoretime)
        broadband = broadband[keep]
        t_metric = t_metric[keep]
    metric_plot = _norm_to_range(broadband, np.log2(sw_freqs[0]), np.log2(sw_freqs[-1]))
    return {
        "freqs": sw_freqs,
        "t_spec": t_spec,
        "fftspec": fftspec,
        "t_metric": t_metric,
        "metric_plot": metric_plot,
        "mu": mu,
        "sig": sig,
    }


def _build_th_plot_payload(
    lfp_ch: np.ndarray,
    fs: float,
    window_sec: float,
    smoothfact: float,
    ignoretime: np.ndarray,
) -> dict[str, np.ndarray]:
    th_freqs = np.logspace(np.log10(2.0), np.log10(20.0), 100, dtype=np.float64)
    t_spec, fftspec = _spectrogram_to_target_freqs(
        lfp_ch=lfp_ch,
        fs=fs,
        target_freqs=th_freqs,
        window_sec=window_sec,
        noverlap_sec=0.0,
    )
    if t_spec.size == 0:
        return {
            "freqs": th_freqs,
            "t_spec": np.asarray([], dtype=np.float64),
            "fftspec": np.empty((th_freqs.size, 0), dtype=np.float64),
            "t_metric": np.asarray([], dtype=np.float64),
            "metric_plot": np.asarray([], dtype=np.float64),
            "mu": 0.0,
            "sig": 1.0,
        }
    fftspec = np.maximum(fftspec, np.finfo(np.float64).eps)
    log_spec = np.log10(fftspec)
    mu = float(np.nanmean(log_spec))
    sig = float(np.nanstd(log_spec))
    th_mask = (th_freqs >= 5.0) & (th_freqs <= 10.0)
    th_power = np.sum(fftspec[th_mask, :], axis=0)
    all_power = np.sum(fftspec, axis=0)
    thratio = np.divide(th_power, np.maximum(all_power, np.finfo(np.float64).eps))
    spec_dt = float(np.median(np.diff(t_spec))) if t_spec.size > 1 else (1.0 / max(fs, 1e-12))
    smooth_win = max(1, int(np.rint(float(smoothfact) / max(spec_dt, 1e-12))))
    thratio = _moving_average(thratio, smooth_win)
    t_metric = t_spec.copy()
    if ignoretime.size:
        keep = ~_in_intervals(t_metric, ignoretime)
        thratio = thratio[keep]
        t_metric = t_metric[keep]
    metric_plot = _norm_to_range(thratio, np.log2(th_freqs[0]), np.log2(th_freqs[-1]))
    return {
        "freqs": th_freqs,
        "t_spec": t_spec,
        "fftspec": fftspec,
        "t_metric": t_metric,
        "metric_plot": metric_plot,
        "mu": mu,
        "sig": sig,
    }


def _save_swth_figure(
    *,
    basepath: Path,
    basename: str,
    histbins: np.ndarray,
    swhists: np.ndarray,
    sw_order: np.ndarray,
    sw_good_idx: int,
    th_freqs: np.ndarray,
    th_meanspec: np.ndarray,
    th_order: np.ndarray,
    th_good_idx: int,
    sw_plot: dict[str, np.ndarray],
    th_plot: dict[str, np.ndarray],
    sw_chan_id: int,
    th_chan_id: int,
) -> Path:
    fig_dir = basepath / "StateScoreFigures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"{basename}_SWTHChannels.jpg"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        out_path.touch()
        return out_path

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        height_ratios=[1.3, 1.3, 1.0, 1.0, 1.0],
        hspace=0.35,
        wspace=0.28,
    )
    ax_sw_spec = fig.add_subplot(gs[0:2, :])
    ax_th_spec = fig.add_subplot(gs[2, :], sharex=ax_sw_spec)
    ax_sw_hist = fig.add_subplot(gs[3:, 0])
    ax_th_all = fig.add_subplot(gs[3:, 1])

    # SW histogram panel: imagesc + thin traces + selected trace.
    if swhists.size:
        order = np.asarray(sw_order, dtype=np.int64).reshape(-1)
        panel = swhists[:, order].T
        ax_sw_hist.imshow(
            panel,
            aspect="auto",
            origin="lower",
            extent=[float(histbins[0]), float(histbins[-1]), 1, panel.shape[0]],
            interpolation="nearest",
        )
        n_ch = panel.shape[0]
        normhists = _norm_to_range(panel.T, 0.0, max(1.0, float(n_ch) * 0.6))
        for i in range(normhists.shape[1]):
            ax_sw_hist.plot(histbins, normhists[:, i], color=(0.9, 0.9, 0.9), linewidth=0.3)
        sel_sorted_idx = int(np.where(order == int(sw_good_idx))[0][0]) if np.any(order == int(sw_good_idx)) else 0
        ax_sw_hist.plot(histbins, normhists[:, sel_sorted_idx], color="k", linewidth=1.0)
        ax_sw_hist.set_ylim(1, max(1, n_ch))
    ax_sw_hist.set_ylabel("Channel #")
    ax_sw_hist.set_xlabel("SW weight")
    ax_sw_hist.set_title("SW Histogram: All Channels")

    # Theta meanspectrum panel.
    if th_meanspec.size:
        order = np.asarray(th_order, dtype=np.int64).reshape(-1)
        panel = th_meanspec[:, order].T
        ax_th_all.imshow(
            panel,
            aspect="auto",
            origin="lower",
            extent=[float(np.log2(th_freqs[0])), float(np.log2(th_freqs[-1])), 1, panel.shape[0]],
            interpolation="nearest",
        )
        n_ch = panel.shape[0]
        normspec = _norm_to_range(panel.T, 0.0, max(1.0, float(n_ch) * 0.6))
        for i in range(normspec.shape[1]):
            ax_th_all.plot(np.log2(th_freqs), normspec[:, i], color=(0.9, 0.9, 0.9), linewidth=0.3)
        sel_sorted_idx = int(np.where(order == int(th_good_idx))[0][0]) if np.any(order == int(th_good_idx)) else 0
        ax_th_all.plot(np.log2(th_freqs), normspec[:, sel_sorted_idx], color="k", linewidth=1.0)
        ax_th_all.set_ylim(1, max(1, n_ch))
    ax_th_all.set_ylabel("Channel #")
    ax_th_all.set_xlabel("f (Hz)")
    ax_th_all.set_title("TH Spectrum: All Channels")
    tick_x = np.asarray([2, 4, 8, 16], dtype=float)
    ax_th_all.set_xticks(np.log2(tick_x))
    ax_th_all.set_xticklabels([str(int(v)) for v in tick_x])

    # Selected SW channel spectrogram.
    sw_t_spec = np.asarray(sw_plot.get("t_spec", np.asarray([]))).reshape(-1)
    sw_freqs = np.asarray(sw_plot.get("freqs", np.asarray([]))).reshape(-1)
    sw_fftspec = np.asarray(sw_plot.get("fftspec", np.empty((0, 0))))
    if sw_fftspec.size and sw_t_spec.size and sw_freqs.size:
        im = ax_sw_spec.imshow(
            sw_fftspec,
            aspect="auto",
            origin="lower",
            extent=[float(sw_t_spec[0]), float(sw_t_spec[-1]), float(np.log2(sw_freqs[0])), float(np.log2(sw_freqs[-1]))],
            interpolation="nearest",
        )
        mu = float(sw_plot.get("mu", 0.0))
        sig = float(sw_plot.get("sig", 1.0))
        if np.isfinite(mu) and np.isfinite(sig) and sig > 0:
            im.set_clim(mu - 2.0 * sig, mu + 2.0 * sig)
        sw_t = np.asarray(sw_plot.get("t_metric", np.asarray([]))).reshape(-1)
        sw_metric = np.asarray(sw_plot.get("metric_plot", np.asarray([]))).reshape(-1)
        if sw_t.size and sw_metric.size:
            ax_sw_spec.plot(sw_t, sw_metric, "k", linewidth=0.4)
        tick_y = np.asarray([1, 2, 4, 8, 16, 32, 64], dtype=float)
        tick_y = tick_y[(tick_y >= sw_freqs[0]) & (tick_y <= sw_freqs[-1])]
        ax_sw_spec.set_yticks(np.log2(tick_y))
        ax_sw_spec.set_yticklabels([str(int(v)) for v in tick_y])
    ax_sw_spec.set_ylabel("f (Hz)")
    ax_sw_spec.set_title(f"SW Channel: {int(sw_chan_id)}")

    # Selected TH channel spectrogram.
    th_t_spec = np.asarray(th_plot.get("t_spec", np.asarray([]))).reshape(-1)
    th_freqs_plot = np.asarray(th_plot.get("freqs", np.asarray([]))).reshape(-1)
    th_fftspec = np.asarray(th_plot.get("fftspec", np.empty((0, 0))))
    if th_fftspec.size and th_t_spec.size and th_freqs_plot.size:
        im = ax_th_spec.imshow(
            np.log10(np.maximum(th_fftspec, np.finfo(np.float64).eps)),
            aspect="auto",
            origin="lower",
            extent=[float(th_t_spec[0]), float(th_t_spec[-1]), float(np.log2(th_freqs_plot[0])), float(np.log2(th_freqs_plot[-1]))],
            interpolation="nearest",
        )
        mu = float(th_plot.get("mu", 0.0))
        sig = float(th_plot.get("sig", 1.0))
        if np.isfinite(mu) and np.isfinite(sig) and sig > 0:
            im.set_clim(mu - 2.5 * sig, mu + 2.5 * sig)
        ax_th_spec.plot([th_t_spec[0], th_t_spec[-1]], [np.log2(5.0), np.log2(5.0)], "w", linewidth=0.8)
        ax_th_spec.plot([th_t_spec[0], th_t_spec[-1]], [np.log2(10.0), np.log2(10.0)], "w", linewidth=0.8)
        th_t = np.asarray(th_plot.get("t_metric", np.asarray([]))).reshape(-1)
        th_metric = np.asarray(th_plot.get("metric_plot", np.asarray([]))).reshape(-1)
        if th_t.size and th_metric.size:
            ax_th_spec.plot(th_t, th_metric, "k", linewidth=0.4)
        tick_y = np.asarray([2, 4, 8, 16], dtype=float)
        tick_y = tick_y[(tick_y >= th_freqs_plot[0]) & (tick_y <= th_freqs_plot[-1])]
        ax_th_spec.set_yticks(np.log2(tick_y))
        ax_th_spec.set_yticklabels([str(int(v)) for v in tick_y])
    ax_th_spec.set_ylabel("f (Hz)")
    ax_th_spec.set_title(f"Theta Channel: {int(th_chan_id)}")
    ax_th_spec.set_xlabel("t (s)")

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _compute_sleepscore_lfp(
    *,
    basepath: Path,
    basename: str,
    lfp_path: Path,
    session_struct: dict[str, Any],
    reject_channels_1based: np.ndarray,
    sw_channels_1based: np.ndarray | None,
    th_channels_1based: np.ndarray | None,
    ignoretime: np.ndarray,
    window_sec: float,
    smoothfact: float,
    overwrite: bool,
    save_files: bool,
    parallel_jobs: int = 1,
    pss_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    out_path = basepath / f"{basename}.SleepScoreLFP.LFP.mat"
    if out_path.exists() and not overwrite:
        loaded = loadmat(out_path, simplify_cells=True)
        if "SleepScoreLFP" in loaded:
            fig_path = basepath / "StateScoreFigures" / f"{basename}_SWTHChannels.jpg"
            return loaded["SleepScoreLFP"], out_path, fig_path

    n_channels = int(session_struct["extracellular"]["nChannels"])
    fs = float(session_struct["extracellular"]["srLfp"])

    reject_set = set(int(v) for v in np.asarray(reject_channels_1based).reshape(-1))
    usechannels = np.asarray([ch for ch in range(1, n_channels + 1) if ch not in reject_set], dtype=np.int64)
    if usechannels.size == 0:
        usechannels = np.asarray([1], dtype=np.int64)

    sw_cands = usechannels.copy() if sw_channels_1based is None or sw_channels_1based.size == 0 else sw_channels_1based
    th_cands = usechannels.copy() if th_channels_1based is None or th_channels_1based.size == 0 else th_channels_1based
    sw_cands = np.asarray([ch for ch in sw_cands if ch in usechannels], dtype=np.int64)
    th_cands = np.asarray([ch for ch in th_cands if ch in usechannels], dtype=np.int64)
    if sw_cands.size == 0:
        sw_cands = usechannels[:1]
    if th_cands.size == 0:
        th_cands = usechannels[:1]

    needed_channels = np.unique(np.concatenate([sw_cands, th_cands]).astype(np.int64, copy=False))
    lfp_subset, loaded_channels_1based, channel_to_col = _load_binary_lfp_channels(
        lfp_path,
        n_channels=n_channels,
        channels_1based=needed_channels,
    )
    if loaded_channels_1based.size == 0:
        raise RuntimeError("No LFP channels loaded for state scoring.")

    downsamplefactor = 5
    lfp_ds = _matlab_downsample(lfp_subset.astype(np.float64), downsamplefactor)
    fs_ds = fs / float(downsamplefactor)

    def _sw_eval(ch: int) -> tuple[float, np.ndarray]:
        col = channel_to_col[int(ch)]
        return _score_sw_candidate(
            lfp_ds[:, col],
            fs_ds,
            window_sec,
            smoothfact,
            ignoretime,
            pss_cache=None,
            cache_key=None,
        )

    def _th_eval(ch: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        col = channel_to_col[int(ch)]
        return _score_theta_candidate(
            lfp_ds[:, col],
            fs_ds,
            window_sec,
            smoothfact,
            ignoretime,
            pss_cache=None,
            cache_key=None,
        )

    sw_scores: list[float] = []
    sw_metric_vals: list[np.ndarray] = []
    sw_iter = [int(ch) for ch in np.asarray(sw_cands, dtype=np.int64).reshape(-1)]
    if int(parallel_jobs) > 1 and len(sw_iter) > 1:
        with ThreadPoolExecutor(max_workers=min(int(parallel_jobs), len(sw_iter))) as ex:
            sw_results = list(ex.map(_sw_eval, sw_iter))
    else:
        sw_results = [_sw_eval(ch) for ch in sw_iter]
    for score, vals in sw_results:
        sw_scores.append(float(score))
        sw_metric_vals.append(np.asarray(vals, dtype=np.float64).reshape(-1))

    th_scores: list[float] = []
    th_metric_vals: list[np.ndarray] = []
    th_meanspec_list: list[np.ndarray] = []
    th_freqs_ref: np.ndarray | None = None
    th_iter = [int(ch) for ch in np.asarray(th_cands, dtype=np.int64).reshape(-1)]
    if int(parallel_jobs) > 1 and len(th_iter) > 1:
        with ThreadPoolExecutor(max_workers=min(int(parallel_jobs), len(th_iter))) as ex:
            th_results = list(ex.map(_th_eval, th_iter))
    else:
        th_results = [_th_eval(ch) for ch in th_iter]
    for score, vals, meanspec, freqs in th_results:
        th_scores.append(float(score))
        th_metric_vals.append(np.asarray(vals, dtype=np.float64).reshape(-1))
        th_meanspec_list.append(np.asarray(meanspec, dtype=np.float64).reshape(-1))
        if th_freqs_ref is None and np.asarray(freqs).size:
            th_freqs_ref = np.asarray(freqs, dtype=np.float64).reshape(-1)

    sw_best_idx = int(np.argmax(np.asarray(sw_scores))) if sw_scores else 0
    th_best_idx = int(np.argmax(np.asarray(th_scores))) if th_scores else 0
    sw_ch = int(sw_cands[sw_best_idx])
    th_ch = int(th_cands[th_best_idx])

    histbins = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    sw_hists = np.zeros((histbins.size, sw_cands.size), dtype=np.float64)
    hist_edges = np.linspace(0.0, 1.0, histbins.size + 1, dtype=np.float64)
    for i, vals in enumerate(sw_metric_vals):
        vv = np.asarray(vals, dtype=np.float64).reshape(-1)
        vv = vv[np.isfinite(vv)]
        if vv.size == 0:
            continue
        vv = np.clip(vv, 0.0, 1.0)
        counts, _ = np.histogram(vv, bins=hist_edges)
        sw_hists[:, i] = counts.astype(np.float64, copy=False)

    th_freqs = th_freqs_ref if th_freqs_ref is not None else np.logspace(np.log10(2.0), np.log10(20.0), 100, dtype=np.float64)
    th_meanspec = np.zeros((th_freqs.size, th_cands.size), dtype=np.float64)
    for i, ms in enumerate(th_meanspec_list):
        if ms.size == th_freqs.size:
            th_meanspec[:, i] = ms

    sw_order = np.argsort(np.nan_to_num(np.asarray(sw_scores, dtype=np.float64), nan=-np.inf))
    th_order = np.argsort(np.nan_to_num(np.asarray(th_scores, dtype=np.float64), nan=-np.inf))

    sw_lfp = lfp_subset[:, channel_to_col[sw_ch]].astype(np.int16, copy=False).reshape(-1, 1)
    th_lfp = lfp_subset[:, channel_to_col[th_ch]].astype(np.int16, copy=False).reshape(-1, 1)
    t = (np.arange(sw_lfp.shape[0], dtype=np.float64) / fs).reshape(1, -1)

    sw_freq_list = np.logspace(0.5, 2.0, 100, dtype=np.float64).reshape(1, -1)
    params = {
        "SWfreqlist": sw_freq_list,
        "SWweights": "PSS",
        "SWWeightsName": "PSS",
        "Notch60Hz": np.asarray([[0]], dtype=np.uint8),
        "NotchUnder3Hz": np.asarray([[0]], dtype=np.uint8),
        "NotchHVS": np.asarray([[0]], dtype=np.uint8),
        "NotchTheta": np.asarray([[0]], dtype=np.uint8),
        "ignoretime": np.asarray(ignoretime, dtype=np.float64).reshape(-1, 2) if np.asarray(ignoretime).size else np.empty((0, 0), dtype=np.uint8),
        "window": np.asarray([[int(np.rint(window_sec))]], dtype=np.uint8),
        "smoothfact": np.asarray([[int(np.rint(smoothfact))]], dtype=np.uint8),
        "IRASA": np.asarray([[1]], dtype=np.uint8),
    }
    sleepscore_lfp = {
        "thLFP": th_lfp,
        "swLFP": sw_lfp,
        "THchanID": np.asarray([[th_ch]], dtype=_safe_uint_dtype(th_ch)),
        "SWchanID": np.asarray([[sw_ch]], dtype=_safe_uint_dtype(sw_ch)),
        "sf": np.asarray([[int(np.rint(fs))]], dtype=_safe_uint_dtype(int(np.rint(fs)))),
        "t": t,
        "params": params,
    }
    if save_files:
        savemat(out_path, {"SleepScoreLFP": sleepscore_lfp}, do_compression=True)
    sw_plot = _build_sw_plot_payload(
        lfp_ch=lfp_ds[:, channel_to_col[sw_ch]],
        fs=fs_ds,
        window_sec=window_sec,
        smoothfact=smoothfact,
        ignoretime=ignoretime,
        pss_cache=pss_cache,
        cache_key=("sw_plot", int(sw_ch), int(lfp_ds.shape[0]), float(fs_ds), float(window_sec), float(smoothfact)),
    )
    th_plot = _build_th_plot_payload(
        lfp_ch=lfp_ds[:, channel_to_col[th_ch]],
        fs=fs_ds,
        window_sec=window_sec,
        smoothfact=smoothfact,
        ignoretime=ignoretime,
    )
    fig_path = _save_swth_figure(
        basepath=basepath,
        basename=basename,
        histbins=histbins,
        swhists=sw_hists,
        sw_order=sw_order,
        sw_good_idx=sw_best_idx,
        th_freqs=th_freqs,
        th_meanspec=th_meanspec,
        th_order=th_order,
        th_good_idx=th_best_idx,
        sw_plot=sw_plot,
        th_plot=th_plot,
        sw_chan_id=sw_ch,
        th_chan_id=th_ch,
    )
    return sleepscore_lfp, out_path, fig_path


def _find_next_to_ints(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if aa.size == 0 or bb.size == 0:
        return np.zeros((aa.shape[0],), dtype=bool), np.zeros((aa.shape[0],), dtype=bool)
    next_to_right = np.zeros((aa.shape[0],), dtype=bool)
    next_to_left = np.zeros((aa.shape[0],), dtype=bool)
    for i, (s, e) in enumerate(aa):
        next_to_right[i] = np.any(np.abs(bb[:, 0] - e) <= 1.0)
        next_to_left[i] = np.any(np.abs(bb[:, 1] - s) <= 1.0)
    return next_to_right, next_to_left


def _suppress_wake_to_rem_transitions(
    states: np.ndarray,
    timestamps: np.ndarray,
    *,
    min_wake_before_rem_secs: float = 0.0,
) -> np.ndarray:
    arr = np.asarray(states, dtype=np.uint8).reshape(-1).copy()
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return arr
    if ts.size != arr.size:
        ts = np.arange(arr.size, dtype=np.float64)
    min_wake = float(max(0.0, min_wake_before_rem_secs))
    i = 1
    while i < arr.size:
        if arr[i - 1] == 1 and arr[i] == 5:
            w_start = i - 1
            while w_start - 1 >= 0 and arr[w_start - 1] == 1:
                w_start -= 1
            wake_dur = float(ts[i - 1] - ts[w_start]) if i - 1 >= w_start else 0.0
            if wake_dur < min_wake:
                i += 1
                continue
            j = i + 1
            while j < arr.size and arr[j] == 5:
                j += 1
            arr[i:j] = 1
            i = j
            continue
        i += 1
    return arr


def _compute_sleep_state(
    *,
    basepath: Path,
    basename: str,
    sleepscore_lfp: dict[str, Any],
    emg: dict[str, Any],
    ignoretime: np.ndarray,
    sticky_trigger: bool,
    window_sec: float,
    smoothfact: float,
    reject_channels_1based: np.ndarray,
    sw_channels_1based: np.ndarray | None,
    th_channels_1based: np.ndarray | None,
    state_ignore_manual: bool,
    state_save_lfp_mat: bool,
    emg_th_alpha: float,
    min_state_length: float,
    block_wake_to_rem: bool,
    overwrite: bool,
    pss_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], Path]:
    out_path = basepath / f"{basename}.SleepState.states.mat"
    if out_path.exists() and not overwrite:
        loaded = loadmat(out_path, simplify_cells=True)
        if "SleepState" in loaded:
            return loaded["SleepState"], out_path

    sw_lfp = np.asarray(sleepscore_lfp["swLFP"], dtype=np.float64).reshape(-1)
    th_lfp = np.asarray(sleepscore_lfp["thLFP"], dtype=np.float64).reshape(-1)
    sf = float(np.asarray(sleepscore_lfp["sf"]).reshape(-1)[0])
    t_full = np.asarray(sleepscore_lfp["t"], dtype=np.float64).reshape(-1)
    sw_chan_id = int(np.asarray(sleepscore_lfp["SWchanID"]).reshape(-1)[0])
    th_chan_id = int(np.asarray(sleepscore_lfp["THchanID"]).reshape(-1)[0])

    if int(np.rint(sf)) == 1250:
        downsamplefactor = 5
    elif int(np.rint(sf)) == 250:
        downsamplefactor = 1
    elif int(np.rint(sf)) == 1000:
        downsamplefactor = 4
    else:
        downsamplefactor = 1
    sf_lfp = sf / float(downsamplefactor)
    sw_d = _matlab_downsample(sw_lfp, downsamplefactor)
    th_d = _matlab_downsample(th_lfp, downsamplefactor)
    t_d = _matlab_downsample(t_full, downsamplefactor)
    if t_d.size == 0:
        t_d = np.arange(sw_d.size, dtype=np.float64) / max(sf_lfp, 1e-12)

    dt_spec = max(1e-6, float(window_sec) - 1.0)
    pss_sw = _compute_power_spectrum_slope_cached(
        cache=pss_cache,
        cache_key=("sleep_state_sw", int(sw_chan_id), int(sw_d.size), float(sf_lfp), float(window_sec), float(dt_spec)),
        lfp_1d=sw_d,
        timestamps=t_d,
        fs=sf_lfp,
        winsize=float(window_sec),
        dt=dt_spec,
        frange=(4.0, 90.0),
        nfreqs=200,
        irasa=True,
    )
    t_clus = np.asarray(pss_sw["timestamps"], dtype=np.float64).reshape(-1)
    sw_freqs = np.asarray(pss_sw["freqs"], dtype=np.float64).reshape(-1)
    sw_amp = np.asarray(pss_sw["amp"], dtype=np.float64)  # time x freq (log10 power)
    swFFTspec = np.power(10.0, sw_amp.T)  # freq x time
    broadband = -np.asarray(pss_sw["slope"], dtype=np.float64).reshape(-1)
    specdt_sw = 1.0 / max(float(np.asarray(pss_sw["sampling_rate"]).reshape(-1)[0]), 1e-12)
    z_sw = _normtoint_modz(sw_amp)
    totz_sw = _normtoint_modz(np.abs(np.nansum(z_sw, axis=1)))
    badtimes = np.flatnonzero(totz_sw > 3.0)
    if badtimes.size:
        broadband[badtimes] = np.nan
    broadband = _matlab_smooth(broadband, float(smoothfact) / max(specdt_sw, 1e-12))

    pss_th = _compute_power_spectrum_slope_cached(
        cache=pss_cache,
        cache_key=("sleep_state_th", int(th_chan_id), int(th_d.size), float(sf_lfp), float(window_sec), float(dt_spec)),
        lfp_1d=th_d,
        timestamps=t_d,
        fs=sf_lfp,
        winsize=float(window_sec),
        dt=dt_spec,
        frange=(2.0, 20.0),
        nfreqs=200,
        irasa=True,
    )
    t_thclu = np.asarray(pss_th["timestamps"], dtype=np.float64).reshape(-1)
    th_freqs = np.asarray(pss_th["freqs"], dtype=np.float64).reshape(-1)
    th_amp = np.asarray(pss_th["amp"], dtype=np.float64)  # time x freq (log10 power)
    thFFTspec = np.array(pss_th["resid"], dtype=np.float64, copy=True).T  # freq x time
    thFFTspec[thFFTspec < 0] = 0.0
    IRASAsmooth_th = np.asarray(pss_th["irasa_smooth"], dtype=np.float64).T
    thFFTspec_raw = np.power(10.0, th_amp.T)
    z_th = _normtoint_modz(th_amp)
    totz_th = _modzscore(np.abs(np.nansum(z_th, axis=1)))
    badtimes_th = np.flatnonzero(totz_th > 3.0)
    th_mask = (th_freqs >= 5.0) & (th_freqs <= 10.0)
    if np.any(th_mask) and thFFTspec.size:
        thratio = np.max(thFFTspec[th_mask, :], axis=0)
    else:
        thratio = np.zeros((t_thclu.size,), dtype=np.float64)
    if badtimes_th.size:
        bad_idx = badtimes_th[badtimes_th < thratio.size]
        thratio[bad_idx] = np.nan
    specdt_th = 1.0 / max(float(np.asarray(pss_th["sampling_rate"]).reshape(-1)[0]), 1e-12)
    thratio = _matlab_smooth(thratio, float(smoothfact) / max(specdt_th, 1e-12))

    emg_t = np.asarray(emg["timestamps"], dtype=np.float64).reshape(-1)
    emg_v = np.asarray(emg["data"], dtype=np.float64).reshape(-1)
    emg_sf = float(np.asarray(emg["samplingFrequency"], dtype=np.float64).reshape(-1)[0]) if np.asarray(emg["samplingFrequency"]).size else 2.0
    dt_emg = 1.0 / max(emg_sf, 1e-12)
    emg_smooth = _matlab_smooth(emg_v, float(smoothfact) / max(dt_emg, 1e-12))

    if ignoretime.size or badtimes.size or badtimes_th.size:
        ignore_idx = _in_intervals(t_clus, ignoretime) | ~np.isfinite(broadband) | ~np.isfinite(thratio)
        broadband = broadband[~ignore_idx]
        thratio = thratio[~ignore_idx]
        t_clus = t_clus[~ignore_idx]
        if t_thclu.size == ignore_idx.size:
            t_thclu = t_thclu[~ignore_idx]

    if emg_t.size:
        pr_emg = (t_clus < emg_t[0]) | (t_clus > emg_t[-1])
        broadband = broadband[~pr_emg]
        thratio = thratio[~pr_emg]
        t_clus = t_clus[~pr_emg]

    emg_interp = _nearest_interp(emg_t, emg_smooth, t_clus)
    broadband = _norm_to_range(broadband, 0.0, 1.0)
    thratio = _norm_to_range(thratio, 0.0, 1.0)
    emg_interp = _norm_to_range(emg_interp, 0.0, 1.0)

    # MATLAB-like histogram/valley thresholding
    numpeaks = 1
    numbins = 12
    swhist = np.asarray([0.0], dtype=np.float64)
    swhistbins = np.asarray([0.0], dtype=np.float64)
    sw_locs = np.asarray([], dtype=np.int64)
    while numpeaks != 2 and numbins <= 200:
        swhist, swhistbins = _hist_counts_centers(broadband, numbins)
        sw_locs = _find_peak_locs_for_threshold(swhist, mode="top2", prepend_zero=False)
        numpeaks = int(sw_locs.size)
        numbins += 1
    swthresh = _find_hist_dip_threshold_between(swhist, swhistbins, sw_locs)
    nrem_times = broadband > swthresh

    numpeaks = 1
    numbins = 12
    emghist = np.asarray([0.0], dtype=np.float64)
    emghistbins = np.asarray([0.0], dtype=np.float64)
    emg_locs = np.asarray([], dtype=np.int64)
    while numpeaks != 2 and numbins <= 200:
        emghist, emghistbins = _hist_counts_centers(emg_interp, numbins)
        # MATLAB parity: findpeaks([0 EMGhist],'NPeaks',2) without SortStr
        # -> first two peaks by occurrence order (after the prepended 0).
        emg_locs = _find_peak_locs_for_threshold(emghist, mode="leftmost2", prepend_zero=True)
        numpeaks = int(emg_locs.size)
        numbins += 1
    emgthresh = _find_hist_dip_threshold_between(emghist, emghistbins, emg_locs)
    emgthresh = float(np.clip(emgthresh * float(emg_th_alpha), 0.0, 1.0))
    mov_times = (broadband < swthresh) & (emg_interp > emgthresh)

    numpeaks = 1
    numbins = 12
    thhist = np.asarray([0.0], dtype=np.float64)
    thhistbins = np.asarray([0.0], dtype=np.float64)
    th_locs = np.asarray([], dtype=np.int64)
    th_for_hist = thratio[~mov_times] if np.any(~mov_times) else thratio
    while numpeaks != 2 and numbins <= 25:
        thhist, thhistbins = _hist_counts_centers(th_for_hist, numbins)
        th_locs = _find_peak_locs_for_threshold(thhist, mode="top2", prepend_zero=False)
        numpeaks = int(th_locs.size)
        numbins += 1
    if numpeaks != 2:
        numbins = 12
        th_for_hist = thratio[(~nrem_times) & (~mov_times)] if np.any((~nrem_times) & (~mov_times)) else thratio
        while numpeaks != 2 and numbins <= 25:
            thhist, thhistbins = _hist_counts_centers(th_for_hist, numbins)
            th_locs = _find_peak_locs_for_threshold(thhist, mode="top2", prepend_zero=False)
            numpeaks = int(th_locs.size)
            numbins += 1
    ththresh = _find_hist_dip_threshold_between(thhist, thhistbins, th_locs) if numpeaks == 2 else 0.0

    nrem = broadband > swthresh
    high_theta = thratio > ththresh
    high_emg = emg_interp > emgthresh
    rem = (~nrem) & (~high_emg) & high_theta
    wake = (~nrem) & (~rem)

    states = np.zeros((t_clus.size,), dtype=np.uint8)
    states[wake] = 1
    states[nrem] = 3
    states[rem] = 5

    statename_list = ["WAKE", "", "NREM", "", "REM"]
    ints = _idx_to_int(states=states, timestamps=t_clus, statenames=statename_list)
    idx_round = _int_to_idx(ints=ints, statenames=statename_list, sf=1.0)
    states = np.asarray(idx_round["states"], dtype=np.uint8).reshape(-1)
    idx_timestamps = np.asarray(idx_round["timestamps"], dtype=np.float64).reshape(-1)

    min_len = float(max(0.0, min_state_length))
    if block_wake_to_rem:
        states = _suppress_wake_to_rem_transitions(
            states,
            idx_timestamps,
            min_wake_before_rem_secs=min_len,
        )
    ints = _idx_to_int(states=states, timestamps=idx_timestamps, statenames=statename_list)

    # Minimum interruption passes (approximation of ClusterStates_DetermineStates)
    min_sws = min_len
    min_w_next_rem = min_len
    min_rem_in_w = min_len
    min_rem = min_len
    min_wake = min_len

    wake_int = np.asarray(ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    nrem_int = np.asarray(ints.get("NREMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    rem_int = np.asarray(ints.get("REMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)

    # short nrem -> wake
    short_nrem = (nrem_int[:, 1] - nrem_int[:, 0]) <= min_sws if nrem_int.size else np.asarray([], dtype=bool)
    if short_nrem.size and np.any(short_nrem):
        for st, en in nrem_int[short_nrem]:
            m = (idx_timestamps >= st) & (idx_timestamps <= en)
            states[m] = 1

    ints = _idx_to_int(states=states, timestamps=idx_timestamps, statenames=statename_list)
    wake_int = np.asarray(ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    rem_int = np.asarray(ints.get("REMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)

    # short wake next to rem -> rem
    short_w = (wake_int[:, 1] - wake_int[:, 0]) <= min_w_next_rem if wake_int.size else np.asarray([], dtype=bool)
    wr, rw = _find_next_to_ints(wake_int, rem_int)
    if short_w.size and np.any(short_w & (wr | rw)):
        for st, en in wake_int[short_w & (wr | rw)]:
            m = (idx_timestamps >= st) & (idx_timestamps <= en)
            states[m] = 5

    ints = _idx_to_int(states=states, timestamps=idx_timestamps, statenames=statename_list)
    rem_int = np.asarray(ints.get("REMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    wake_int = np.asarray(ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)

    # short rem in wake -> wake
    short_r = (rem_int[:, 1] - rem_int[:, 0]) <= min_rem_in_w if rem_int.size else np.asarray([], dtype=bool)
    rr, rl = _find_next_to_ints(rem_int, wake_int)
    if short_r.size and np.any(short_r & (rr & rl)):
        for st, en in rem_int[short_r & (rr & rl)]:
            m = (idx_timestamps >= st) & (idx_timestamps <= en)
            states[m] = 1

    ints = _idx_to_int(states=states, timestamps=idx_timestamps, statenames=statename_list)
    rem_int = np.asarray(ints.get("REMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    short_r2 = (rem_int[:, 1] - rem_int[:, 0]) <= min_rem if rem_int.size else np.asarray([], dtype=bool)
    if short_r2.size and np.any(short_r2):
        for st, en in rem_int[short_r2]:
            m = (idx_timestamps >= st) & (idx_timestamps <= en)
            states[m] = 1

    ints = _idx_to_int(states=states, timestamps=idx_timestamps, statenames=statename_list)
    wake_int = np.asarray(ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    nrem_int = np.asarray(ints.get("NREMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    short_w2 = (wake_int[:, 1] - wake_int[:, 0]) <= min_wake if wake_int.size else np.asarray([], dtype=bool)
    wn, nw = _find_next_to_ints(wake_int, nrem_int)
    if short_w2.size and np.any(short_w2 & (wn | nw)):
        for st, en in wake_int[short_w2 & (wn | nw)]:
            m = (idx_timestamps >= st) & (idx_timestamps <= en)
            states[m] = 3

    ints = _idx_to_int(states=states, timestamps=idx_timestamps, statenames=statename_list)
    nrem_int = np.asarray(ints.get("NREMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    short_nrem2 = (nrem_int[:, 1] - nrem_int[:, 0]) <= min_sws if nrem_int.size else np.asarray([], dtype=bool)
    if short_nrem2.size and np.any(short_nrem2):
        for st, en in nrem_int[short_nrem2]:
            m = (idx_timestamps >= st) & (idx_timestamps <= en)
            states[m] = 1

    ints = _idx_to_int(states=states, timestamps=idx_timestamps, statenames=statename_list)

    statenames = np.asarray(statename_list, dtype=object).reshape(1, -1)
    idx_struct = {
        "states": states.astype(np.uint8, copy=False).reshape(-1, 1),
        "timestamps": _to_uint_vector(np.rint(idx_timestamps)).reshape(-1, 1),
        "statenames": statenames,
    }
    ints_struct = {
        "WAKEstate": _to_uint_intervals(ints.get("WAKEstate", np.empty((0, 2)))),
        "NREMstate": _to_uint_intervals(ints.get("NREMstate", np.empty((0, 2)))),
        "REMstate": _to_uint_intervals(ints.get("REMstate", np.empty((0, 2)))),
    }

    hists = {
        "swhist": np.asarray(swhist, dtype=np.float64).reshape(1, -1),
        "swhistbins": np.asarray(swhistbins, dtype=np.float64).reshape(1, -1),
        "swthresh": float(swthresh),
        "EMGhist": np.asarray(emghist, dtype=np.float64).reshape(1, -1),
        "EMGhistbins": np.asarray(emghistbins, dtype=np.float64).reshape(1, -1),
        "EMGthresh": float(emgthresh),
        "THhist": np.asarray(thhist, dtype=np.float64).reshape(1, -1),
        "THhistbins": np.asarray(thhistbins, dtype=np.float64).reshape(1, -1),
        "THthresh": float(ththresh),
        "stickySW": bool(sticky_trigger),
        "stickyTH": False,
        "stickyEMG": bool(sticky_trigger),
    }

    ss_metrics = {
        "broadbandSlowWave": np.asarray(broadband, dtype=np.float64).reshape(-1, 1),
        "thratio": np.asarray(thratio, dtype=np.float64).reshape(-1, 1),
        "EMG": np.asarray(emg_interp, dtype=np.float64).reshape(-1, 1),
        "t_clus": np.asarray(t_clus, dtype=np.float64).reshape(-1, 1),
        "badtimes": (np.asarray(badtimes, dtype=np.float64) + 1.0).reshape(-1, 1),
        "badtimes_TH": (np.asarray(badtimes_th, dtype=np.float64) + 1.0).reshape(-1, 1),
        "histsandthreshs": hists,
        "LFPparams": sleepscore_lfp["params"],
        "WindowParams": {"window": float(window_sec), "smoothfact": float(smoothfact)},
        "THchanID": sleepscore_lfp["THchanID"],
        "SWchanID": sleepscore_lfp["SWchanID"],
        "recordingname": basename,
        "THdiptest": float(0.0),
        "EMGdiptest": float(0.0),
        "SWdiptest": float(0.0),
    }

    min_time_parms = {
        "minSWSsecs": 6,
        "minWnexttoREMsecs": 6,
        "minWinREMsecs": 6,
        "minREMinWsecs": 6,
        "minREMsecs": 6,
        "minWAKEsecs": 6,
    }

    userinputs = {
        "ignoreManual": bool(state_ignore_manual),
        "ignoretime": np.asarray(ignoretime, dtype=np.float64).reshape(-1, 2) if np.asarray(ignoretime).size else np.empty((0,), dtype=np.uint8),
        "noPrompts": True,
        "EMGthAlpha": float(emg_th_alpha),
        "Notch60Hz": np.uint8(0),
        "NotchHVS": np.uint8(0),
        "NotchTheta": np.uint8(0),
        "NotchUnder3Hz": np.uint8(0),
        "overwrite": bool(overwrite),
        "rejectChannels": reject_channels_1based.reshape(-1),
        "savebool": True,
        "savedir": str(basepath.parent),
        "saveLFP": bool(state_save_lfp_mat),
        "scoretime": np.asarray([0.0, np.inf], dtype=np.float64),
        "stickytrigger": bool(sticky_trigger),
        "minStateLength": float(min_len),
        "blockWakeToREM": bool(block_wake_to_rem),
        "SWChannels": np.asarray(sw_channels_1based if sw_channels_1based is not None else [0]).reshape(-1),
        "SWWeightsName": "PSS",
        "ThetaChannels": np.asarray(th_channels_1based if th_channels_1based is not None else [0]).reshape(-1),
        "winparms": np.asarray([window_sec, smoothfact], dtype=np.float64),
    }
    detectionparms = {
        "userinputs": userinputs,
        "MinTimeWindowParms": min_time_parms,
        "SleepScoreMetrics": ss_metrics,
        "histsandthreshs_orig": hists,
    }

    state_plot_materials = {
        "t_clus": np.asarray(t_clus, dtype=np.float32).reshape(-1, 1),
        "swFFTfreqs": np.asarray(sw_freqs, dtype=np.float64).reshape(-1, 1),
        "swFFTspec": np.asarray(swFFTspec, dtype=np.float32),
        "thFFTfreqs": np.asarray(th_freqs, dtype=np.float64).reshape(-1, 1),
        "thFFTspec": np.asarray(thFFTspec, dtype=np.float32),
        "IRASAsmooth_th": np.asarray(IRASAsmooth_th, dtype=np.float32),
        "thFFTspec_raw": np.asarray(thFFTspec_raw, dtype=np.float32),
        "IRASAsmooth": np.asarray(pss_sw["irasa_smooth"], dtype=np.float32).T,
        "IRASAintercept": np.asarray(pss_sw["intercept"], dtype=np.float64).reshape(-1, 1),
        "IRASAslope": np.asarray(pss_sw["slope"], dtype=np.float64).reshape(-1, 1),
    }
    date_text = datetime.now().strftime("%Y-%m-%d")
    detectorinfo = {
        "detectorname": "SleepScoreMaster",
        "detectionparms": detectionparms,
        "detectiondate": _matlab_datetime_like(date_text),
        "StatePlotMaterials": state_plot_materials,
    }
    sleep_state = {
        "ints": ints_struct,
        "idx": idx_struct,
        "detectorinfo": detectorinfo,
    }
    savemat(out_path, {"SleepState": sleep_state}, do_compression=True)
    return sleep_state, out_path


def _state_colors(idx: np.ndarray) -> np.ndarray:
    i = np.asarray(idx).reshape(-1)
    colors = np.full((i.size, 3), 0.8, dtype=np.float64)
    colors[i == 1] = np.asarray([0.0, 0.0, 0.0])  # WAKE/MA
    colors[i == 3] = np.asarray([0.0, 0.0, 1.0])  # NREM
    colors[i == 5] = np.asarray([1.0, 0.0, 0.0])  # REM
    return colors


def _save_state_figures(basepath: Path, basename: str, sleep_state: dict[str, Any]) -> list[Path]:
    fig_dir = basepath / "StateScoreFigures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    p_results = fig_dir / f"{basename}_SSResults.jpg"
    p_2d = fig_dir / f"{basename}_SSCluster2D.jpg"
    p_3d = fig_dir / f"{basename}_SSCluster3D.jpg"

    metrics = sleep_state["detectorinfo"]["detectionparms"]["SleepScoreMetrics"]
    t = np.asarray(metrics["t_clus"], dtype=np.float64).reshape(-1)
    sw = np.asarray(metrics["broadbandSlowWave"], dtype=np.float64).reshape(-1)
    th = np.asarray(metrics["thratio"], dtype=np.float64).reshape(-1)
    emg = np.asarray(metrics["EMG"], dtype=np.float64).reshape(-1)
    hists = metrics.get("histsandthreshs", {})
    spm = sleep_state["detectorinfo"].get("StatePlotMaterials", {})

    idx_src = np.asarray(sleep_state["idx"].get("states", np.asarray([])), dtype=np.float64).reshape(-1)
    idx_ts_src = np.asarray(sleep_state["idx"].get("timestamps", np.asarray([])), dtype=np.float64).reshape(-1)
    if idx_src.size and idx_ts_src.size and t.size:
        idx = np.rint(_nearest_interp(idx_ts_src, idx_src, t)).astype(np.int64, copy=False)
    else:
        idx = np.rint(idx_src).astype(np.int64, copy=False)
        if idx.size != t.size:
            idx = np.zeros((t.size,), dtype=np.int64)

    ints = sleep_state.get("ints", {})
    wake_ints = np.asarray(ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    nrem_ints = np.asarray(ints.get("NREMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    rem_ints = np.asarray(ints.get("REMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)

    sw_hist = np.asarray(hists.get("swhist", np.asarray([])), dtype=np.float64).reshape(-1)
    sw_bins = np.asarray(hists.get("swhistbins", np.asarray([])), dtype=np.float64).reshape(-1)
    sw_thr = float(hists.get("swthresh", 0.5))
    emg_hist = np.asarray(hists.get("EMGhist", np.asarray([])), dtype=np.float64).reshape(-1)
    emg_bins = np.asarray(hists.get("EMGhistbins", np.asarray([])), dtype=np.float64).reshape(-1)
    emg_thr = float(hists.get("EMGthresh", 0.5))
    th_hist = np.asarray(hists.get("THhist", np.asarray([])), dtype=np.float64).reshape(-1)
    th_bins = np.asarray(hists.get("THhistbins", np.asarray([])), dtype=np.float64).reshape(-1)
    th_thr = float(hists.get("THthresh", 0.5))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        for p in (p_results, p_2d, p_3d):
            p.touch()
        return [p_results, p_2d, p_3d]

    # Figure 1: SSResults (spectrograms + state bars + metrics)
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(8, 1, hspace=0.18)
    ax_sw_spec = fig.add_subplot(gs[0:2, 0])
    ax_th_spec = fig.add_subplot(gs[2, 0], sharex=ax_sw_spec)
    ax_state = fig.add_subplot(gs[3, 0], sharex=ax_sw_spec)
    ax_sw = fig.add_subplot(gs[4, 0], sharex=ax_sw_spec)
    ax_th = fig.add_subplot(gs[5, 0], sharex=ax_sw_spec)
    ax_emg = fig.add_subplot(gs[6, 0], sharex=ax_sw_spec)

    viewwin = (float(t[0]), float(t[-1])) if t.size else (0.0, 1.0)
    sw_spec = np.asarray(spm.get("swFFTspec", np.empty((0, 0))), dtype=np.float64)
    sw_freqs = np.asarray(spm.get("swFFTfreqs", np.asarray([])), dtype=np.float64).reshape(-1)
    if sw_spec.size and sw_freqs.size and t.size:
        log_sw = np.log10(np.maximum(sw_spec, np.finfo(np.float64).eps))
        mu = float(np.nanmean(log_sw))
        sig = float(np.nanstd(log_sw))
        im = ax_sw_spec.imshow(
            log_sw,
            aspect="auto",
            origin="lower",
            extent=[viewwin[0], viewwin[1], float(np.log2(sw_freqs[0])), float(np.log2(sw_freqs[-1]))],
            interpolation="nearest",
        )
        if np.isfinite(mu) and np.isfinite(sig) and sig > 0:
            im.set_clim(mu - 2.0 * sig, mu + 2.0 * sig)
        yt = np.asarray([1, 2, 4, 8, 16, 32, 64], dtype=float)
        yt = yt[(yt >= sw_freqs[0]) & (yt <= sw_freqs[-1])]
        ax_sw_spec.set_yticks(np.log2(yt))
        ax_sw_spec.set_yticklabels([str(int(v)) for v in yt])
    ax_sw_spec.set_ylabel("swLFP\nf (Hz)")
    ax_sw_spec.set_title(f"{basename}: State Scoring Results")
    ax_sw_spec.set_xlim(viewwin)
    ax_sw_spec.tick_params(axis="x", labelbottom=False)

    th_spec = np.asarray(spm.get("thFFTspec", np.empty((0, 0))), dtype=np.float64)
    th_freqs = np.asarray(spm.get("thFFTfreqs", np.asarray([])), dtype=np.float64).reshape(-1)
    if th_spec.size and th_freqs.size and t.size:
        if np.all(np.isfinite(np.log10(np.maximum(th_spec, np.finfo(np.float64).eps)))):
            base = np.log10(np.maximum(th_spec, np.finfo(np.float64).eps))
        else:
            base = th_spec
        mu = float(np.nanmean(base))
        sig = float(np.nanstd(base))
        im = ax_th_spec.imshow(
            base,
            aspect="auto",
            origin="lower",
            extent=[viewwin[0], viewwin[1], float(np.log2(th_freqs[0])), float(np.log2(th_freqs[-1]))],
            interpolation="nearest",
        )
        if np.isfinite(mu) and np.isfinite(sig) and sig > 0:
            im.set_clim(mu - 2.0 * sig, mu + 2.0 * sig)
        yt = np.asarray([1, 2, 4, 8, 16, 32, 64], dtype=float)
        yt = yt[(yt >= th_freqs[0]) & (yt <= th_freqs[-1])]
        ax_th_spec.set_yticks(np.log2(yt))
        ax_th_spec.set_yticklabels([str(int(v)) for v in yt])
    ax_th_spec.set_ylabel("thLFP\nf (Hz)")
    ax_th_spec.set_xlim(viewwin)
    ax_th_spec.tick_params(axis="x", labelbottom=False)

    for arr, y, c in ((wake_ints, -1.0, "k"), (nrem_ints, -2.0, "b"), (rem_ints, -3.0, "r")):
        for s, e in arr:
            ax_state.plot([s, e], [y, y], color=c, linewidth=6, solid_capstyle="butt")
    ax_state.set_ylim(-4.0, 0.0)
    ax_state.set_yticks([-3, -2, -1])
    ax_state.set_yticklabels(["REM", "SWS", "Wake/MA"])
    ax_state.set_xlim(viewwin)
    ax_state.tick_params(axis="x", labelbottom=False)

    ax_sw.plot(t, sw, "k", linewidth=0.8)
    ax_sw.set_ylabel("SW")
    ax_sw.set_ylim(0, 1)
    ax_sw.set_xlim(viewwin)
    ax_sw.tick_params(axis="x", labelbottom=False)

    ax_th.plot(t, th, "k", linewidth=0.8)
    ax_th.set_ylabel("Theta")
    ax_th.set_ylim(0, 1)
    ax_th.set_xlim(viewwin)
    ax_th.tick_params(axis="x", labelbottom=False)

    ax_emg.plot(t, emg, "k", linewidth=0.8)
    ax_emg.set_ylabel("EMG")
    ax_emg.set_ylim(0, 1)
    ax_emg.set_xlim(viewwin)
    ax_emg.set_xlabel("t (s)")

    fig.savefig(p_results, dpi=120)
    plt.close(fig)

    # Figure 2: SSCluster2D
    fig = plt.figure(figsize=(10, 10.5))
    # 6-row grid lets the right column use two equal-height panels (3 rows each).
    gs = fig.add_gridspec(6, 2, wspace=0.35, hspace=0.9)
    ax11 = fig.add_subplot(gs[0:2, 0])
    ax21 = fig.add_subplot(gs[2:4, 0])
    ax31 = fig.add_subplot(gs[4:6, 0])
    ax12 = fig.add_subplot(gs[0:3, 1])
    ax22 = fig.add_subplot(gs[3:6, 1])

    def _bar_split(ax, bins, vals, thr, c_hi, c_lo, title, xlabel):
        if bins.size == 0 or vals.size == 0:
            ax.set_title(title, pad=10.0)
            ax.set_xlabel(xlabel)
            return
        bw = float(np.median(np.diff(bins))) if bins.size > 1 else 0.05
        hi = bins > thr
        lo = ~hi
        ax.bar(bins[hi], vals[hi], color=c_hi, width=bw * 0.9, linewidth=0.8)
        ax.bar(bins[lo], vals[lo], color=c_lo, width=bw * 0.9, linewidth=0.8)
        ax.plot([thr, thr], [0, float(np.nanmax(vals) if vals.size else 1.0)], "r", linewidth=1.0)
        ax.set_title(title, pad=10.0)
        ax.set_xlabel(xlabel)

    _bar_split(ax11, sw_bins, sw_hist, sw_thr, "b", (0.9, 0.9, 0.9), "Step 1: Broadband for NREM", "Broadband Slow Wave")
    _bar_split(ax21, emg_bins, emg_hist, emg_thr, "k", (0.9, 0.9, 0.9), "Step 2: EMG for Muscle Tone", "EMG")
    if th_bins.size and th_hist.size:
        bw = float(np.median(np.diff(th_bins))) if th_bins.size > 1 else 0.05
        hi = th_bins >= th_thr
        ax31.bar(th_bins[hi], th_hist[hi], color="r", width=bw * 0.9, linewidth=0.8)
        ax31.bar(th_bins[~hi], th_hist[~hi], color="k", width=bw * 0.9, linewidth=0.8)
        ax31.plot([th_thr, th_thr], [0, float(np.nanmax(th_hist) if th_hist.size else 1.0)], "r", linewidth=1.0)
    ax31.set_title("Step 3: Theta for REM", pad=10.0)
    ax31.set_xlabel("Theta")

    # MATLAB-like state coloring for 2D split panels
    nrem_mask = idx == 3
    wake_mask = idx == 1
    rem_mask = idx == 5
    wake_high_emg = wake_mask & (emg > emg_thr)
    wake_low_or_rem = (wake_mask & (emg < emg_thr)) | rem_mask
    ax12.plot(sw[nrem_mask], emg[nrem_mask], "b.", markersize=1.0)
    ax12.plot(sw[wake_high_emg], emg[wake_high_emg], "k.", markersize=1.0)
    ax12.plot(sw[wake_low_or_rem], emg[wake_low_or_rem], ".", color=(0.8, 0.8, 0.8), markersize=1.0)
    ax12.plot([sw_thr, sw_thr], list(ax12.get_ylim()), "r", linewidth=1.0)
    ax12.plot([0.0, sw_thr], [emg_thr, emg_thr], "r", linewidth=1.0)
    ax12.set_xlabel("Broadband SW")
    ax12.set_ylabel("EMG")

    ax22.plot(th[wake_mask], emg[wake_mask], "k.", markersize=1.0)
    ax22.plot(th[rem_mask], emg[rem_mask], "r.", markersize=1.0)
    ax22.plot([th_thr, th_thr], [0.0, emg_thr], "r", linewidth=1.0)
    ax22.plot([0.0, 1.0], [emg_thr, emg_thr], "r", linewidth=1.0)
    ax22.set_xlabel("Narrowband Theta")
    ax22.set_ylabel("EMG")

    fig.savefig(p_2d, dpi=120)
    plt.close(fig)

    # Figure 3: SSCluster3D
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 3, wspace=0.4, hspace=0.45)
    ax_sw_hist = fig.add_subplot(gs[0, 0])
    ax_emg_hist = fig.add_subplot(gs[1, 0])
    ax_th_hist = fig.add_subplot(gs[2, 0])
    ax3d = fig.add_subplot(gs[:, 1:], projection="3d")

    def _bar_outline(ax, bins, vals, thr, title, xlabel):
        if bins.size == 0 or vals.size == 0:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            return
        bw = float(np.median(np.diff(bins))) if bins.size > 1 else 0.05
        ax.bar(bins, vals, color="none", edgecolor="k", width=bw * 0.9, linewidth=1.2)
        ax.plot([thr, thr], [0, float(np.nanmax(vals))], "r", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    _bar_outline(ax_sw_hist, sw_bins, sw_hist, sw_thr, "Step 1: Broadband for NREM", "Slow Wave")
    _bar_outline(ax_emg_hist, emg_bins, emg_hist, emg_thr, "Step 2: EMG for Muscle Tone", "EMG")
    _bar_outline(ax_th_hist, th_bins, th_hist, th_thr, "Step 3: Theta for REM", "Theta")

    ax3d.scatter(sw, th, emg, s=1.0, c=_state_colors(idx), marker="o", depthshade=False)
    ax3d.view_init(elev=18.8, azim=133.7)
    ax3d.grid(True)
    ax3d.set_xlabel("Broadband SW")
    ax3d.set_ylabel("Narrowband Theta")
    ax3d.set_zlabel("EMG")

    fig.savefig(p_3d, dpi=120)
    plt.close(fig)

    return [p_results, p_2d, p_3d]


def _states_to_episodes(sleep_state: dict[str, Any], basepath: Path, basename: str) -> tuple[dict[str, Any], Path]:
    ints = sleep_state.get("ints", {})
    nrem = np.asarray(ints.get("NREMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    wake = np.asarray(ints.get("WAKEstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)
    rem = np.asarray(ints.get("REMstate", np.empty((0, 2))), dtype=np.float64).reshape(-1, 2)

    min_packet = 30.0
    min_w_episode = 20.0
    min_n_episode = 20.0
    min_r_episode = 20.0
    max_micro = 100.0
    max_w_interrupt = 40.0
    max_n_interrupt = max_micro
    max_r_interrupt = 40.0

    packet = nrem[(nrem[:, 1] - nrem[:, 0]) >= min_packet] if nrem.size else np.empty((0, 2))
    wake_len = (wake[:, 1] - wake[:, 0]) if wake.size else np.asarray([])
    ma = wake[wake_len <= max_micro] if wake.size else np.empty((0, 2))
    wake_intervals = wake[wake_len > max_micro] if wake.size else np.empty((0, 2))

    wake_episode = _merge_intervals(wake_intervals, max_w_interrupt, min_w_episode)
    nrem_episode = _merge_intervals(nrem, max_n_interrupt, min_n_episode)
    rem_episode = _merge_intervals(rem, max_r_interrupt, min_r_episode)

    # split NREM episodes containing REM intervals
    if nrem_episode.size and rem.size:
        kept: list[list[float]] = []
        for ns, ne in nrem_episode:
            inside = rem[(rem[:, 0] >= ns) & (rem[:, 0] <= ne)]
            if inside.size == 0:
                kept.append([float(ns), float(ne)])
                continue
            cur_start = float(ns)
            for rs, re in inside:
                if rs > cur_start:
                    kept.append([cur_start, float(rs)])
                cur_start = float(re)
            if ne > cur_start:
                kept.append([cur_start, float(ne)])
        nrem_episode = np.asarray(kept, dtype=np.float64) if kept else np.empty((0, 2), dtype=np.float64)
        if nrem_episode.size:
            keep = (nrem_episode[:, 1] - nrem_episode[:, 0]) >= min_n_episode
            nrem_episode = nrem_episode[keep, :]

    ma_rem = np.empty((0, 2), dtype=np.float64)
    if ma.size and rem.size:
        flags = _in_intervals(ma[:, 0], rem) | _in_intervals(ma[:, 1], rem)
        ma_rem = ma[flags, :]
        ma = ma[~flags, :]

    # remove overlap among episodes
    no_overlap = [wake_episode.copy(), nrem_episode.copy(), rem_episode.copy()]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            no_overlap[i] = _subtract_intervals(no_overlap[i], no_overlap[j])
    wake_episode, nrem_episode, rem_episode = no_overlap

    det = {
        "originaldetectorinfo": sleep_state["detectorinfo"],
        "detectionparms": {
            "EpisodeDetectionParms": {
                "minPacketDuration": min_packet,
                "minWAKEEpisodeDuration": min_w_episode,
                "minNREMEpisodeDuration": min_n_episode,
                "minREMEpisodeDuration": min_r_episode,
                "maxMicroarousalDuration": max_micro,
                "maxWAKEEpisodeInterruption": max_w_interrupt,
                "maxNREMEpisodeInterruption": max_n_interrupt,
                "maxREMEpisodeInterruption": max_r_interrupt,
            }
        },
        "detectiondate": datetime.now().strftime("%Y-%m-%d"),
    }
    episodes = {
        "ints": {
            "NREMepisode": _to_uint_intervals(nrem_episode),
            "REMepisode": _to_uint_intervals(rem_episode),
            "WAKEepisode": _to_uint_intervals(wake_episode),
            "NREMpacket": _to_uint_intervals(packet),
            "MA": _to_uint_intervals(ma),
            "MA_REM": _to_uint_intervals(ma_rem),
        },
        "detectorinfo": det,
    }
    out_path = basepath / f"{basename}.SleepStateEpisodes.states.mat"
    savemat(out_path, {"SleepStateEpisodes": episodes}, do_compression=True)
    return episodes, out_path


def _append_theta_epochs(sleep_state: dict[str, Any], basepath: Path, basename: str) -> tuple[dict[str, Any], Path]:
    metrics = sleep_state["detectorinfo"]["detectionparms"]["SleepScoreMetrics"]
    hists = metrics["histsandthreshs"]

    thratio = np.asarray(metrics["thratio"], dtype=np.float64).reshape(-1)
    emg = np.asarray(metrics["EMG"], dtype=np.float64).reshape(-1)
    ththr = float(hists["THthresh"])
    emgthr = float(hists["EMGthresh"])

    states = np.asarray(sleep_state["idx"]["states"]).reshape(-1)
    theta_ndx = (thratio > ththr) & (emg > emgthr)

    theta_states = np.zeros(states.shape, dtype=np.uint8)
    theta_states[theta_ndx] = 7
    non_theta = (states == 1) & (theta_states == 0)
    theta_states[non_theta] = 9

    timestamps = np.asarray(sleep_state["idx"]["timestamps"]).reshape(-1).astype(np.float64)
    statenames = np.asarray(["", "", "", "", "", "", "THETA", "", "nonTHETA"], dtype=object)
    theta_idx = {
        "states": theta_states.reshape(-1, 1),
        "timestamps": np.asarray(sleep_state["idx"]["timestamps"]).reshape(-1, 1),
        "statenames": statenames.reshape(1, -1),
    }

    theta_ints = _idx_to_int(theta_states, timestamps, list(statenames))
    sleep_state["idx"]["theta_epochs"] = theta_idx
    sleep_state["ints"]["THETA"] = _to_uint_intervals(theta_ints.get("THETAstate", np.empty((0, 2))))
    sleep_state["ints"]["nonTHETA"] = _to_uint_intervals(theta_ints.get("nonTHETAstate", np.empty((0, 2))))

    out_path = basepath / f"{basename}.SleepState.states.mat"
    savemat(out_path, {"SleepState": sleep_state}, do_compression=True)
    return sleep_state, out_path


def _extract_ignoretime_from_pulses(pulses: dict[str, Any] | None) -> np.ndarray:
    if pulses is None:
        return np.empty((0, 2), dtype=np.float64)
    try:
        ints = np.asarray(pulses.get("intsPeriods", np.empty((0, 2))), dtype=np.float64)
    except Exception:
        return np.empty((0, 2), dtype=np.float64)
    if ints.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return ints.reshape(-1, 2)


def run_state_scoring(
    *,
    basepath: Path,
    basename: str,
    session_struct: dict[str, Any],
    pulses: dict[str, Any] | None,
    config: PreprocessConfig,
) -> StateScoreResult:
    basepath = Path(basepath)
    lfp_candidates = [
        basepath / f"{basename}.lfp",
        basepath / f"{basename}.eeg",
    ]
    lfp_path = next((p for p in lfp_candidates if p.exists()), None)
    if lfp_path is None:
        raise FileNotFoundError(
            f"State scoring requires {basename}.lfp/.eeg in {basepath}, but none found."
        )

    ignoretime = _extract_ignoretime_from_pulses(pulses)
    reject_channels_1based = _extract_bad_channels_1based(session_struct)
    parallel_jobs = _resolve_parallel_jobs(config.job_kwargs)
    pss_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
    sw_channels_1based = None
    if config.sw_channels:
        sw_channels_1based = np.asarray(config.sw_channels, dtype=np.int64).reshape(-1)
    th_channels_1based = None
    if config.theta_channels:
        th_channels_1based = np.asarray(config.theta_channels, dtype=np.int64).reshape(-1)

    emg, emg_path = _compute_emg_from_lfp(
        basepath=basepath,
        basename=basename,
        lfp_path=lfp_path,
        session_struct=session_struct,
        reject_channels_1based=reject_channels_1based,
        overwrite=config.overwrite,
        sampling_frequency=2.0,
    )
    sleepscore_lfp, sleep_lfp_path, swth_fig = _compute_sleepscore_lfp(
        basepath=basepath,
        basename=basename,
        lfp_path=lfp_path,
        session_struct=session_struct,
        reject_channels_1based=reject_channels_1based,
        sw_channels_1based=sw_channels_1based,
        th_channels_1based=th_channels_1based,
        ignoretime=ignoretime,
        window_sec=float(config.state_winparms[0]),
        smoothfact=float(config.state_winparms[1]),
        overwrite=config.overwrite,
        save_files=config.state_save_lfp_mat,
        parallel_jobs=parallel_jobs,
        pss_cache=pss_cache,
    )
    sleep_state, sleep_state_path = _compute_sleep_state(
        basepath=basepath,
        basename=basename,
        sleepscore_lfp=sleepscore_lfp,
        emg=emg,
        ignoretime=ignoretime,
        sticky_trigger=config.state_sticky_trigger,
        window_sec=float(config.state_winparms[0]),
        smoothfact=float(config.state_winparms[1]),
        reject_channels_1based=reject_channels_1based,
        sw_channels_1based=sw_channels_1based,
        th_channels_1based=th_channels_1based,
        state_ignore_manual=config.state_ignore_manual,
        state_save_lfp_mat=config.state_save_lfp_mat,
        emg_th_alpha=float(config.emg_th_alpha),
        min_state_length=float(config.state_min_state_length),
        block_wake_to_rem=bool(config.state_block_wake_to_rem),
        overwrite=config.overwrite,
        pss_cache=pss_cache,
    )
    fig_paths = _save_state_figures(basepath, basename, sleep_state)
    episodes, episodes_path = _states_to_episodes(sleep_state, basepath, basename)
    if not episodes:
        warnings.warn("Failed to build SleepStateEpisodes.", RuntimeWarning, stacklevel=2)
    sleep_state, sleep_state_path = _append_theta_epochs(sleep_state, basepath, basename)

    all_fig_paths = [swth_fig, *fig_paths]
    return StateScoreResult(
        emg_mat_path=emg_path,
        sleepscore_lfp_mat_path=sleep_lfp_path,
        sleep_state_mat_path=sleep_state_path,
        sleep_state_episodes_mat_path=episodes_path,
        figure_paths=all_fig_paths,
    )
