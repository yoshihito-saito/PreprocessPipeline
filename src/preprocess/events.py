from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat


ANALOG_BEHAVIOR_DEFAULT_FS = 1250.0
ANALOG_PULSE_PERIOD_LAG_SEC = 20.0
DIGITAL_PERIOD_LAG_SEC = 5.0


def _concat_binary_files(paths: list[Path], out_path: Path, overwrite: bool) -> Path | None:
    if not paths:
        return None
    if out_path.exists() and not overwrite:
        return out_path

    with open(out_path, "wb") as fout:
        for p in paths:
            with open(p, "rb") as fin:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
    return out_path


def materialize_intermediate_dat(
    *,
    output_dir: Path,
    basename: str,
    analogin_paths: list[Path],
    digitalin_paths: list[Path],
    auxiliary_paths: list[Path],
    supply_paths: list[Path],
    time_paths: list[Path],
    sample_counts: list[int],
    overwrite: bool,
) -> dict[str, Path]:
    out: dict[str, Path] = {}

    p = _concat_binary_files(analogin_paths, output_dir / "analogin.dat", overwrite)
    if p is not None:
        out["analogin"] = p

    p = _concat_binary_files(digitalin_paths, output_dir / "digitalin.dat", overwrite)
    if p is not None:
        out["digitalin"] = p

    p = _concat_binary_files(auxiliary_paths, output_dir / "auxiliary.dat", overwrite)
    if p is not None:
        out["auxiliary"] = p

    p = _concat_binary_files(supply_paths, output_dir / "supply.dat", overwrite)
    if p is not None:
        out["supply"] = p

    p = _concat_binary_files(time_paths, output_dir / "time.dat", overwrite)
    if p is not None:
        out["time"] = p

    if "time" not in out:
        time_auto = output_dir / "time.dat"
        if overwrite or not time_auto.exists():
            t = (
                np.concatenate([np.arange(n, dtype=np.int64) for n in sample_counts])
                if sample_counts
                else np.array([], dtype=np.int64)
            )
            t.tofile(time_auto)
        out["time"] = time_auto

    return out


def _normalize_channel_indices(channels: list[int] | None, n_channels: int) -> list[int]:
    if n_channels <= 0:
        return []
    if not channels:
        return list(range(n_channels))

    vals = [int(ch) for ch in channels]
    if min(vals) >= 1 and max(vals) <= n_channels:
        vals = [v - 1 for v in vals]
    vals = sorted(set(v for v in vals if 0 <= v < n_channels))
    return vals


def _build_analog_behavior_struct(
    *,
    analog_data_u16: np.ndarray,
    selected_channels_0based: list[int],
) -> dict[str, Any]:
    n_samples = int(analog_data_u16.shape[0])
    if selected_channels_0based:
        selected = analog_data_u16[:, selected_channels_0based]
        channels_1based = [ch + 1 for ch in selected_channels_0based]
    else:
        selected = np.empty((n_samples, 0), dtype=np.uint16)
        channels_1based = []

    scaled = (selected.astype(np.float64) - 6800.0) * 5.0 / (59000.0 - 6800.0)
    timestamps = (np.arange(1, n_samples + 1, dtype=np.float64) / ANALOG_BEHAVIOR_DEFAULT_FS).reshape(-1, 1)
    duration = float(n_samples) / float(ANALOG_BEHAVIOR_DEFAULT_FS)

    return {
        "Filename": "analogin.dat",
        "data": scaled,
        "timestamps": timestamps,
        "channels": np.asarray(channels_1based, dtype=np.float64).reshape(1, -1),
        "samplingRate": float(ANALOG_BEHAVIOR_DEFAULT_FS),
        "interval": np.asarray([[0.0, duration]], dtype=np.float64),
        "duration": duration,
        "region": "analog",
    }


def _find_rising_falling_edges(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rising = np.flatnonzero((binary[:-1] == 0) & (binary[1:] == 1)).astype(np.int64)
    falling = np.flatnonzero((binary[:-1] == 1) & (binary[1:] == 0)).astype(np.int64)
    return rising, falling


def _pair_intervals(on_times: np.ndarray, off_times: np.ndarray) -> np.ndarray:
    if on_times.size == 0 or off_times.size == 0:
        return np.empty((2, 0), dtype=np.float64)

    off_idx = np.searchsorted(off_times, on_times, side="right")
    valid = off_idx < off_times.size
    if not np.any(valid):
        return np.empty((2, 0), dtype=np.float64)

    on_valid = on_times[valid]
    off_valid = off_times[off_idx[valid]]
    return np.vstack([on_valid, off_valid]).astype(np.float64)


def _build_periods(intervals_2xN: np.ndarray, lag_sec: float) -> np.ndarray:
    if intervals_2xN.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    starts = intervals_2xN[0, :]
    ends = intervals_2xN[1, :]
    periods: list[list[float]] = [[float(starts[0]), np.nan]]
    gap_idx = np.flatnonzero(np.diff(starts) > float(lag_sec))
    for idx in gap_idx:
        periods[-1][1] = float(ends[idx])
        periods.append([float(starts[idx + 1]), np.nan])
    periods[-1][1] = float(ends[-1])
    return np.asarray(periods, dtype=np.float64)


def _baseline_correct_by_epochs(signal: np.ndarray, merge_timestamps_sec: np.ndarray | None, sr: float) -> np.ndarray:
    if merge_timestamps_sec is None:
        return signal
    ts = np.asarray(merge_timestamps_sec, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[1] != 2:
        return signal

    corrected = signal.copy()
    n = corrected.size
    for start_sec, stop_sec in ts:
        start = max(0, int(np.floor(float(start_sec) * float(sr))))
        stop = min(n, int(np.floor(float(stop_sec) * float(sr))))
        if stop <= start:
            continue
        med = np.median(corrected[start:stop])
        corrected[start:stop] = corrected[start:stop] - med
    return corrected


def _detect_analog_pulses(
    *,
    analog_data_u16: np.ndarray,
    selected_channels_0based: list[int],
    sr: float,
    merge_timestamps_sec: np.ndarray | None,
) -> dict[str, np.ndarray]:
    pul_rows: list[np.ndarray] = []
    amp_rows: list[np.ndarray] = []
    dur_rows: list[np.ndarray] = []
    event_id_rows: list[np.ndarray] = []
    channel_rows: list[np.ndarray] = []
    period_rows: list[np.ndarray] = []

    for jj, ch0 in enumerate(selected_channels_0based, start=1):
        d = analog_data_u16[:, ch0].astype(np.float64)
        d = _baseline_correct_by_epochs(d, merge_timestamps_sec, sr=sr)

        if np.any(d < 0):
            d = d - np.min(d)

        sampled = d[::100] / 0.6745 if d.size > 0 else np.array([], dtype=np.float64)
        thr = float(150.0 * np.median(sampled)) if sampled.size else 0.0
        if thr == 0.0 or not np.any(d > thr):
            thr = float(4.5 * np.std(d)) if d.size else 0.0

        d_bin = (d > thr).astype(np.uint8)
        on_idx, off_idx = _find_rising_falling_edges(d_bin)
        on_t = on_idx.astype(np.float64) / float(sr)
        off_t = off_idx.astype(np.float64) / float(sr)
        intervals = _pair_intervals(on_t, off_t)
        if intervals.size == 0:
            continue

        baseline_d = np.int32(np.median(d[::100])) if d.size else np.int32(0)
        vals: list[float] = []
        keep_mask: list[bool] = []
        for start_t, stop_t in intervals.T:
            s = max(0, int(np.floor(start_t * sr)))
            e = min(d.size, int(np.ceil(stop_t * sr)))
            if e <= s:
                vals.append(0.0)
                keep_mask.append(False)
                continue
            med_val = np.median(d[s:e]).astype(np.float64) - float(baseline_d)
            vals.append(float(med_val))
            keep_mask.append(bool((med_val >= (thr - baseline_d) * 0.4) and (start_t >= 0)))

        keep = np.asarray(keep_mask, dtype=bool)
        if not np.any(keep):
            continue

        kept = intervals[:, keep]
        amps = np.asarray(vals, dtype=np.float64)[keep]
        durs = kept[1, :] - kept[0, :]
        event_ids = np.full(kept.shape[1], float(jj), dtype=np.float64)
        channels = np.full(kept.shape[1], float(ch0 + 1), dtype=np.float64)
        periods = _build_periods(kept, lag_sec=ANALOG_PULSE_PERIOD_LAG_SEC)

        pul_rows.append(kept.T)
        amp_rows.append(amps.reshape(-1, 1))
        dur_rows.append(durs.reshape(-1, 1))
        event_id_rows.append(event_ids.reshape(-1, 1))
        channel_rows.append(channels.reshape(-1, 1))
        if periods.size:
            period_rows.append(periods)

    if not pul_rows:
        return {
            "timestamps": np.empty((0, 2), dtype=np.float64),
            "amplitude": np.empty((0, 1), dtype=np.float64),
            "duration": np.empty((0, 1), dtype=np.float64),
            "eventGroupID": np.empty((0, 1), dtype=np.float64),
            "analogChannel": np.empty((0, 1), dtype=np.float64),
            "intsPeriods": np.empty((0, 2), dtype=np.float64),
        }

    timestamps = np.vstack(pul_rows).astype(np.float64)
    amplitude = np.vstack(amp_rows).astype(np.float64)
    duration = np.vstack(dur_rows).astype(np.float64)
    event_group_id = np.vstack(event_id_rows).astype(np.float64)
    analog_channel = np.vstack(channel_rows).astype(np.float64)
    ints_periods = (
        np.vstack(period_rows).astype(np.float64) if period_rows else np.empty((0, 2), dtype=np.float64)
    )

    sort_idx = np.argsort(timestamps[:, 0], kind="mergesort")
    timestamps = timestamps[sort_idx, :]
    amplitude = amplitude[sort_idx, :]
    duration = duration[sort_idx, :]
    event_group_id = event_group_id[sort_idx, :]
    analog_channel = analog_channel[sort_idx, :]

    if ints_periods.size:
        period_sort_idx = np.argsort(ints_periods[:, 0], kind="mergesort")
        ints_periods = ints_periods[period_sort_idx, :]

    return {
        "timestamps": timestamps,
        "amplitude": amplitude,
        "duration": duration,
        "eventGroupID": event_group_id,
        "analogChannel": analog_channel,
        "intsPeriods": ints_periods,
    }


def _build_empty_cell_row(n_cells: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.empty((1, n_cells), dtype=object)
    for i in range(n_cells):
        out[0, i] = np.empty(shape, dtype=np.float64)
    return out


def _build_digital_in_struct(
    *,
    digital_data_u16: np.ndarray,
    digital_channels: list[int] | None,
    fs: float,
    word_channels: int,
) -> dict[str, Any]:
    n_channels = 16
    selected = _normalize_channel_indices(digital_channels, n_channels)
    selected_set = set(selected)

    timestamps_on = _build_empty_cell_row(n_channels, (0, 1))
    timestamps_off = _build_empty_cell_row(n_channels, (0, 1))
    ints = _build_empty_cell_row(n_channels, (2, 0))
    dur = _build_empty_cell_row(n_channels, (1, 0))
    ints_periods = _build_empty_cell_row(n_channels, (0, 2))

    if digital_data_u16.ndim == 1:
        digital_data_u16 = digital_data_u16.reshape(-1, 1)

    for ch in range(n_channels):
        if selected and ch not in selected_set:
            continue

        if word_channels <= 1:
            bit = ((digital_data_u16[:, 0] >> ch) & 1).astype(np.uint8)
        else:
            if ch >= digital_data_u16.shape[1]:
                continue
            bit = (digital_data_u16[:, ch] != 0).astype(np.uint8)

        on_idx, off_idx = _find_rising_falling_edges(bit)
        if on_idx.size == 0 and off_idx.size == 0:
            continue

        on_t = on_idx.astype(np.float64) / float(fs)
        off_t = off_idx.astype(np.float64) / float(fs)
        d = np.zeros((2, max(on_t.size, off_t.size)), dtype=np.float64)
        if on_t.size:
            d[0, : on_t.size] = on_t
        if off_t.size:
            d[1, : off_t.size] = off_t

        if d.shape[1] and d[0, 0] > d[1, 0]:
            d = np.flip(d, axis=0)
        if d.shape[1] and d[1, -1] == 0:
            d[1, -1] = np.nan

        channel_periods = _build_periods(d, lag_sec=DIGITAL_PERIOD_LAG_SEC)

        timestamps_on[0, ch] = on_t.reshape(-1, 1)
        timestamps_off[0, ch] = off_t.reshape(-1, 1)
        ints[0, ch] = d
        dur[0, ch] = (d[1, :] - d[0, :]).reshape(1, -1)
        ints_periods[0, ch] = channel_periods

    return {
        "timestampsOn": timestamps_on,
        "timestampsOff": timestamps_off,
        "ints": ints,
        "dur": dur,
        "intsPeriods": ints_periods,
    }


def export_analog_digital_events(
    *,
    output_dir: Path,
    basename: str,
    analog_inputs: bool,
    analog_channels: list[int] | None,
    analog_num_channels: int,
    digital_inputs: bool,
    digital_channels: list[int] | None,
    digital_word_channels: int,
    sr: float,
    analog_sr: float | None,
    digital_sr: float | None,
    analog_dat_path: Path | None,
    digital_dat_path: Path | None,
    merge_timestamps_sec: np.ndarray | None,
    overwrite: bool,
) -> tuple[list[Path], list[Path]]:
    analog_paths: list[Path] = []
    digital_paths: list[Path] = []
    analog_sr_eff = float(analog_sr) if analog_sr is not None else float(sr)
    digital_sr_eff = float(digital_sr) if digital_sr is not None else float(sr)

    if analog_inputs and analog_dat_path is not None and analog_dat_path.exists():
        behavior_out = output_dir / f"{basename}.analogInput.behavior.mat"
        pulses_out = output_dir / f"{basename}.pulses.events.mat"

        if overwrite or not behavior_out.exists() or not pulses_out.exists():
            raw = np.fromfile(analog_dat_path, dtype=np.uint16)
            n_ch = int(analog_num_channels) if int(analog_num_channels) > 0 else 1
            if raw.size % n_ch != 0:
                raise ValueError(
                    f"analogin.dat size is not divisible by analog channel count: size={raw.size}, n_ch={n_ch}"
                )
            analog_data = raw.reshape(-1, n_ch)
            selected = _normalize_channel_indices(analog_channels, n_ch)

            analog_inp = _build_analog_behavior_struct(
                analog_data_u16=analog_data,
                selected_channels_0based=selected,
            )
            pulses = _detect_analog_pulses(
                analog_data_u16=analog_data,
                selected_channels_0based=selected,
                sr=analog_sr_eff,
                merge_timestamps_sec=merge_timestamps_sec,
            )
            savemat(behavior_out, {"analogInp": analog_inp}, do_compression=True)
            savemat(pulses_out, {"pulses": pulses}, do_compression=True)

        analog_paths.extend([behavior_out, pulses_out])

    if digital_inputs and digital_dat_path is not None and digital_dat_path.exists():
        digital_out = output_dir / "digitalIn.events.mat"

        if overwrite or not digital_out.exists():
            raw = np.fromfile(digital_dat_path, dtype=np.uint16)
            word_ch = int(digital_word_channels) if int(digital_word_channels) > 0 else 1
            if raw.size % word_ch != 0:
                raise ValueError(
                    f"digitalin.dat size is not divisible by digital word channel count: size={raw.size}, n_ch={word_ch}"
                )
            digital_data = raw.reshape(-1, word_ch)
            digital_in = _build_digital_in_struct(
                digital_data_u16=digital_data,
                digital_channels=digital_channels,
                fs=digital_sr_eff,
                word_channels=word_ch,
            )
            savemat(digital_out, {"digitalIn": digital_in}, do_compression=True)

        digital_paths.append(digital_out)

    return analog_paths, digital_paths
