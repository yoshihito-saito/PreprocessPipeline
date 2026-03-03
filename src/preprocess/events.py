from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat


ANALOG_BEHAVIOR_DEFAULT_FS = 1250.0
ANALOG_PULSE_PERIOD_LAG_SEC = 20.0
DIGITAL_PERIOD_LAG_SEC = 5.0
ANALOG_EM_THRESHOLD = 0.3


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
    active_channels_1based: list[int],
    sampling_rate: float = ANALOG_BEHAVIOR_DEFAULT_FS,
    interval_start: float = 0.0,
) -> dict[str, Any]:
    n_samples = int(analog_data_u16.shape[0])
    scaled = (analog_data_u16.astype(np.float64) - 6800.0) * 5.0 / (59000.0 - 6800.0)

    timestamps = (np.arange(1, n_samples + 1, dtype=np.float64) / float(sampling_rate)).reshape(-1, 1)
    if interval_start > 0:
        timestamps = timestamps + np.floor(interval_start * sampling_rate) / sampling_rate
        timestamps = timestamps - 1.0 / float(sampling_rate)
    duration = float(n_samples) / float(sampling_rate)

    return {
        "Filename": "analogin.dat",
        "data": scaled,
        "timestamps": timestamps,
        "channels": np.asarray(active_channels_1based, dtype=np.float64).reshape(1, -1),
        "samplingRate": float(sampling_rate),
        "interval": np.asarray([[float(interval_start), float(interval_start) + duration]], dtype=np.float64),
        "duration": duration,
        "region": "analog",
    }


def _find_rising_falling_edges_matlab(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = np.diff(binary.astype(np.int8))
    # MATLAB getAnalogPulses/getDigitalIn use find(diff(...)==...) and strfind([0,1]/[1,0]),
    # both of which return the transition index itself (no +1 shift).
    rising = np.flatnonzero(diff == 1).astype(np.int64)
    falling = np.flatnonzero(diff == -1).astype(np.int64)
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


def _effectiveness_metric(signal: np.ndarray) -> float:
    if signal.size < 2:
        return 0.0
    v = float(np.var(signal))
    if v <= 0.0:
        return 0.0
    midpoint = 0.5 * (float(np.min(signal)) + float(np.max(signal)))
    upper = signal >= midpoint
    lower = ~upper
    pu = float(np.mean(upper))
    pl = float(np.mean(lower))
    vu = float(np.var(signal[upper])) if np.any(upper) else 0.0
    vl = float(np.var(signal[lower])) if np.any(lower) else 0.0
    return 1.0 - (pu * vu + pl * vl) / v


def _detect_analog_pulses(
    *,
    analog_data_u16: np.ndarray,
    channel_ids_1based: list[int],
    sr: float,
    merge_timestamps_sec: np.ndarray | None,
    min_dur_sec: float | None = None,
    sess_epochs_1based: list[int] | None = None,
) -> dict[str, np.ndarray] | None:
    pul_rows: list[np.ndarray] = []
    amp_rows: list[np.ndarray] = []
    dur_rows: list[np.ndarray] = []
    event_id_rows: list[np.ndarray] = []
    channel_rows: list[np.ndarray] = []
    period_rows: list[np.ndarray] = []

    n_cols = int(analog_data_u16.shape[1]) if analog_data_u16.ndim == 2 else 0
    if n_cols <= 0:
        return None

    for jj, ch1 in enumerate(channel_ids_1based, start=1):
        col_idx = jj - 1
        if col_idx < 0 or col_idx >= n_cols:
            continue

        d = analog_data_u16[:, col_idx].astype(np.float64)
        d = _baseline_correct_by_epochs(d, merge_timestamps_sec, sr=sr)

        em_values: list[float] = []
        if merge_timestamps_sec is not None:
            ts = np.asarray(merge_timestamps_sec, dtype=np.float64)
            if ts.ndim == 2 and ts.shape[1] == 2:
                for start_sec, stop_sec in ts:
                    s = max(0, int(np.floor(float(start_sec) * float(sr))))
                    e = min(d.size, int(np.floor(float(stop_sec) * float(sr))))
                    if e - s > 1:
                        em_values.append(_effectiveness_metric(d[s:e]))
        if not em_values:
            em_values = [_effectiveness_metric(d)]
        if max(em_values) < ANALOG_EM_THRESHOLD:
            d[:] = 0.0

        if np.any(d < 0):
            d = d - np.min(d)

        sampled = d[::100] / 0.6745 if d.size > 0 else np.array([], dtype=np.float64)
        thr = float(150.0 * np.median(sampled)) if sampled.size else 0.0
        if thr == 0.0 or not np.any(d > thr):
            thr = float(4.5 * np.std(d)) if d.size else 0.0

        d_bin = d > thr
        on_idx, off_idx = _find_rising_falling_edges_matlab(d_bin)
        on_t = on_idx.astype(np.float64) / float(sr)
        off_t = off_idx.astype(np.float64) / float(sr)
        intervals = _pair_intervals(on_t, off_t)
        if intervals.size == 0:
            continue

        baseline_d = np.int32(np.median(d[::100])) if d.size else np.int32(0)
        vals: list[float] = []
        keep_mask: list[bool] = []
        for start_t, stop_t in intervals.T:
            s = max(0, int(np.rint(start_t * sr)) - 1)
            e = min(d.size - 1, int(np.rint(stop_t * sr)) - 1)
            if e < s:
                vals.append(0.0)
                keep_mask.append(False)
                continue
            med_val = float(np.median(np.int32(d[s : e + 1])) - baseline_d)
            vals.append(med_val)
            keep_mask.append(bool((med_val >= (thr - baseline_d) * 0.4) and (start_t >= 0.0)))

        keep = np.asarray(keep_mask, dtype=bool)
        if not np.any(keep):
            continue

        kept = intervals[:, keep]
        amps = np.asarray(vals, dtype=np.float64)[keep]
        durs = kept[1, :] - kept[0, :]
        event_ids = np.full(kept.shape[1], float(jj), dtype=np.float64)
        channels = np.full(kept.shape[1], float(ch1), dtype=np.float64)
        periods = _build_periods(kept, lag_sec=ANALOG_PULSE_PERIOD_LAG_SEC)

        pul_rows.append(kept.T)
        amp_rows.append(amps.reshape(-1, 1))
        dur_rows.append(durs.reshape(-1, 1))
        event_id_rows.append(event_ids.reshape(-1, 1))
        channel_rows.append(channels.reshape(-1, 1))
        if periods.size:
            period_rows.append(periods)

    if not pul_rows:
        return None

    pulses = {
        "timestamps": np.vstack(pul_rows).astype(np.float64),
        "amplitude": np.vstack(amp_rows).astype(np.float64),
        "duration": np.vstack(dur_rows).astype(np.float64),
        "eventGroupID": np.vstack(event_id_rows).astype(np.float64),
        "analogChannel": np.vstack(channel_rows).astype(np.float64),
        "intsPeriods": np.vstack(period_rows).astype(np.float64) if period_rows else np.empty((0, 2), dtype=np.float64),
    }

    sort_idx = np.argsort(pulses["timestamps"][:, 0], kind="mergesort")
    for key in ("timestamps", "amplitude", "duration", "eventGroupID", "analogChannel"):
        pulses[key] = pulses[key][sort_idx, :]
    if pulses["intsPeriods"].size:
        period_sort_idx = np.argsort(pulses["intsPeriods"][:, 0], kind="mergesort")
        pulses["intsPeriods"] = pulses["intsPeriods"][period_sort_idx, :]

    if min_dur_sec is not None and float(min_dur_sec) > 0.0:
        keep = np.flatnonzero(pulses["duration"].reshape(-1) > float(min_dur_sec))
        pulses["timestamps"] = pulses["timestamps"][keep, :]
        pulses["amplitude"] = pulses["amplitude"][keep, :]
        pulses["duration"] = pulses["duration"][keep, :]
        pulses["eventGroupID"] = pulses["eventGroupID"][keep, :]
        pulses["analogChannel"] = pulses["analogChannel"][keep, :]

    if sess_epochs_1based:
        ts = np.asarray(merge_timestamps_sec, dtype=np.float64)
        if ts.ndim == 2 and ts.shape[1] == 2:
            keep_rows: list[int] = []
            for epoch_id in sess_epochs_1based:
                idx = int(epoch_id) - 1
                if idx < 0 or idx >= ts.shape[0]:
                    continue
                start, stop = float(ts[idx, 0]), float(ts[idx, 1])
                mask = (pulses["timestamps"][:, 0] >= start) & (pulses["timestamps"][:, 1] <= stop)
                keep_rows.extend(np.flatnonzero(mask).tolist())
            if keep_rows:
                keep = np.asarray(sorted(set(keep_rows)), dtype=np.int64)
                pulses["timestamps"] = pulses["timestamps"][keep, :]
                pulses["amplitude"] = pulses["amplitude"][keep, :]
                pulses["duration"] = pulses["duration"][keep, :]
                pulses["eventGroupID"] = pulses["eventGroupID"][keep, :]
                pulses["analogChannel"] = pulses["analogChannel"][keep, :]
            else:
                return None

    return pulses


def _build_empty_cell_row(n_cells: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.empty((1, n_cells), dtype=object)
    for i in range(n_cells):
        out[0, i] = np.empty(shape, dtype=np.float64)
    return out


def _build_digital_in_struct(
    *,
    digital_data_u16: np.ndarray,
    digital_channels_1based: list[int] | None,
    fs: float,
    word_channels: int,
) -> tuple[dict[str, Any], bool]:
    n_channels = 16
    selected = set(digital_channels_1based or [])

    timestamps_on = _build_empty_cell_row(n_channels, (0, 1))
    timestamps_off = _build_empty_cell_row(n_channels, (0, 1))
    ints = _build_empty_cell_row(n_channels, (2, 0))
    dur = _build_empty_cell_row(n_channels, (1, 0))
    ints_periods = _build_empty_cell_row(n_channels, (0, 2))

    if digital_data_u16.ndim == 1:
        digital_data_u16 = digital_data_u16.reshape(-1, 1)

    has_any = False
    for ch1 in range(1, n_channels + 1):
        if selected and ch1 not in selected:
            continue

        if word_channels <= 1:
            bit = ((digital_data_u16[:, 0] & (1 << (ch1 - 1))) > 0).astype(np.uint8)
        else:
            ch0 = ch1 - 1
            if ch0 >= digital_data_u16.shape[1]:
                continue
            bit = (digital_data_u16[:, ch0] != 0).astype(np.uint8)

        on_idx, off_idx = _find_rising_falling_edges_matlab(bit)
        if on_idx.size == 0 and off_idx.size == 0:
            continue

        has_any = True
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
        ch0 = ch1 - 1
        timestamps_on[0, ch0] = on_t.reshape(-1, 1)
        timestamps_off[0, ch0] = off_t.reshape(-1, 1)
        ints[0, ch0] = d
        dur[0, ch0] = (d[1, :] - d[0, :]).reshape(1, -1)
        ints_periods[0, ch0] = channel_periods

    return (
        {
            "timestampsOn": timestamps_on,
            "timestampsOff": timestamps_off,
            "ints": ints,
            "dur": dur,
            "intsPeriods": ints_periods,
        },
        has_any,
    )


def _save_analog_plot(output_dir: Path, analog_data_u16: np.ndarray, channel_ids_1based: list[int], sr: float) -> None:
    out_dir = output_dir / "pulses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "analogPulsesDetection.png"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        out_path.touch()
        return

    n_samples = int(analog_data_u16.shape[0])
    if n_samples == 0:
        out_path.touch()
        return
    step = max(1, n_samples // 5000)
    xt = np.arange(0, n_samples, step, dtype=np.float64) / float(sr)

    n_show = min(max(1, len(channel_ids_1based)), 8)
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        col = i if i < analog_data_u16.shape[1] else 0
        axes[i].plot(xt, analog_data_u16[::step, col], linewidth=0.6)
        axes[i].set_ylabel(f"Ch{channel_ids_1based[i] if i < len(channel_ids_1based) else i + 1}")
    axes[-1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_digital_plot(output_dir: Path, digital_in: dict[str, Any]) -> None:
    out_dir = output_dir / "Pulses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "digitalIn.png"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        out_path.touch()
        return

    ints_periods = digital_in["intsPeriods"]
    fig, ax = plt.subplots(figsize=(12, 4))
    y = 0
    for i in range(16):
        periods = np.asarray(ints_periods[0, i]) if isinstance(ints_periods, np.ndarray) else np.empty((0, 2))
        if periods.size:
            for start, stop in periods:
                if np.isnan(stop):
                    continue
                ax.plot([start, stop], [y, y], color="k", linewidth=3)
        y += 1
    ax.set_ylim(-1, 16)
    ax.set_xlabel("s")
    ax.set_ylabel("Digital channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _find_existing_digital_events_file(output_dir: Path) -> Path | None:
    exact = output_dir / "digitalIn.events.mat"
    if exact.exists():
        return exact
    candidates = sorted(output_dir.glob("*DigitalIn.events.mat"))
    if candidates:
        return candidates[0]
    return None


def export_analog_digital_events(
    *,
    output_dir: Path,
    basename: str,
    analog_inputs: bool,
    analog_channels: list[int] | None,
    analog_num_channels: int,
    analog_active_channels_1based: list[int] | None = None,
    digital_inputs: bool,
    digital_channels: list[int] | None,
    digital_word_channels: int,
    digital_active_channels_1based: list[int] | None = None,
    sr: float,
    analog_sr: float | None,
    digital_sr: float | None,
    analog_dat_path: Path | None,
    digital_dat_path: Path | None,
    merge_timestamps_sec: np.ndarray | None,
    overwrite: bool,
    pulse_min_dur_sec: float | None = None,
    pulse_sess_epochs_1based: list[int] | None = None,
) -> tuple[list[Path], list[Path]]:
    del analog_channels  # neurocode preprocessSession computes analogInp/pulses on all active ADC channels
    del digital_active_channels_1based  # neurocode getDigitalIn decodes 16-bit word space

    analog_paths: list[Path] = []
    digital_paths: list[Path] = []
    analog_sr_eff = float(analog_sr) if analog_sr is not None else float(sr)
    digital_sr_eff = float(digital_sr) if digital_sr is not None else float(sr)

    if analog_inputs and analog_dat_path is not None and analog_dat_path.exists():
        behavior_out = output_dir / f"{basename}.analogInput.behavior.mat"
        pulses_out = output_dir / f"{basename}.pulses.events.mat"
        recompute = overwrite or not behavior_out.exists() or not pulses_out.exists()

        if recompute:
            raw = np.fromfile(analog_dat_path, dtype=np.uint16)
            n_ch = int(analog_num_channels) if int(analog_num_channels) > 0 else 1
            if raw.size % n_ch != 0:
                raise ValueError(
                    f"analogin.dat size is not divisible by analog channel count: size={raw.size}, n_ch={n_ch}"
                )
            analog_data = raw.reshape(-1, n_ch)
            active_channels_1based = (
                [int(ch) for ch in analog_active_channels_1based]
                if analog_active_channels_1based
                else [i + 1 for i in range(n_ch)]
            )

            analog_inp = _build_analog_behavior_struct(
                analog_data_u16=analog_data,
                active_channels_1based=active_channels_1based,
                sampling_rate=ANALOG_BEHAVIOR_DEFAULT_FS,
            )
            savemat(behavior_out, {"analogInp": analog_inp}, do_compression=True)
            _save_analog_plot(output_dir, analog_data, active_channels_1based, analog_sr_eff)

            pulses = _detect_analog_pulses(
                analog_data_u16=analog_data,
                channel_ids_1based=active_channels_1based,
                sr=analog_sr_eff,
                merge_timestamps_sec=merge_timestamps_sec,
                min_dur_sec=pulse_min_dur_sec,
                sess_epochs_1based=pulse_sess_epochs_1based,
            )
            if pulses is None:
                if pulses_out.exists():
                    pulses_out.unlink()
            else:
                savemat(pulses_out, {"pulses": pulses}, do_compression=True)

        if behavior_out.exists():
            analog_paths.append(behavior_out)
        if pulses_out.exists():
            analog_paths.append(pulses_out)

    if digital_inputs and digital_dat_path is not None and digital_dat_path.exists():
        existing_digital = _find_existing_digital_events_file(output_dir)
        if (not overwrite) and existing_digital is not None:
            digital_paths.append(existing_digital)
        else:
            digital_out = output_dir / "digitalIn.events.mat"
            raw = np.fromfile(digital_dat_path, dtype=np.uint16)
            word_ch = int(digital_word_channels) if int(digital_word_channels) > 0 else 1
            if raw.size % word_ch != 0:
                raise ValueError(
                    f"digitalin.dat size is not divisible by digital word channel count: size={raw.size}, n_ch={word_ch}"
                )
            digital_data = raw.reshape(-1, word_ch)

            selected_0based = _normalize_channel_indices(digital_channels, 16) if digital_channels else []
            selected_1based = [i + 1 for i in selected_0based] if selected_0based else None
            digital_in, has_any = _build_digital_in_struct(
                digital_data_u16=digital_data,
                digital_channels_1based=selected_1based,
                fs=digital_sr_eff,
                word_channels=word_ch,
            )
            if has_any:
                savemat(digital_out, {"digitalIn": digital_in}, do_compression=True)
                _save_digital_plot(output_dir, digital_in)
                digital_paths.append(digital_out)
            elif digital_out.exists():
                digital_out.unlink()

    return analog_paths, digital_paths
