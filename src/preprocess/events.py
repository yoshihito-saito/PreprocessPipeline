from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat


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
            t = np.concatenate([np.arange(n, dtype=np.int64) for n in sample_counts]) if sample_counts else np.array([], dtype=np.int64)
            t.tofile(time_auto)
        out["time"] = time_auto

    return out


def export_analog_digital_events(
    *,
    output_dir: Path,
    basename: str,
    analog_inputs: bool,
    analog_channels: list[int] | None,
    digital_inputs: bool,
    digital_channels: list[int] | None,
    sr: float,
    analog_dat_path: Path | None,
    digital_dat_path: Path | None,
) -> tuple[list[Path], list[Path]]:
    analog_paths: list[Path] = []
    digital_paths: list[Path] = []

    if analog_inputs and analog_dat_path is not None and analog_dat_path.exists():
        out = output_dir / f"{basename}.analogin.timeseries.mat"
        n_ch = len(analog_channels) if analog_channels else 1
        raw = np.fromfile(analog_dat_path, dtype=np.int16)
        if raw.size % n_ch != 0:
            n_ch = 1
        ts = raw.reshape(-1, n_ch)
        savemat(
            out,
            {
                "analogin": {
                    "data": ts,
                    "timestamps": np.arange(ts.shape[0]) / sr,
                    "sr": sr,
                    "channels": np.asarray(analog_channels or list(range(n_ch))),
                }
            },
            do_compression=True,
        )
        analog_paths.append(out)

    if digital_inputs and digital_dat_path is not None and digital_dat_path.exists():
        out = output_dir / f"{basename}.digitalin.events.mat"
        raw = np.fromfile(digital_dat_path, dtype=np.uint16)
        n_ch = 1
        data = raw.reshape(-1, n_ch)

        timestamps = []
        channels = digital_channels or list(range(16))
        packed = data[:, 0]
        for ch in range(16):
            s = ((packed >> ch) & 1).astype(bool)
            on = np.flatnonzero(np.diff(s.astype(np.int8), prepend=0) == 1)
            off = np.flatnonzero(np.diff(s.astype(np.int8), append=0) == -1)
            n = min(len(on), len(off))
            intervals = np.column_stack((on[:n], off[:n])) / sr if n else np.empty((0, 2))
            timestamps.append(intervals)

        savemat(
            out,
            {
                "digitalIn": {
                    "timestamps": np.asarray(timestamps, dtype=object),
                    "channels": np.asarray(channels),
                    "sr": sr,
                }
            },
            do_compression=True,
        )
        digital_paths.append(out)

    return analog_paths, digital_paths
