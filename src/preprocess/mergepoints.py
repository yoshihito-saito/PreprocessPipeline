from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import savemat
from .io import atomic_savemat, validate_mat_output

from .metafile import MergePointsData


def compute_mergepoints(
    dat_paths: list[Path],
    n_channels: int,
    dtype: str,
    sampling_frequency: float,
    foldernames: list[str],
    sample_counts: list[int] | None = None,
) -> MergePointsData:
    if sample_counts is not None:
        if len(sample_counts) != len(dat_paths):
            raise ValueError(
                "sample_counts must align with dat_paths for mergepoints: "
                f"{len(sample_counts)} != {len(dat_paths)}"
            )
        n_samp = [int(n) for n in sample_counts]
    else:
        itemsize = np.dtype(dtype).itemsize
        frame_bytes = int(n_channels) * int(itemsize)
        if frame_bytes <= 0:
            raise ValueError(f"Invalid frame size for mergepoints: n_channels={n_channels}, dtype={dtype}")
        n_samp = []
        for p in dat_paths:
            bytes_ = int(p.stat().st_size)
            if bytes_ % frame_bytes != 0:
                raise ValueError(
                    f"Dat size is not divisible by frame size for mergepoints: "
                    f"{p} size={bytes_}, frame_bytes={frame_bytes}"
                )
            n_samp.append(int(bytes_ // frame_bytes))

    n_samp_arr = np.asarray(n_samp, dtype=np.int64)
    cumsum = np.cumsum(n_samp_arr)
    starts = np.concatenate(([0], cumsum[:-1]))

    timestamps_samples = np.column_stack((starts, cumsum))
    timestamps_sec = timestamps_samples / float(sampling_frequency)
    firstlast = np.column_stack((np.zeros_like(n_samp_arr), n_samp_arr))

    return MergePointsData(
        timestamps_sec=timestamps_sec,
        timestamps_samples=timestamps_samples,
        firstlasttimepoints_samples=firstlast,
        foldernames=foldernames,
    )


def save_mergepoints_events_mat(path: Path, data: MergePointsData, *, overwrite: bool = True) -> Path:
    if Path(path).exists() and not overwrite:
        existing = validate_mat_output(Path(path), "MergePoints")["MergePoints"]
        try:
            timestamps = np.asarray(existing["timestamps"], dtype=float)
            timestamps_samples = np.asarray(existing["timestamps_samples"], dtype=float)
            firstlast = np.asarray(existing["firstlasttimpoints_samples"], dtype=float)
            foldernames = [str(item) for item in np.asarray(existing["foldernames"], dtype=object).reshape(-1)]
        except Exception as exc:
            raise ValueError(f"Invalid existing MergePoints output: {path}") from exc
        same = (
            timestamps.shape == np.asarray(data.timestamps_sec).shape
            and np.allclose(timestamps, data.timestamps_sec, equal_nan=True)
            and timestamps_samples.shape == np.asarray(data.timestamps_samples).shape
            and np.allclose(timestamps_samples, data.timestamps_samples, equal_nan=True)
            and firstlast.shape == np.asarray(data.firstlasttimepoints_samples).shape
            and np.allclose(firstlast, data.firstlasttimepoints_samples, equal_nan=True)
            and foldernames == [str(item) for item in data.foldernames]
        )
        if not same:
            raise ValueError(f"Existing MergePoints output is incompatible with current inputs: {path}")
        return Path(path)
    merge_points = {
        "timestamps": data.timestamps_sec,
        # neurocode MAT output stores these as MATLAB double
        "timestamps_samples": np.asarray(data.timestamps_samples, dtype=np.float64),
        "firstlasttimpoints_samples": np.asarray(data.firstlasttimepoints_samples, dtype=np.float64),
        "foldernames": np.asarray(data.foldernames, dtype=object),
        "detectorinfo": {
            "detectorname": "preprocessSession.py",
            "detectiondate": datetime.now().strftime("%Y-%m-%d"),
        },
    }
    return atomic_savemat(Path(path), {"MergePoints": merge_points}, required_key="MergePoints")
