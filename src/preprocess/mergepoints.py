from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .metafile import MergePointsData


def compute_mergepoints(
    dat_paths: list[Path],
    n_channels: int,
    dtype: str,
    sampling_frequency: float,
    foldernames: list[str],
) -> MergePointsData:
    itemsize = np.dtype(dtype).itemsize
    n_samp = []
    for p in dat_paths:
        bytes_ = p.stat().st_size
        n_samp.append(int(bytes_ // (n_channels * itemsize)))

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


def save_mergepoints_events_mat(path: Path, data: MergePointsData) -> Path:
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
    savemat(path, {"MergePoints": merge_points}, do_compression=True)
    return path
