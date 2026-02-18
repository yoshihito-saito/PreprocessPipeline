from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat


def build_session_struct(
    *,
    basepath: Path,
    basename: str,
    dat_path: Path,
    lfp_path: Path | None,
    sr: float,
    sr_lfp: float | None,
    n_channels: int,
    bad_channels_1based: list[int],
    analog_event_paths: list[Path],
    digital_event_paths: list[Path],
) -> dict[str, Any]:
    session = {
        "general": {
            "name": basename,
            "basePath": str(basepath),
        },
        "extracellular": {
            "fileName": dat_path.name,
            "fileFormat": "dat",
            "sr": float(sr),
            "srLfp": float(sr_lfp) if sr_lfp is not None else np.nan,
            "nChannels": int(n_channels),
            "precision": "int16",
            "leastSignificantBit": 0.195,
            "lfpFileName": lfp_path.name if lfp_path is not None else "",
        },
        "channelTags": {
            "Bad": {
                "channels": np.asarray(bad_channels_1based, dtype=np.int64),
            }
        },
    }

    inputs = []
    for p in analog_event_paths:
        inputs.append({"file": p.name, "inputType": "adc"})
    for p in digital_event_paths:
        inputs.append({"file": p.name, "inputType": "dig"})
    if inputs:
        session["inputs"] = np.asarray(inputs, dtype=object)

    return session


def save_session_mat(path: Path, session_struct: dict[str, Any]) -> Path:
    savemat(path, {"session": session_struct}, do_compression=True)
    return path
