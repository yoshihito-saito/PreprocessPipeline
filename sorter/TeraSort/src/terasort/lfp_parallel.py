"""Run optional LFP export alongside sorting without holding the sorter thread."""

from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

EXPECTED_CLUSTER_SHA256 = "b7e1c0f2d8be57fa5e1235baf830fcd8cf3e5557df6bdb3581fc23140d18ff76"


def _validated_cluster_module():
    if importlib.metadata.version("kilosort") != "4.1.7":
        raise RuntimeError("Parallel LFP hook is validated only for Kilosort 4.1.7")
    module = importlib.import_module("kilosort.run_kilosort")
    source_hash = hashlib.sha256(inspect.getsource(module.cluster_spikes).encode()).hexdigest()
    if source_hash != EXPECTED_CLUSTER_SHA256:
        raise RuntimeError("Kilosort clustering entry point changed; parallel LFP hook disabled")
    return module


@contextmanager
def parallel_lfp(filename, output, settings, *, passband_hz=500.0, workers=8):
    """Start LFP after spike detection, during final clustering; then wait.

    This overlaps work but reads the source through a separate OS file handle.
    It does not guarantee that overall sorting is free of CPU or I/O contention.
    """
    source, target = Path(filename).resolve(), Path(output).resolve()
    if not source.is_file() or source == target:
        raise ValueError("LFP needs an existing source and a distinct output")
    if any(Path(str(target) + suffix).exists() for suffix in
           ("", ".json", ".json.partial", ".partial", ".progress.json", ".log")):
        raise FileExistsError("LFP output, checkpoint, or log already exists")
    fs = settings.get("fs")
    channels = settings.get("n_chan_bin")
    if not isinstance(fs, (int, float)) or not math.isfinite(fs) or int(fs) != fs or fs <= 1250:
        raise ValueError("LFP requires an integer input sample rate above 1250 Hz")
    if not isinstance(channels, int) or channels < 1:
        raise ValueError("LFP requires positive n_chan_bin")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("LFP workers must be positive")
    if not 0 < passband_hz < 625:
        raise ValueError("LFP passband must be between 0 and 625 Hz")
    scale = settings.get("scale")
    if scale is not None and (not np.isscalar(scale) or
                              not np.isfinite(scale) or scale <= 0):
        raise ValueError("Parallel LFP requires a positive scalar scale")
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(target) + ".log")
    command = [sys.executable, "-m", "terasort.cli", "lfp",
               "--filename", str(source), "--output", str(target),
               "--sample-rate", str(int(fs)), "--n-channels", str(channels),
               "--passband-hz", str(passband_hz), "--workers", str(workers)]
    if scale is not None:
        command += ["--scale-uv-per-count", str(scale)]
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) |
             getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)) if sys.platform == "win32" else 0
    module = _validated_cluster_module()
    original = module.cluster_spikes
    process = None

    def start_and_cluster(*args, **kwargs):
        nonlocal process
        if process is None:
            try:
                with log_path.open("xb") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=flags)
            except BaseException:
                log_path.unlink(missing_ok=True)
                raise
        return original(*args, **kwargs)

    module.cluster_spikes = start_and_cluster
    try:
        yield
    except BaseException:
        if process is not None:
            process.terminate()
            process.wait()
        raise
    else:
        if process is None:
            raise RuntimeError("Kilosort finished without final clustering; LFP was not started")
        if process.wait() != 0:
            raise RuntimeError(f"Parallel LFP export failed; inspect {log_path}")
        log_path.unlink()
    finally:
        module.cluster_spikes = original
