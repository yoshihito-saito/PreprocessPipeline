"""Dependency and CUDA smoke checks for the Windows installer and launcher."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import struct
import sys


def python_info() -> int:
    print(json.dumps({
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "bits": struct.calcsize("P") * 8,
    }))
    return 0


def torch_info() -> int:
    import torch

    print(json.dumps({
        "version": torch.__version__.split("+", 1)[0],
        "cuda": torch.version.cuda,
    }))
    return 0


def package_check() -> int:
    errors = []
    try:
        import torch
        import cupy
        import kilosort
        import terasort
        import terasort.web
    except Exception as exc:
        errors.append(f"Package import failed: {type(exc).__name__}: {exc}")
    else:
        try:
            kilosort_version = importlib.metadata.version("kilosort")
        except importlib.metadata.PackageNotFoundError:
            kilosort_version = "missing"
        torch_version = torch.__version__.split("+", 1)[0]
        if kilosort_version != "4.1.7":
            errors.append(f"Kilosort must be 4.1.7; found {kilosort_version}")
        if torch_version != "2.10.0" or torch.version.cuda != "12.8":
            errors.append(
                f"PyTorch CUDA build must be 2.10.0/cu128; found {torch.__version__} "
                f"with CUDA {torch.version.cuda}"
            )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Required packages are installed.")
    return 0


def progress_smoke() -> int:
    """Confirm this install can load the three-phase progress hooks."""
    import importlib

    from terasort.job_progress import JobProgress
    from terasort.kilosort_progress import SOURCES, instrument

    tracker = JobProgress()
    failures = []
    for (module_name, function_name) in SOURCES:
        try:
            module = importlib.import_module("kilosort." + module_name)
            instrument(getattr(module, function_name), module_name, tracker)
        except (AttributeError, OSError, SyntaxError, TypeError, ValueError) as exc:
            failures.append(f"{module_name}.{function_name}: {exc}")
    if failures:
        print("Progress hooks unavailable:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print(f"Three-phase progress hooks ready ({len(SOURCES)} Kilosort counters).")
    return 0


def cuda_smoke() -> int:
    import cupy as cp
    import torch
    import terasort
    from terasort.candidates.detectors import CudaDetector

    if not torch.cuda.is_available():
        print("PyTorch cannot access a CUDA device.", file=sys.stderr)
        return 1
    signal = cp.zeros((16, 2), dtype=cp.float32)
    signal[3, 0] = 5
    found = cp.asnumpy(CudaDetector().detect(signal, floor=3))
    print(json.dumps({
        "terasort": terasort.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "detected_indices": found.tolist(),
    }))
    return 0 if found.tolist() == [6] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("python-info", "torch-info", "packages", "progress-smoke", "cuda-smoke"))
    action = parser.parse_args().check
    return {
        "python-info": python_info,
        "torch-info": torch_info,
        "packages": package_check,
        "progress-smoke": progress_smoke,
        "cuda-smoke": cuda_smoke,
    }[action]()


if __name__ == "__main__":
    raise SystemExit(main())
