"""Fixed-budget Kilosort4 preview identities for the streaming session sorter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .api import run_kilosort
from .seed_cluster_gpu import cuda_seed_cluster_updates
from .session_calibration import select_calibration_windows
from .session_seed import stage_seed_previews, rebuild_seed_models
from .session_contact_policy import BAD_CONTACT_POLICY
from .kilosort_settings import validate_seed_settings


def learn_seed_models(probe, day_id, output_root, *, budget_seconds=60.,
                      window_count=3, preprocessing_backend="cpu",
                      cuda_cluster_updates_enabled=False,
                      keep_preview=True, exclude_bad_contacts=False, nt=61,
                      ks4_settings=None, ks4_backend="deep_tiled"):
    """Learn from bounded raw windows, then rebuild in the session frame.

    ``output_root`` must be new. The preview cap is per probe/day and does not
    depend on the dense source duration. Callers may atomically publish this
    complete directory after the function returns.
    """
    from .session_models import validate_nt
    nt = validate_nt(nt)
    ks4_settings = validate_seed_settings(ks4_settings)
    if ks4_backend not in ("auto", "standard", "deep_tiled", "cublas"):
        raise ValueError("Unsupported KS4 seed backend")
    if not isinstance(exclude_bad_contacts, bool):
        raise ValueError('Bad-contact exclusion must be boolean')
    if exclude_bad_contacts and len(set(probe.bad_channels)) == probe.n_channels:
        raise ValueError('No usable channels for seed learning')
    if not 6 <= budget_seconds <= 180:
        raise ValueError("Calibration budget must be 6–180 seconds")
    if (isinstance(window_count, bool) or not isinstance(window_count, int)
            or not 3 <= window_count <= 12):
        raise ValueError("Calibration window count must be 3–12")
    if preprocessing_backend not in ("cpu", "cuda"):
        raise ValueError("Seed preprocessing backend must be cpu or cuda")
    windows = select_calibration_windows(
        probe, day_id, window_seconds=budget_seconds/window_count,
        target_windows=window_count)
    raw_bytes = sum(stop-start for start, stop in windows) * probe.n_channels * 2
    if raw_bytes > 5 * 1024**3:
        raise ValueError("Preview exceeds the 5 GiB scratch cap; lower budget_seconds")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previews, io_report = stage_seed_previews(probe, windows, root, nt=nt)
    preview_seconds = time.perf_counter() - started
    print(json.dumps({"stage": "preview_written", "raw_bytes": raw_bytes,
                      "windows": windows}), flush=True)
    ksprobe = dict(chanMap=np.arange(probe.n_channels, dtype=np.int32),
                   xc=probe.x_um.astype(np.float32),
                   yc=probe.y_um.astype(np.float32),
                   kcoords=probe.shank.astype(np.float32),
                   n_chan=probe.n_channels)
    if exclude_bad_contacts:
        active = np.setdiff1d(np.arange(probe.n_channels, dtype=np.int32), probe.bad_channels)
        if not len(active):
            raise ValueError('No usable channels for seed learning')
        for key in ('chanMap', 'xc', 'yc', 'kcoords'):
            ksprobe[key] = ksprobe[key][active]
        ksprobe['n_chan'] = len(active)
    sorting_started = time.perf_counter()
    preprocessed = getattr(probe, "input_is_preprocessed", False)
    if preprocessed and "highpass_cutoff" in ks4_settings:
        raise ValueError("Preprocessed seed input does not use highpass_cutoff")
    with cuda_seed_cluster_updates(enabled=cuda_cluster_updates_enabled) as updates:
        run_kilosort(
            dict(n_chan_bin=probe.n_channels, fs=probe.sample_rate_hz,
                 scale=probe.gain_uv_per_count, nblocks=0, **ks4_settings,
                 **({"nt": nt} if nt != 61 else {})),
            filename=[item.path for item in previews], probe=ksprobe,
            results_dir=root / "kilosort", backend=ks4_backend,
            input_is_preprocessed=preprocessed,
            data_dtype="int16", verbose_console=True,
            **({"do_CAR": False} if preprocessed else {}))
    sorting_seconds = time.perf_counter() - sorting_started
    effective_ks4 = json.loads((root / "kilosort" / "terasort_settings.json").read_text(encoding="utf-8"))
    rebuilding_started = time.perf_counter()
    preprocessor = None
    if preprocessing_backend == "cuda":
        import torch
        torch.cuda.empty_cache()
        from .session_preprocess_gpu import CudaPreprocessor
        gpu = CudaPreprocessor(probe)
        gpu.cp.get_default_memory_pool().set_limit(size=2*1024**3)
        preprocessor = gpu.host
    models, model_report = rebuild_seed_models(
        probe, previews, root / "kilosort", preprocessor=preprocessor,
        exclude_bad_contacts=exclude_bad_contacts, nt=nt)
    rebuild_seconds = time.perf_counter() - rebuilding_started
    seeds = root / "seeds.npz"
    with seeds.open("xb") as handle:
        np.savez_compressed(handle, waveforms=models.waveforms,
                            channels=models.channels, anchors=models.anchors,
                            assigned=models.assigned, version=models.version)
    if not keep_preview:
        for preview in previews:
            for path in (preview.path, preview.prefix, preview.suffix):
                target = path.resolve(strict=True)
                if target.parent != root:
                    raise ValueError("Preview cleanup path left the seed directory")
                target.unlink()
        scratch = root / "kilosort"
        if scratch.exists():
            if scratch.is_symlink() or scratch.resolve(strict=True).parent != root:
                raise ValueError("Kilosort cleanup path left the seed directory")
            shutil.rmtree(scratch)
    seed_hash = hashlib.sha256(seeds.read_bytes()).hexdigest()
    report = dict(
        method="bounded_kilosort4_identities_session_waveforms_v1",
        preprocessing=getattr(probe, "preprocessing_id", "terasort-session-v1"), probe_id=probe.probe_id,
        day_id=day_id, windows=windows,
        requested_budget_seconds=float(budget_seconds),
        requested_window_count=window_count,
        preprocessing_backend=preprocessing_backend,
        requested_cuda_cluster_updates=cuda_cluster_updates_enabled,
        cuda_cluster_updates=updates,
        preview_seconds=preview_seconds, sorting_seconds=sorting_seconds,
        rebuild_seconds=rebuild_seconds, seeds_sha256=seed_hash,
        preview_retained=keep_preview,
        **io_report, **model_report,
        effective_kilosort=effective_ks4,
        wall_seconds=time.perf_counter() - started)
    if exclude_bad_contacts:
        report['bad_contact_neighborhoods'] = BAD_CONTACT_POLICY
    if nt != 61:
        report["nt"] = nt
    if ks4_settings:
        report["ks4_settings"] = ks4_settings
    if ks4_backend != "deep_tiled":
        report["ks4_backend"] = ks4_backend
    (root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "seeds_complete",
                      "units": model_report["seed_units"],
                      "wall_seconds": report["wall_seconds"]}), flush=True)
    return report
