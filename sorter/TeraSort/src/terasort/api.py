"""A drop-in Kilosort API with scoped, source-guarded CUDA hot paths."""

from contextlib import ExitStack
from pathlib import Path
import json

import numpy as np


def available_backends():
    """Report selectable backends without creating a CUDA context."""
    from .kilosort_cublas import native_bridge_path, cublas_library_path
    bridge = native_bridge_path()
    native = bridge is not None and bridge.is_file()
    if native:
        try:
            import torch
        except ImportError:
            native = False
        else:
            native = ((torch.version.cuda or "").startswith("12.") and
                      cublas_library_path(torch) is not None)
    return ("standard", "deep_tiled", "cublas") if native else ("standard", "deep_tiled")


def _select_backend(backend, device, settings):
    if backend not in ("auto", "standard", "deep_tiled", "cublas"):
        raise ValueError(f"Unknown backend {backend!r}")
    if backend == "standard":
        return backend
    specialized_geometry = (settings.get("n_templates", 6) == 6 and
                            settings.get("nt", 61) in (41, 61))
    import torch
    requested_device = torch.device(device) if device is not None else None
    cuda = torch.cuda.is_available() and (requested_device is None or requested_device.type == "cuda")
    if backend == "auto":
        return "cublas" if cuda and specialized_geometry and "cublas" in available_backends() else "standard"
    if not cuda:
        raise RuntimeError(f"{backend} requires a CUDA-capable PyTorch installation and GPU")
    if backend not in available_backends():
        raise RuntimeError("The native cuBLAS bridge or CUDA 12 library is unavailable on this platform")
    if backend == "cublas" and not specialized_geometry:
        raise ValueError("The cuBLAS path requires n_templates=6 and nt=41 or 61")
    return backend


def _use_int16_reader(filename, file_object, data_dtype, backend, fast_int16):
    if not fast_int16 or backend == "standard" or file_object is not None:
        return False
    if filename is None or (isinstance(filename, (list, tuple)) and not filename):
        return False
    return np.dtype("int16" if data_dtype is None else data_dtype) == np.dtype("int16")


def run_kilosort(settings, probe=None, probe_name=None, filename=None,
                 data_dir=None, file_object=None, results_dir=None,
                 data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                 progress_bar=None, save_extra_vars=False, clear_cache=False,
                 save_preprocessed_copy=False, bad_channels=None, shank_idx=None,
                 verbose_console=False, verbose_log=False, torch_thread_lim=None,
                 *, backend="auto", fast_int16=True, input_is_preprocessed=False,
                 skip_drift_correction=False, lfp_output=None,
                 lfp_passband_hz=500.0, lfp_workers=8,
                 stage_dir=None, read_cache_dir=None, read_cache_mb=4096, read_cache_slots=3,
                 read_cache_workers=4, read_cache_retain=False, read_cache_retain_gb=0.):
    """Run Kilosort with the same input arguments, return tuple and Phy files.

    ``backend='auto'`` selects the tested Windows x64 cuBLAS path when CUDA is
    available and otherwise uses Kilosort's standard detection and matching
    algorithms. ``backend='standard'`` disables those alternate detection and
    matching paths. On verified Kilosort 4.1.7 installs, both modes still use
    TeraSort's guarded clustering gather and, with CUDA, GPU neighbor search.
    FAISS CPU work is capped at eight threads unless the process already has
    a lower limit; the previous setting is restored after the run.
    ``deep_tiled`` uses the portable CuPy CUDA kernels, while ``cublas`` requires
    the platform-native bridge.
    The INT16 reader supports named read-only INT16 binaries, including ordered
    Kilosort multi-file sessions.
    ``input_is_preprocessed=True`` bypasses high-pass filtering and CAR,
    retaining KS4 whitening, drift estimation and learning.
    ``skip_drift_correction=True`` sets Kilosort's ``nblocks=0`` for this run
    without changing the caller's settings dictionary. This skips drift
    estimation and its extra detection pass; use it only when appropriate for
    the recording's motion.
    ``lfp_output`` starts an optional lower-priority CPU LFP export after both
    spike-detection stages, during final clustering, then waits before returning.
    It reads the raw file separately and may still contend for CPU or disk.
    ``stage_dir`` copies all named sources once to a new local scratch directory
    before sorting. The copies remain there after the run.
    ``read_cache_retain=True`` retains a disk-budgeted source prefix across
    Kilosort passes while recycling rolling slots for the tail. Set
    ``read_cache_retain_gb`` to cap that prefix; zero uses available space with
    a free-space reserve. Retained blocks are temporary, not a restart cache.
    """
    import kilosort
    if not isinstance(input_is_preprocessed, bool):
        raise ValueError("input_is_preprocessed must be boolean")
    if input_is_preprocessed and "highpass_cutoff" in (settings or {}):
        raise ValueError("Preprocessed KS4 input does not use highpass_cutoff")
    if input_is_preprocessed and lfp_output is not None:
        raise ValueError("LFP export requires raw input")
    from .session import prepare_session, write_session_manifest

    if isinstance(filename, tuple):
        filename = list(filename)
    session = None
    if isinstance(filename, list) and len(filename) > 1:
        if results_dir is None:
            raise ValueError("A multi-file session requires results_dir for its source map")
        session = prepare_session(filename, settings or {}, "int16" if data_dtype is None else data_dtype)

    run_settings = ({**(settings or {}), "nblocks": 0}
                    if skip_drift_correction else settings)
    if read_cache_retain and (read_cache_dir is None or lfp_output is not None):
        raise ValueError('Retaining source blocks requires a read cache and no separate LFP export; '
                         'use full staging to share one download with LFP')
    requested_settings = dict(run_settings or {})
    selected = _select_backend(backend, device, run_settings or {})
    if read_cache_dir is not None and filename is not None and file_object is None:
        from .source_location import is_network_source
        sources = filename if isinstance(filename, (list, tuple)) else [filename]
        if not any(is_network_source(p) for p in sources):
            print('TERASORT_READ_CACHE_BYPASS all sources are local', flush=True)
            read_cache_dir = None
    if read_cache_dir is not None:
        if stage_dir is not None or not _use_int16_reader(filename,file_object,data_dtype,selected,fast_int16):
            raise ValueError('Streaming read cache requires the fast CUDA INT16 reader and cannot combine with full staging')
        cache_path = Path(read_cache_dir).expanduser().resolve()
        sources = filename if isinstance(filename,(list,tuple)) else [filename]
        if any(cache_path.is_relative_to(Path(p).resolve().parent) for p in sources):
            raise ValueError('Read cache must be outside source folders')
        if results_dir is not None:
            result_path = Path(results_dir).expanduser().resolve()
            if cache_path.is_relative_to(result_path) or result_path.is_relative_to(cache_path):
                raise ValueError('Read cache and result directories must be separate')

    staging = None
    if stage_dir is not None:
        if filename is None or file_object is not None or results_dir is None:
            raise ValueError("Local staging requires named files and results_dir")
        if lfp_output is not None and isinstance(filename, list):
            raise ValueError("Parallel LFP with staging requires one source file")
        channels = (settings or {}).get("n_chan_bin")
        if type(channels) is not int or channels <= 0:
            raise ValueError("Local staging requires positive integer n_chan_bin")
        frame_bytes = channels * np.dtype("int16" if data_dtype is None else data_dtype).itemsize
        files = filename if isinstance(filename, list) else [filename]
        if any(Path(path).stat().st_size % frame_bytes for path in files):
            raise ValueError("Source size is not divisible by the configured channel frame")
        scratch = Path(stage_dir).expanduser().resolve()
        result_path = Path(results_dir).expanduser().resolve()
        if scratch.is_relative_to(result_path) or result_path.is_relative_to(scratch):
            raise ValueError("Staging directory and results directory must be separate")
        if result_path.exists() and (not result_path.is_dir() or any(result_path.iterdir())):
            raise FileExistsError("Staged sorting requires a new or empty results directory")
        from .staging import stage_inputs

        staged, staging = stage_inputs(
            files, scratch, progress=lambda percent: print(f"TERASORT_STAGING {percent}", flush=True))
        print(f"TERASORT_STAGING_SECONDS {staging['total_seconds']}", flush=True)
        filename = staged if len(staged) > 1 else staged[0]

    with ExitStack() as stack:
        from .preprocessed_input import preprocessed_seed_input
        stack.enter_context(preprocessed_seed_input(input_is_preprocessed))
        from .kilosort_cpu import bounded_faiss_threads
        stack.enter_context(bounded_faiss_threads())
        from .kilosort_gpu_cluster import vectorized_kilosort_neighbors
        stack.enter_context(vectorized_kilosort_neighbors(device=device))
        from .kilosort_cpu import vectorized_kilosort_gather
        stack.enter_context(vectorized_kilosort_gather())
        if selected == "cublas":
            from .kilosort_cublas import cublas_kilosort
            stack.enter_context(cublas_kilosort(device=device))
        elif selected == "deep_tiled":
            from .kilosort_tiled import tiled_kilosort
            stack.enter_context(tiled_kilosort(tile_samples=2048, deep=True))
        read_backend = None
        if _use_int16_reader(filename, file_object, data_dtype, selected, fast_int16):
            from .kilosort_int16 import native_int16_reader
            if read_cache_dir is None:
                read_backend = stack.enter_context(native_int16_reader(filename))
            else:
                read_backend = stack.enter_context(native_int16_reader(filename,read_cache_dir=read_cache_dir,
                    read_cache_mb=read_cache_mb,read_cache_slots=read_cache_slots,
                    read_cache_workers=read_cache_workers, read_cache_retain=read_cache_retain,
                    read_cache_retain_gb=read_cache_retain_gb))
        def prepare_dense_pass(label):
            if (label in ('Detection batches', 'Learned-detection batches') and read_backend is not None
                    and read_backend.read_cache is not None):
                first = filename[0] if isinstance(filename, (list, tuple)) else filename
                from .source_location import is_network_source
                if is_network_source(first):
                    read_backend.read_cache.prepare_pass(first)
        from .kilosort_progress import kilosort_progress
        stack.enter_context(kilosort_progress(
            results_dir, skip_drift=(run_settings or {}).get("nblocks", 1) == 0,
            staging_seconds=staging['total_seconds'] if staging else 0.,
            lfp=lfp_output is not None, on_task_start=prepare_dense_pass))
        if lfp_output is not None:
            if filename is None or isinstance(filename, (list, tuple)) or file_object is not None:
                raise ValueError("Parallel LFP requires one named INT16 source file")
            if np.dtype("int16" if data_dtype is None else data_dtype) != np.dtype("int16"):
                raise ValueError("Parallel LFP requires INT16 source data")
            from .lfp_parallel import parallel_lfp
            stack.enter_context(parallel_lfp(filename, lfp_output, run_settings or {},
                                             passband_hz=lfp_passband_hz,
                                             workers=lfp_workers))
        result = kilosort.run_kilosort(
            run_settings, probe=probe, probe_name=probe_name, filename=filename,
            data_dir=data_dir, file_object=file_object, results_dir=results_dir,
            data_dtype=data_dtype, do_CAR=False if input_is_preprocessed else do_CAR, invert_sign=invert_sign,
            device=device, progress_bar=progress_bar,
            save_extra_vars=save_extra_vars, clear_cache=clear_cache,
            save_preprocessed_copy=save_preprocessed_copy,
            bad_channels=bad_channels, shank_idx=shank_idx,
            verbose_console=verbose_console, verbose_log=verbose_log,
            torch_thread_lim=torch_thread_lim)
        if session is not None:
            write_session_manifest(results_dir, session)
        if staging is not None:
            from .staging import write_staging_record

            write_staging_record(results_dir, staging)
        if results_dir is not None and shank_idx is None:
            from kilosort.parameters import DEFAULT_SETTINGS
            output = Path(results_dir).resolve()
            effective = {key: result[0]["settings"][key] for key in DEFAULT_SETTINGS}
            def json_value(value):
                if isinstance(value, Path):
                    return str(value)
                if isinstance(value, np.ndarray):
                    return json_value(value.tolist())
                if isinstance(value, np.generic):
                    return json_value(value.item())
                if isinstance(value, dict):
                    return {key: json_value(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [json_value(item) for item in value]
                if isinstance(value, float) and not np.isfinite(value):
                    # Metadata only: native unbounded limits remain unchanged.
                    return str(value)
                return value
            (output / "terasort_settings.json").write_text(json.dumps(json_value({
                "requested_settings": requested_settings, "settings": effective,
                "requested_backend": backend, "backend": selected,
                "input_is_preprocessed": input_is_preprocessed,
                "do_CAR": False if input_is_preprocessed else bool(do_CAR),
            }), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if input_is_preprocessed:
                phy_params = output / "params.py"
                lines = phy_params.read_text(encoding="utf-8").splitlines()
                lines = ["hp_filtered = True" if line.startswith("hp_filtered") else line
                         for line in lines]
                phy_params.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result
