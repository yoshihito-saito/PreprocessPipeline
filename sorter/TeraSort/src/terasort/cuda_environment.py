"""Runtime setup shared by TeraSort's optional CUDA backends."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def configure_cupy_cache() -> Path:
    """Choose a local kernel-cache directory before CuPy is imported.

    CuPy's default Windows cache may sit under a redirected or protected user
    profile. Kernel cache writes there can stall before any CUDA work starts.
    An explicit CUPY_CACHE_DIR always takes precedence.
    """
    configured = os.environ.get("CUPY_CACHE_DIR")
    cache_dir = (Path(configured).expanduser() if configured else
                 Path(tempfile.gettempdir()) / "terasort-cupy-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not configured:
        os.environ["CUPY_CACHE_DIR"] = str(cache_dir)
    return cache_dir


def configure_cupy_libraries():
    """Retain bundled Windows CUDA handles without importing Torch's runtime."""
    import ctypes
    import importlib.util
    import sys
    directory = None
    handles = []
    if sys.platform == "win32":
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.origin is not None:
            library = Path(spec.origin).parent / "lib"
            if (library / "cublas64_12.dll").is_file():
                directory = os.add_dll_directory(str(library))
                for name in ("cublasLt64_12.dll", "cublas64_12.dll",
                             "cusparse64_12.dll", "cusolver64_11.dll"):
                    path = library / name
                    if path.is_file():
                        handles.append(ctypes.WinDLL(str(path)))
    return directory, handles
