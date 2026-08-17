import os
from importlib import import_module
from typing import TYPE_CHECKING

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if TYPE_CHECKING:
    from .postprocess import (
        PostprocessConfig,
        PostprocessResult,
        attach_existing_sorting_result,
        build_preprocessed_recording_from_result,
        make_post_recording,
        run_postprocess_session,
        use_existing_sorting,
    )
    from .preprocess import PreprocessConfig, PreprocessResult, run_preprocess_session


_LAZY_EXPORTS = {
    "PreprocessConfig": ("src.preprocess.metafile", "PreprocessConfig"),
    "PreprocessResult": ("src.preprocess.metafile", "PreprocessResult"),
    "run_preprocess_session": ("src.preprocess.pipeline", "run_preprocess_session"),
    "PostprocessConfig": ("src.postprocess.metafile", "PostprocessConfig"),
    "PostprocessResult": ("src.postprocess.metafile", "PostprocessResult"),
    "run_postprocess_session": ("src.postprocess.pipeline", "run_postprocess_session"),
    "attach_existing_sorting_result": (
        "src.postprocess.pipeline",
        "attach_existing_sorting_result",
    ),
    "build_preprocessed_recording_from_result": (
        "src.postprocess.pipeline",
        "build_preprocessed_recording_from_result",
    ),
    "use_existing_sorting": ("src.postprocess.pipeline", "use_existing_sorting"),
    "make_post_recording": ("src.postprocess.pipeline", "make_post_recording"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "run_preprocess_session",
    "PostprocessConfig",
    "PostprocessResult",
    "run_postprocess_session",
    "attach_existing_sorting_result",
    "build_preprocessed_recording_from_result",
    "use_existing_sorting",
    "make_post_recording",
]
