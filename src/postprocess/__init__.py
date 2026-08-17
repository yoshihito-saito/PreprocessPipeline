from importlib import import_module
from typing import TYPE_CHECKING

from .metafile import PostprocessConfig, PostprocessResult

if TYPE_CHECKING:
    from .pipeline import (
        attach_existing_sorting_result,
        build_preprocessed_recording_from_result,
        make_post_recording,
        run_postprocess_session,
        use_existing_sorting,
    )


_LAZY_EXPORTS = {
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
    "PostprocessConfig",
    "PostprocessResult",
    "run_postprocess_session",
    "attach_existing_sorting_result",
    "build_preprocessed_recording_from_result",
    "use_existing_sorting",
    "make_post_recording",
]
