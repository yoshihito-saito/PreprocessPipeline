from importlib import import_module
from typing import TYPE_CHECKING

from .metafile import MergePointsData, PreprocessConfig, PreprocessResult, XmlMeta
from .behavior import BehaviorProcessingResult, discover_dlc_files, process_dlc_behavior
from .io import (
    derive_probe_assignments_from_xml,
    prepare_chanmap,
    select_basepath,
    select_paths_with_gui,
    show_chanmap,
)

if TYPE_CHECKING:
    from .pipeline import run_preprocess_session
    from .sorting_stage import run_sorting_stage
    from .state_scoring import StateScoreResult, run_state_scoring


_LAZY_EXPORTS = {
    "run_preprocess_session": ("src.preprocess.pipeline", "run_preprocess_session"),
    "run_sorting_stage": ("src.preprocess.sorting_stage", "run_sorting_stage"),
    "StateScoreResult": ("src.preprocess.state_scoring", "StateScoreResult"),
    "run_state_scoring": ("src.preprocess.state_scoring", "run_state_scoring"),
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
    "XmlMeta",
    "MergePointsData",
    "BehaviorProcessingResult",
    "discover_dlc_files",
    "process_dlc_behavior",
    "run_preprocess_session",
    "run_sorting_stage",
    "run_state_scoring",
    "StateScoreResult",
    "select_basepath",
    "select_paths_with_gui",
    "prepare_chanmap",
    "show_chanmap",
    "derive_probe_assignments_from_xml",
]
