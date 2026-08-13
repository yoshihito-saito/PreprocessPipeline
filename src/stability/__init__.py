"""Standalone recording-stability analyses.

These helpers are intentionally not wired into the preprocess pipeline or GUI.
"""

from .channel_mad import (
    MultiDayMadResult,
    analyze_multi_day_mad,
    compute_mad_noise,
    compute_multi_day_mad,
    evenly_spaced_windows,
    plot_multi_day_mad,
    plot_multi_day_mad_by_channel,
)
from .mapping_check import (
    MultiDayMappingCheckResult,
    analyze_multi_day_mapping_check,
    compute_multi_day_mapping_check,
    plot_multi_day_mapping_check,
)

__all__ = [
    "MultiDayMadResult",
    "MultiDayMappingCheckResult",
    "analyze_multi_day_mad",
    "analyze_multi_day_mapping_check",
    "compute_mad_noise",
    "compute_multi_day_mad",
    "compute_multi_day_mapping_check",
    "evenly_spaced_windows",
    "plot_multi_day_mad",
    "plot_multi_day_mad_by_channel",
    "plot_multi_day_mapping_check",
]
