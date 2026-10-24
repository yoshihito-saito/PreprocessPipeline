"""Validation of native KS4 settings used by full sorting and seed learning."""
from __future__ import annotations

import math


FILE_SETTINGS = {"filename", "data_dir", "results_dir", "probe_name", "probe_path"}
MANAGED_SEED_SETTINGS = {"n_chan_bin", "fs", "scale", "shift", "nt", "tmin", "tmax"} | FILE_SETTINGS


def validate_kilosort_settings(settings):
    """Check names and values against the pinned Kilosort parameter schema."""
    from kilosort.parameters import MAIN_PARAMETERS, EXTRA_PARAMETERS

    if not isinstance(settings, dict) or any(not isinstance(key, str) for key in settings):
        raise ValueError("KS4 settings must be a dictionary with string keys")
    specs = {**MAIN_PARAMETERS, **EXTRA_PARAMETERS}
    unknown = set(settings) - (set(specs) | FILE_SETTINGS)
    if unknown:
        raise ValueError(f"Unsupported KS4 settings: {sorted(unknown)}")
    for key, value in settings.items():
        if key in FILE_SETTINGS:
            continue
        spec = specs[key]
        if value is None and (spec["default"] is None or key == "max_channel_distance"):
            continue
        kind = spec["type"]
        if kind is float:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif kind is int:
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = isinstance(value, kind)
        if not valid:
            raise ValueError(f"KS4 {key} must be {kind.__name__}")
        if kind in (int, float):
            if (math.isnan(value) or value in spec["exclude"]
                    or (spec["min"] is not None and value < spec["min"])
                    or (spec["max"] is not None and value > spec["max"])):
                raise ValueError(f"KS4 {key} is outside its supported range")
        elif kind is list:
            if any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in value):
                raise ValueError(f"KS4 {key} must contain finite numbers")
    return dict(settings)


def validate_seed_settings(settings):
    """Leave source metadata, waveform length and no-drift policy to the caller."""
    settings = validate_kilosort_settings({} if settings is None else settings)
    managed = set(settings) & MANAGED_SEED_SETTINGS
    if managed:
        raise ValueError(f"Seed settings are supplied by the manifest/nt: {sorted(managed)}")
    if settings.get("nblocks", 0) != 0:
        raise ValueError("Bounded KS4 seed learning requires nblocks=0")
    settings.pop("nblocks", None)
    return settings
