"""Kilosort 4.1.7-compatible sorting entry point with optional CUDA acceleration."""

from .api import run_kilosort

__version__ = "0.5.0"

__all__ = ["DEFAULT_SETTINGS", "run_kilosort"]


def __getattr__(name):
    if name == "DEFAULT_SETTINGS":
        from kilosort import DEFAULT_SETTINGS
        return DEFAULT_SETTINGS
    raise AttributeError(name)
