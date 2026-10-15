"""Opt-in KS4 input frame for voltage already filtered and referenced upstream."""
from contextlib import contextmanager


@contextmanager
def preprocessed_seed_input(enabled):
    """Keep KS4 whitening/learning while bypassing a second high-pass filter."""
    if not enabled:
        yield
        return
    import kilosort
    import torch
    from kilosort import preprocessing

    if kilosort.__version__ != "4.1.7":
        raise RuntimeError("Preprocessed KS4 input requires the pinned Kilosort 4.1.7")
    original = preprocessing.get_highpass_filter

    def identity_filter(fs=30000, cutoff=300, device=None):
        # Match KS4's centered impulse and FFT-shift convention.
        impulse = torch.zeros(30122, dtype=torch.float32, device=device)
        impulse[30122 // 2] = 1
        return impulse

    preprocessing.get_highpass_filter = identity_filter
    try:
        yield
    finally:
        preprocessing.get_highpass_filter = original
