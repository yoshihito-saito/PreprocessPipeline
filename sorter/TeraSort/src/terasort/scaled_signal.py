"""INT16 host voltage with dequantization restricted to requested samples."""
import numpy as np


class ScaledInt16Signal:
    def __init__(self, data, scales):
        self.data = np.asarray(data)
        self.scales = np.asarray(scales, np.float32)
        if (self.data.dtype != np.int16 or self.data.ndim != 2 or
                self.scales.shape != (self.data.shape[1],) or
                not np.all(np.isfinite(self.scales) & (self.scales > 0))):
            raise ValueError("Expected INT16 time/channel data and positive channel scales")
        self.shape, self.ndim = self.data.shape, 2

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Request selected voltage samples; full-core conversion must be explicit")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Expected time and channel indices")
            return self.data[key].astype(np.float32) * self.scales[key[1]]
        if isinstance(key, slice):
            return ScaledInt16Signal(self.data[key], self.scales)
        return self.data[key].astype(np.float32) * self.scales

    def mask_channels(self, usable):
        self.data[:, ~np.asarray(usable, bool)] = 0

    def to_float32(self):
        """Explicit bounded calibration/debug conversion; not the dense path."""
        result = self.data.astype(np.float32)
        result *= self.scales
        return result
