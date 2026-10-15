"""Experimental scaled INT16 residual/template storage with FP32 fitting.

Quantization is per core/contact and per template, after floating-point
preprocessing. Fourfold residual headroom allows subtraction; exhausted
headroom raises before a shard is committed instead of wrapping or clipping.
"""
from dataclasses import dataclass

import numpy as np

from .session_gpu import CudaResidualMatcher


def _scales(peaks, *, headroom):
    peaks = np.asarray(peaks, dtype=np.float32)
    if not np.all(np.isfinite(peaks)):
        raise ValueError("INT16 storage requires finite voltage and templates")
    # Power-of-two scales keep dequantization exact in FP32. The minimum
    # quantum is 1/256 microvolt, rather than one raw ADC count.
    needed = np.maximum(peaks.astype(np.float64) * headroom / 32760., 1. / 256.)
    return np.exp2(np.ceil(np.log2(needed))).astype(np.float32)


@dataclass(frozen=True)
class _QuantizedCore:
    data: object
    scales: object


class CudaInt16ResidualMatcher(CudaResidualMatcher):
    """INT16 global arrays, FP32 register/shared-memory arithmetic and scores."""

    _kernel_prefix = "#define SESSION_INT16 1\n"
    _sample_dtype_name = "int16"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cp = self.cp
        self.residual_scales = None
        self.template_scales = cp.ones(1, cp.float32)
        self.overflow = cp.zeros(1, cp.int32)
        self._quantize = cp.ElementwiseKernel(
            'float32 value, raw float32 scales, int32 channels', 'int16 result',
            'result = (short)__float2int_rn(value / scales[i % channels]);',
            'session_quantize_int16_v1')

    def prepare_signal(self, voltage_uv):
        """Quantize one core, so callers can release its FP32 device buffer."""
        cp = self.cp
        source = cp.asarray(voltage_uv, dtype=cp.float32, order='C')
        if source.ndim != 2 or not source.size:
            raise ValueError("INT16 storage requires a nonempty time/channel core")
        # Two reductions avoid allocating abs(source), a full FP32 core.
        peaks = cp.maximum(cp.abs(cp.min(source, axis=0)),
                           cp.abs(cp.max(source, axis=0)))
        scales = cp.asarray(_scales(cp.asnumpy(peaks), headroom=4.))
        # CAS subtraction accesses a 32-bit word even for the final short.
        # Keep padding inside this allocation, including odd channel counts.
        storage = cp.empty(source.size + (source.size & 1), dtype=cp.int16)
        storage[-1] = 0
        data = storage[:source.size].reshape(source.shape)
        self._quantize(source, scales, np.int32(source.shape[1]), data)
        return _QuantizedCore(data, scales)

    def allocate_core(self, shape):
        """Allocate a padded residual for slab-wise preprocessing output."""
        cp = self.cp
        count = int(np.prod(shape))
        if len(shape) != 2 or count <= 0:
            raise ValueError("INT16 storage requires a nonempty time/channel core")
        storage = cp.empty(count + (count & 1), dtype=cp.int16)
        if count & 1:
            storage[-1] = 0
        return storage[:count].reshape(shape)

    def quantize_slab(self, source, destination):
        """Quantize one FP32 channel slab into its final INT16 residual."""
        cp = self.cp
        if (source.ndim != 2 or source.shape != destination.shape or
                source.dtype != cp.float32 or destination.dtype != cp.int16):
            raise ValueError("Expected matching FP32/INT16 time-channel slabs")
        peaks = cp.maximum(cp.abs(cp.min(source, axis=0)),
                           cp.abs(cp.max(source, axis=0)))
        scales = _scales(cp.asnumpy(peaks), headroom=4.)
        self._quantize(source, cp.asarray(scales), np.int32(source.shape[1]), destination)
        return scales

    def _prepare_residual(self, voltage_uv):
        core = (voltage_uv if isinstance(voltage_uv, _QuantizedCore)
                else self.prepare_signal(voltage_uv))
        self.residual_scales = core.scales
        self.overflow.fill(0)
        return core.data

    def _upload_templates(self):
        waveforms = self.models.waveforms
        peaks = (np.max(np.abs(waveforms), axis=(1, 2))
                 if len(waveforms) else np.empty(0, np.float32))
        scales = _scales(peaks, headroom=1.)
        self.template_scales = self.cp.asarray(scales)
        quantized = np.rint(waveforms / scales[:, None, None]).astype(np.int16)
        return self.cp.asarray(quantized)

    def _channel_values(self, residual, rows, channels=slice(None)):
        return (residual[rows, channels].astype(self.cp.float32)
                * self.residual_scales[channels])

    def _extra_kernel_args(self):
        if self.residual_scales is None:
            raise ValueError("Prepare an INT16 core before detecting or fitting")
        return self.residual_scales, self.template_scales, self.overflow

    def _check_subtraction(self):
        if int(self.overflow.get()[0]):
            raise OverflowError(
                "INT16 residual headroom exhausted; the incomplete shard was not "
                "committed. Use float32 matching storage in a new output directory.")
