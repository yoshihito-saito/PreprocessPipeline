"""Experimental integer SOS filtering, GPU QC, and selective host voltage.

This versioned frame is opt-in. Bulk filtered arrays stay INT16. Sparse
calibration may explicitly dequantize a core; dense sorting never needs to.
"""
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .cuda_environment import configure_cupy_cache
from .scaled_signal import ScaledInt16Signal
from .session_int16 import _QuantizedCore
from .session_signal import (Quality, QC_SATURATED, QC_FLATLINE, QC_DROPOUT,
                             QC_NOISY, QC_USER_BAD, QC_ARTIFACT)


FRAME = "cuda_int16_sos_q28q12_bias_retry_tile1024_halo20ms_qc_v2"


class FixedInt16Preprocessor:
    def __init__(self, probe, *, reference_mb=8):
        if (isinstance(reference_mb, bool) or not isinstance(reference_mb, int)
                or not 1 <= reference_mb <= 64):
            raise ValueError("Integer reference workspace must be 1-64 MiB")
        high = min(6000., .4*probe.sample_rate_hz)
        if high <= 300.:
            raise ValueError("Sample rate too low for the spike band")
        configure_cupy_cache()
        import cupy as cp
        self.cp, self.probe = cp, probe
        sos = butter(3, [300., high], btype="bandpass",
                     fs=probe.sample_rate_hz, output="sos")
        coefficients = np.rint(sos[:, [0, 1, 2, 4, 5]] * 2**28)
        zi = np.rint(sosfilt_zi(sos)*2**28)
        for values in (coefficients, zi):
            if not np.isfinite(values).all() or np.max(np.abs(values)) > 2147483647:
                raise ValueError("Fixed-point filter coefficients exceed INT32 range")
        for row in coefficients:
            if np.max(np.abs(np.roots([1., row[3]/2**28, row[4]/2**28]))) >= 1:
                raise ValueError("Quantized SOS poles are unstable")
        self.coefficients = cp.asarray(coefficients, dtype=cp.int32, order="C")
        self.zi = cp.asarray(zi, dtype=cp.int32, order="C")
        impulse = np.zeros(16384); impulse[0] = 1.
        self.bound = float(np.sum(np.abs(sosfilt(sos, impulse)))**2 * 1.01)
        self.groups = []
        ids = np.zeros(probe.n_channels, np.int32)
        for shank in np.unique(probe.shank):
            indices = np.flatnonzero(probe.shank == shank)
            good = np.setdiff1d(indices, probe.bad_channels)
            if len(good) >= 2:
                self.groups.append(cp.asarray(good))
                ids[indices] = len(self.groups)
        self.ids = cp.asarray(ids)
        self.reference_bytes = reference_mb*1024**2
        self.tile = 1024
        self.halo = max(32, int(np.ceil(probe.sample_rate_hz*.02)))
        module = cp.RawModule(code=Path(__file__).with_suffix(".cu").read_text(),
                              options=("--std=c++11", "--fmad=false"))
        self.peak_kernel = module.get_function("peaks_and_qc")
        self.filter_kernel = module.get_function("tiled_sos")
        self.headroom_kernel = module.get_function("reserve_headroom")
        self.artifact_kernel = module.get_function("synchronous_artifact")
        self.last_diagnostics = {}

    def _preprocessed_core(self, core, begin, finish):
        cp = self.cp
        raw = core.raw
        nt, nc = raw.shape
        storage = cp.empty(raw.size + (raw.size & 1), cp.int16)
        storage[-1] = 0
        data = storage[:raw.size].reshape(raw.shape)
        cp.cuda.runtime.memcpy(data.data.ptr, raw.ctypes.data, raw.nbytes,
                               cp.cuda.runtime.memcpyHostToDevice)
        # QC counts describe the supplied ADC samples before headroom rounding.
        counts = cp.zeros((3, nc), cp.int32)
        rows = max(1, self.reference_bytes // max(nc * 4, 1))
        for first in range(begin, finish, rows):
            stop = min(first + rows, finish)
            block = data[first:stop]
            counts[0] += cp.count_nonzero((block >= 32760) | (block <= -32760), axis=0).astype(cp.int32)
            counts[1] += cp.count_nonzero(block == 0, axis=0).astype(cp.int32)
            lo = max(begin + 1, first)
            counts[2] += cp.count_nonzero(data[lo:stop] == data[lo-1:stop-1], axis=0).astype(cp.int32)
        extrema = cp.maximum(cp.abs(cp.min(data, axis=0).astype(cp.int32)),
                             cp.abs(cp.max(data, axis=0).astype(cp.int32)))
        shifts = np.ceil(np.log2(np.maximum(cp.asnumpy(extrema) / 8190., 1.))).astype(np.int32)
        self.headroom_kernel(((raw.size+255)//256,), (256,),
                             (data, cp.asarray(shifts), np.int64(raw.size), np.int32(nc)))
        scales = (np.exp2(shifts.astype(np.float64)) * self.probe.gain_uv_per_count).astype(np.float32)
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise OverflowError("Preprocessed input voltage scale is not representable")
        self.last_diagnostics = {"retry_passes": 0, "retried_channels": 0}
        return _QuantizedCore(data, cp.asarray(scales)), scales, cp.asnumpy(counts)

    def filter(self, core):
        cp = self.cp
        raw = core.raw
        if (raw.dtype != np.int16 or raw.ndim != 2 or not raw.flags.c_contiguous
                or raw.shape[1] != self.probe.n_channels or len(raw) < 64
                or max(raw.shape) > 2147483647):
            raise ValueError("Expected a bounded contiguous INT16 core matching the probe")
        nt, nc = raw.shape
        begin, finish = core.core_start-core.data_start, core.core_stop-core.data_start
        if not 0 <= begin < finish <= nt:
            raise ValueError("Core interval is outside its haloed data")
        if getattr(self.probe, "input_is_preprocessed", False):
            return self._preprocessed_core(core, begin, finish)
        stream = cp.cuda.get_current_stream()
        data = cp.empty(raw.shape, cp.int16)
        cp.cuda.runtime.memcpyAsync(data.data.ptr, raw.ctypes.data, raw.nbytes,
            cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
        stream.synchronize()
        refs = cp.zeros((len(self.groups)+1, nt), cp.int32)
        for group_id, good in enumerate(self.groups, 1):
            rows = max(1, self.reference_bytes//(len(good)*2))
            for first in range(0, nt, rows):
                block = cp.sort(data[first:first+rows, good], axis=1)
                refs[group_id, first:first+rows] = (
                    block[:, len(good)//2].astype(cp.int32)
                    + block[:, (len(good)-1)//2].astype(cp.int32))
            del block
        # Bandpass rejects constants. Remove an integer per-contact center
        # before selecting scale so large DC offsets do not consume precision.
        stride = max(1, nt//4000)
        sample = (2*data[::stride].astype(cp.int32) - refs[self.ids, ::stride].T)
        ordered = cp.sort(cp.ascontiguousarray(sample.T), axis=1)
        n = ordered.shape[1]
        bias = (ordered[:, n//2] + ordered[:, (n-1)//2])//2
        del sample, ordered
        peaks = cp.zeros(nc, cp.int32)
        counts = cp.zeros((3, nc), cp.int32)
        self.peak_kernel(((nc+31)//32, (nt+255)//256), (32, 8),
            (data, refs, self.ids, bias, peaks, counts, np.int32(nt), np.int32(nc),
             np.int32(begin), np.int32(finish)))
        peak_host = cp.asnumpy(peaks)
        powers = np.maximum(-8, np.ceil(np.log2(np.maximum(
            peak_host.astype(np.float64)*.5*self.bound/8190., 2.**-8)))-3).astype(np.int32)
        scratch = cp.empty(((nt+self.tile-1)//self.tile,
                            self.tile+2*self.halo, nc), cp.int16)
        storage = cp.empty(raw.size+(raw.size & 1), cp.int16)
        storage[-1] = 0  # Padding makes the last short safe for matcher CAS.
        result = storage[:raw.size].reshape(raw.shape)
        active = np.ones(nc, np.uint8)
        retried = np.zeros(nc, bool)
        for attempt in range(7):
            if np.any(powers > 18):
                raise OverflowError("Integer filter scale exceeded its checked range")
            failure = cp.zeros(nc, cp.int32)
            self.filter_kernel(((nc+31)//32, (nt+self.tile-1)//self.tile), (32,),
                (data, result, scratch, refs, self.ids, bias, cp.asarray(powers),
                 self.coefficients, self.zi, cp.asarray(active), failure,
                 np.int32(nt), np.int32(nc), np.int32(self.tile), np.int32(self.halo)))
            flags = cp.asnumpy(failure)
            if np.any(flags & 1):
                raise OverflowError("Integer filter state overflow; incomplete core discarded")
            failed = flags != 0
            if not np.any(failed):
                break
            retried |= failed
            powers[failed] += 1
            active = failed.astype(np.uint8)
        else:
            raise OverflowError("Integer filter retry limit exhausted; incomplete core discarded")
        # Reserve headroom using the measured output, not a raw-input bound.
        extrema = cp.maximum(cp.abs(cp.min(result, axis=0).astype(cp.int32)),
                             cp.abs(cp.max(result, axis=0).astype(cp.int32)))
        shifts = np.ceil(np.log2(np.maximum(cp.asnumpy(extrema)/8190., 1.))).astype(np.int32)
        self.headroom_kernel(((raw.size+255)//256,), (256,),
            (result, cp.asarray(shifts), np.int64(raw.size), np.int32(nc)))
        scales = (np.exp2((powers+shifts).astype(np.float64))
                  * self.probe.gain_uv_per_count).astype(np.float32)
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise OverflowError("Integer filter voltage scale is not representable")
        self.last_diagnostics = {"retry_passes": attempt,
                                 "retried_channels": int(retried.sum())}
        return _QuantizedCore(result, cp.asarray(scales)), scales, cp.asnumpy(counts)

    def quality(self, core, quantized, scales, counts):
        cp, probe = self.cp, self.probe
        begin, finish = core.core_start-core.data_start, core.core_stop-core.data_start
        length = finish-begin
        sample = cp.ascontiguousarray(quantized.data[begin:finish:max(1, length//4000)].T)
        ordered = cp.sort(sample, axis=1)
        n = sample.shape[1]
        center2 = (ordered[:, n//2].astype(cp.int32)
                   + ordered[:, (n-1)//2].astype(cp.int32))
        deviations = cp.sort(cp.abs(2*sample.astype(cp.int32)-center2[:, None]), axis=1)
        mad = (deviations[:, n//2] + deviations[:, (n-1)//2]).astype(cp.float32)*np.float32(.25)
        noise = cp.asnumpy(mad*quantized.scales/np.float32(.67448975))
        noise = np.maximum(noise, np.float32(.01))
        saturated, zero = counts[0]/length, counts[1]/length
        flat = counts[2]/(length-1) if length > 1 else np.ones(probe.n_channels)
        flags = np.zeros(probe.n_channels, np.uint8)
        flags[saturated > 1e-4] |= QC_SATURATED
        flags[flat > .98] |= QC_FLATLINE
        flags[zero > .99] |= QC_DROPOUT
        for shank in np.unique(probe.shank):
            members = np.flatnonzero(probe.shank == shank)
            baseline = float(np.median(noise[members]))
            flags[members[noise[members] > max(5*baseline, 100.)]] |= QC_NOISY
        if probe.bad_channels:
            flags[list(probe.bad_channels)] |= QC_USER_BAD
        usable = flags == 0
        synchronous = False
        if np.sum(usable) >= 8:
            artifact = cp.zeros(1, cp.int32)
            self.artifact_kernel(((length+7)//8,), (256,),
                (quantized.data, quantized.scales, cp.asarray(noise),
                 cp.asarray(usable, dtype=cp.uint8), artifact, np.int32(probe.n_channels),
                 np.int32(begin), np.int32(finish), np.int32(usable.sum())))
            synchronous = bool(artifact.get()[0])
        bad = bool(synchronous or np.mean(~usable) > .2)
        if synchronous:
            flags |= QC_ARTIFACT
        return Quality(noise, flags, bad, saturated, flat, zero)

    def process(self, core):
        quantized, scales, counts = self.filter(core)
        quality = self.quality(core, quantized, scales, counts)
        host = np.empty(core.raw.shape, np.int16)
        cp = self.cp
        stream = cp.cuda.get_current_stream()
        cp.cuda.runtime.memcpyAsync(host.ctypes.data, quantized.data.data.ptr, host.nbytes,
            cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        stream.synchronize()
        return ScaledInt16Signal(host, scales), quantized, quality

    def host(self, core, probe):
        if probe is not self.probe:
            raise ValueError("Integer preprocessor belongs to another probe")
        signal, _, _ = self.process(core)
        return signal.to_float32()
