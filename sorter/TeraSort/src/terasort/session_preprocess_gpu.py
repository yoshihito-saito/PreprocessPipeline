"""Opt-in bounded CUDA execution of the session median/SOS signal frame."""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter

from .cuda_environment import configure_cupy_cache


class CudaPreprocessor:
    """Transfer INT16 once; return host voltage plus its consumable GPU copy.

    FP64 SOS coefficients preserve the CPU filter definition. CuPy's parallel
    recurrence has different rounding, so this backend is explicit/versioned.
    Temporary filter/reference matrices have a fixed byte budget; the voltage
    core itself is proportional to configured core size and channel count.
    """

    def __init__(self, probe, *, work_mb=16):
        if isinstance(work_mb, bool) or not isinstance(work_mb, int) or not 1 <= work_mb <= 64:
            raise ValueError("GPU preprocessing workspace must be 1-64 MiB")
        high = min(6000., .4*probe.sample_rate_hz)
        if high <= 300.:
            raise ValueError("Sample rate too low for the 300 Hz spike band")
        configure_cupy_cache()
        # Windows CUDA wheels bundled with Torch are not necessarily on the
        # DLL search path. Register that directory without importing Torch's
        # full runtime or modifying the user's PATH/CUDA installation.
        self._dll_directory = None
        self._cuda_libraries = []
        if sys.platform == "win32":
            spec = importlib.util.find_spec("torch")
            if spec is not None and spec.origin is not None:
                library = Path(spec.origin).parent / "lib"
                if (library / "cublas64_12.dll").is_file():
                    self._dll_directory = os.add_dll_directory(str(library))
                    for name in ("cublasLt64_12.dll", "cublas64_12.dll",
                                 "cusparse64_12.dll", "cusolver64_11.dll"):
                        path = library / name
                        if path.is_file():
                            self._cuda_libraries.append(ctypes.WinDLL(str(path)))
        import cupy as cp
        from cupyx.scipy.signal import sosfilt, sosfilt_zi
        self.cp, self.sosfilt, self.probe = cp, sosfilt, probe
        self.work_bytes = work_mb*1024**2
        self.sos = cp.asarray(butter(3, [300., high], btype="bandpass",
                                    fs=probe.sample_rate_hz, output="sos"))
        self.zi = sosfilt_zi(self.sos)[:, None, :]
        # Third-order bandpass has three sections and no origin poles/zeros.
        self.edge = 3*(2*len(self.sos)+1)
        self.groups = []
        for shank in np.unique(probe.shank):
            indices = np.flatnonzero(probe.shank == shank)
            good = np.setdiff1d(indices, probe.bad_channels, assume_unique=True)
            if len(good) >= 2:
                self.groups.append((cp.asarray(indices), cp.asarray(good)))

    def process(self, core):
        cp, probe = self.cp, self.probe
        if (core.raw.dtype != np.int16 or core.raw.ndim != 2 or
                not core.raw.flags.c_contiguous or
                core.raw.shape[1] != probe.n_channels or len(core.raw) < 64):
            raise ValueError("Expected a contiguous haloed INT16 core matching the probe")
        # These transfers deliberately block. CuPy's asynchronous NumPy
        # conversion can retain raw/voltage-sized pinned staging buffers in
        # addition to both real arrays. Network prefetch remains independent.
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        raw = cp.empty(core.raw.shape, dtype=cp.int16)
        cp.cuda.runtime.memcpy(raw.data.ptr, core.raw.ctypes.data, core.raw.nbytes,
                               cp.cuda.runtime.memcpyHostToDevice)
        voltage = cp.multiply(raw, np.float32(probe.gain_uv_per_count), dtype=cp.float32)
        del raw
        if getattr(probe, "input_is_preprocessed", False):
            host = np.empty(voltage.shape, dtype=np.float32)
            stream.synchronize()
            cp.cuda.runtime.memcpy(host.ctypes.data, voltage.data.ptr, host.nbytes,
                                   cp.cuda.runtime.memcpyDeviceToHost)
            return host, voltage
        rows = max(1, self.work_bytes//(4*probe.n_channels))
        for indices, good in self.groups:
            for first in range(0, len(voltage), rows):
                block = voltage[first:first+rows]
                reference = cp.median(block[:, good], axis=1)
                block[:, indices] -= reference[:, None]
        columns = max(1, self.work_bytes//((len(voltage)+2*self.edge)*8))
        for first in range(0, probe.n_channels, columns):
            span = slice(first, first+columns)
            # Channel-major layout keeps each independent recurrence contiguous.
            slab = cp.ascontiguousarray(voltage[:, span].T)
            edge = self.edge
            padded = cp.concatenate((2*slab[:, :1]-slab[:, edge:0:-1], slab,
                                     2*slab[:, -1:]-slab[:, -2:-edge-2:-1]), axis=1)
            forward, _ = self.sosfilt(self.sos, padded, axis=-1,
                                      zi=self.zi*padded[:, :1][None, :, :])
            backward, _ = self.sosfilt(self.sos, forward[:, ::-1], axis=-1,
                                       zi=self.zi*forward[:, -1:][None, :, :])
            voltage[:, span] = backward[:, ::-1][:, edge:-edge].T
            del slab, padded, forward, backward
        host = np.empty(voltage.shape, dtype=np.float32)
        stream.synchronize()
        cp.cuda.runtime.memcpy(host.ctypes.data, voltage.data.ptr, host.nbytes,
                               cp.cuda.runtime.memcpyDeviceToHost)
        return host, voltage

    def process_quantized(self, core, matcher):
        """Filter bounded channel slabs directly into an INT16 GPU residual.

        The host FP32 copy is still needed for QC and selective waveform
        caching. No persistent full FP32 voltage array is allocated on GPU;
        a slab can equal the full core for a small-channel probe.
        """
        cp, probe = self.cp, self.probe
        if (core.raw.dtype != np.int16 or core.raw.ndim != 2 or
                not core.raw.flags.c_contiguous or
                core.raw.shape[1] != probe.n_channels or len(core.raw) < 64):
            raise ValueError("Expected a contiguous haloed INT16 core matching the probe")
        if getattr(probe, "input_is_preprocessed", False):
            raise ValueError("Direct filter-to-INT16 is a raw-input option; use cuda-int16 for preprocessed input")
        from .session_int16 import CudaInt16ResidualMatcher, _QuantizedCore
        if not isinstance(matcher, CudaInt16ResidualMatcher) or matcher.cp is not cp:
            raise ValueError("Direct INT16 preprocessing requires the CUDA INT16 matcher")
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        raw = cp.empty(core.raw.shape, dtype=cp.int16)
        cp.cuda.runtime.memcpy(raw.data.ptr, core.raw.ctypes.data, core.raw.nbytes,
                               cp.cuda.runtime.memcpyHostToDevice)
        host = np.empty(raw.shape, dtype=np.float32)
        data = matcher.allocate_core(raw.shape)
        scales = np.empty(probe.n_channels, dtype=np.float32)
        # One reference trace per populated shank, including a zero trace for
        # singleton shanks. The reference matrix is bounded by the core size.
        references = cp.zeros((len(self.groups)+1, len(raw)), dtype=cp.float32)
        reference_ids = np.zeros(probe.n_channels, dtype=np.int32)
        rows = max(1, self.work_bytes//(4*probe.n_channels))
        for group_id, (indices, good) in enumerate(self.groups, start=1):
            reference_ids[cp.asnumpy(indices)] = group_id
            for first in range(0, len(raw), rows):
                block = cp.multiply(raw[first:first+rows, good],
                                    np.float32(probe.gain_uv_per_count),
                                    dtype=cp.float32)
                references[group_id, first:first+rows] = cp.median(block, axis=1)
        columns = max(1, self.work_bytes//((len(raw)+2*self.edge)*8))
        for first in range(0, probe.n_channels, columns):
            span = slice(first, min(first+columns, probe.n_channels))
            width = span.stop-span.start
            slab = cp.ascontiguousarray(cp.multiply(raw[:, span].T,
                np.float32(probe.gain_uv_per_count), dtype=cp.float32))
            slab -= references[cp.asarray(reference_ids[span])]
            edge = self.edge
            padded = cp.concatenate((2*slab[:, :1]-slab[:, edge:0:-1], slab,
                                     2*slab[:, -1:]-slab[:, -2:-edge-2:-1]), axis=1)
            forward, _ = self.sosfilt(self.sos, padded, axis=-1,
                                      zi=self.zi*padded[:, :1][None, :, :])
            backward, _ = self.sosfilt(self.sos, forward[:, ::-1], axis=-1,
                                       zi=self.zi*forward[:, -1:][None, :, :])
            # The SOS recurrence is FP64; the established path rounds into
            # its FP32 voltage core before quantization and host export.
            filtered = cp.ascontiguousarray(backward[:, ::-1][:, edge:-edge].T,
                                            dtype=cp.float32)
            scales[span] = matcher.quantize_slab(filtered, data[:, span])
            # Blocking 2-D copy writes each contiguous channel slab into the
            # original host row pitch without another full-size GPU array.
            stream.synchronize()
            cp.cuda.runtime.memcpy2D(
                host.ctypes.data + first*host.itemsize, host.strides[0],
                filtered.data.ptr, filtered.strides[0], width*host.itemsize,
                len(raw), cp.cuda.runtime.memcpyDeviceToHost)
            del slab, padded, forward, backward, filtered
        return host, _QuantizedCore(data, cp.asarray(scales))

    def host(self, core, probe):
        if probe is not self.probe:
            raise ValueError("GPU preprocessor belongs to another probe")
        voltage, device = self.process(core)
        del device
        return voltage
