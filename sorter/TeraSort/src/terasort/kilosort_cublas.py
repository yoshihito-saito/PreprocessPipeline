"""Experimental C++/cuBLAS projection and shared-memory temporal convolution.

Torch retains tensor ownership. A platform-native bridge calls cuBLAS directly
with a separate handle, without modifying Torch's handle or precision flags.
These paths may change reduction order; accuracy must be evaluated.
"""
import ctypes
import os
import sys
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import textwrap
import weakref

import numpy as np
from .kilosort_native import NativeReductions


def native_bridge_path():
    if sys.platform == 'win32':
        name = 'cublas_bridge.dll'
    elif sys.platform.startswith('linux'):
        name = 'cublas_bridge.so'
    else:
        return None
    return Path(__file__).with_name('_native') / name


def cublas_library_path(torch):
    torch_root = Path(torch.__file__).resolve().parent
    if sys.platform == 'win32':
        candidates = (torch_root / 'lib/cublas64_12.dll',)
    elif sys.platform.startswith('linux'):
        candidates = (torch_root / 'lib/libcublas.so.12',
                      torch_root.parent / 'nvidia/cublas/lib/libcublas.so.12',
                      Path(sys.prefix) / 'lib/libcublas.so.12')
    else:
        return None
    return next((path for path in candidates if path.is_file()), None)


def bridge_path():
    native = native_bridge_path()
    if native is None or not native.is_file():
        raise RuntimeError('TeraSort native bridge is unavailable; build it with '
                           'scripts/build_cublas_bridge.py on Windows or '
                           'scripts/build_cublas_bridge_linux.py on Linux')
    return native


class CublasProjection(NativeReductions):
    def __init__(self, device=None):
        import torch
        super().__init__(neighbor_mode='shared')
        self.device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        native_cublas = cublas_library_path(torch)
        if native_cublas is None:
            raise RuntimeError('CUDA 12 cuBLAS library was not found in the active PyTorch environment')
        self.library = ctypes.CDLL(str(bridge_path()))
        path_type = ctypes.c_wchar_p if sys.platform == 'win32' else ctypes.c_char_p
        self.library.pp_cublas_create.argtypes = [path_type, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]
        self.library.pp_cublas_create.restype = ctypes.c_int
        self.library.pp_cublas_sgemm.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*3
        self.library.pp_cublas_sgemm.restype = ctypes.c_int
        self.library.pp_cublas_destroy.argtypes = [ctypes.c_void_p]
        self.library.pp_cublas_destroy.restype = ctypes.c_int
        self.handle = ctypes.c_void_p()
        version = ctypes.c_int()
        native_path = str(native_cublas) if sys.platform == 'win32' else os.fsencode(native_cublas)
        with torch.cuda.device(self.device):
            self._check(self.library.pp_cublas_create(native_path, ctypes.byref(self.handle), ctypes.byref(version)))
        self.cublas_version = version.value
        self._packed = None
        self.pack_builds = 0
        self.module_filter = self.cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text(),
                                               options=('--std=c++11', '--fmad=false'))
        self.filter_kernel = self.module_filter.get_function('temporal_6x61_f32')
        self.filter_kernel_41 = self.module_filter.get_function('temporal_6x41_f32')

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f'Native cuBLAS bridge status {status}')

    def close(self):
        import torch
        if self.handle:
            with torch.cuda.device(self.device):
                self._check(self.library.pp_cublas_destroy(self.handle))
            self.handle = ctypes.c_void_p()

    def pack_weights(self, U):
        """Cache the small [unit,channel,PCA] bank, not a reordered signal batch."""
        try:
            version = U._version
        except RuntimeError:
            version = None
        cached = self._packed
        if version is None or cached is None or cached[0]() is not U or cached[1] != version:
            packed = U.transpose(1, 2).contiguous().reshape(U.shape[0], -1)
            self._packed = weakref.ref(U), version, packed
            self.pack_builds += 1
        return self._packed[2]

    def gemm(self, weights, signal):
        import torch
        if (not self.handle or weights.ndim != 2 or signal.ndim != 2 or
            weights.shape[1] != signal.shape[0] or
            any(x.device != self.device or x.dtype != torch.float32 or not x.is_contiguous()
                or not x.numel() for x in (weights, signal)) or
            max(*weights.shape, *signal.shape) > 2**31-1):
            raise ValueError('Expected contiguous CUDA FP32 [unit,feature] and [feature,time] on the handle device')
        output = torch.empty((weights.shape[0], signal.shape[1]), device=self.device, dtype=torch.float32)
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            self._check(self.library.pp_cublas_sgemm(self.handle, stream.cuda_stream,
                        weights.data_ptr(), signal.data_ptr(), output.data_ptr(),
                        weights.shape[0], weights.shape[1], signal.shape[1]))
            for tensor in (weights, signal, output):
                tensor.record_stream(stream)
        return output

    def project(self, U, B, layout='channel'):
        if U.ndim != 3 or B.ndim != 3 or U.shape[1:] != (B.shape[1], B.shape[0]):
            raise ValueError('Expected U[unit,PCA,channel] and B[channel,PCA,time]')
        if layout == 'channel':
            weights, signal = self.pack_weights(U), B.reshape(-1, B.shape[-1])
        elif layout == 'pca':
            weights, signal = U.contiguous().reshape(U.shape[0], -1), B.transpose(0, 1).reshape(-1, B.shape[-1])
        else:
            raise ValueError('Unknown projection layout')
        return self.gemm(weights, signal)

    def convolve(self, X, W, layout='channel'):
        import torch
        if (X.ndim != 2 or not X.is_contiguous() or X.device != self.device or
            X.dtype != torch.float32 or not X.numel() or W.shape not in ((6, 41), (6, 61)) or
            W.device != X.device or W.dtype != X.dtype or not W.is_contiguous() or
            layout not in ('channel', 'pca') or X.shape[0] > 65535 or X.shape[1] > 2**31-1):
            raise ValueError('Expected contiguous CUDA FP32 X[channel,time], W[6,41 or 61] and a supported layout')
        C, T = X.shape
        output = torch.empty((6, C, T) if layout == 'pca' else (C, 6, T), device=X.device, dtype=X.dtype)
        args = tuple(np.uint64(x.data_ptr()) for x in (X, W, output)) + (np.int32(C), np.int32(T), np.int32(layout == 'pca'))
        kernel = self.filter_kernel_41 if W.shape[1] == 41 else self.filter_kernel
        self._launch(kernel, X, args, output.numel(), grid=((T+255)//256, C), block=(256,))
        return output.transpose(0, 1) if layout == 'pca' else output


def make_cublas_learned(deep, projection, mode='filter_pca'):
    from .kilosort_deep import make_deep_matching
    original, source = make_deep_matching(deep, 'all')
    if mode not in ('projection', 'filter_pca', 'filter_channel'):
        raise ValueError('Unknown native learned-matching mode')
    if mode.startswith('filter_'):
        layout = mode.removeprefix('filter_')
        old = 'B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)'
        assert source.count(old) == 1
        source = source.replace(old, f"B = _projection.convolve(X, W, '{layout}')")
    else:
        layout = 'channel'
    old = "B = torch.einsum('ijk, kjl -> il', U, B)"
    assert source.count(old) == 1
    source = source.replace(old, f"B = _projection.project(U, B, '{layout}')")
    namespace = dict(original.__globals__, _projection=projection)
    exec(compile(source, '<Kilosort 4.1.7 native cuBLAS projection>', 'exec'), namespace)
    return namespace['run_matching'], source


def make_cublas_universal(projection, tile_samples=2048):
    from .kilosort_tiled import make_tiled_template_match
    original = make_tiled_template_match(projection, tile_samples)
    source = textwrap.dedent(inspect.getsource(original))
    old = "B = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=nt // 2)"
    if source.count(old) != 1:
        raise RuntimeError('Tiled detector changed; refusing unverified filter replacement')
    source = source.replace(old, "B = _projection.convolve(X, ops['wTEMP'].contiguous(), 'channel')")
    namespace = dict(original.__globals__, **inspect.getclosurevars(original).nonlocals,
                     _projection=projection)
    exec(compile(source, '<Kilosort tiled detector with native temporal filter>', 'exec'), namespace)
    return namespace['template_match'], source


@contextmanager
def cublas_kilosort(device=None):
    """Six-filter 41/61-tap specialization with tiled scores and deep CUDA."""
    from kilosort import spikedetect, template_matching
    from .kilosort_deep import DeepCuda
    projection = CublasProjection(device=device)
    try:
        deep = DeepCuda(parallel=True)
        universal, universal_source = make_cublas_universal(projection, 2048)
        learned, learned_source = make_cublas_learned(deep, projection, 'filter_pca')
        originals = spikedetect.template_match, template_matching.run_matching
        spikedetect.template_match, template_matching.run_matching = universal, learned
        try:
            yield dict(mode='cublas', generated_source=universal_source,
                       learned_generated_source=learned_source, universal_tile_samples=2048,
                       runtime_stats=lambda:dict(overlap_fallback_calls=deep.fallback_count,
                           waveform_bank_builds=deep.waveform_builds, cublas_version=projection.cublas_version,
                           weight_pack_builds=projection.pack_builds, learned_mode='native_filter_pca_cublas',
                            **{('bridge_dll_sha256' if sys.platform == 'win32' else 'bridge_so_sha256'):
                               hashlib.sha256(bridge_path().read_bytes()).hexdigest()}))
        finally:
            spikedetect.template_match, template_matching.run_matching = originals
    finally:
        projection.close()
