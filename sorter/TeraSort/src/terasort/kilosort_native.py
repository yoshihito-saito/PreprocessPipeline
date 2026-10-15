"""Opt-in, source-guarded CUDA C++ reductions for Kilosort 4.1.7.

Kilosort remains installed unchanged. The adapter substitutes two exact
selection reductions in its GPL-3.0 template_match function at runtime.
CUDA compilation uses NVRTC through CuPy; tensor ownership stays with Torch.
"""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import weakref

import numpy as np

from .cuda_environment import configure_cupy_cache

EXPECTED_SOURCE_SHA256 = '8f47f33479d57f1ce7296964b3e3a81711edb0c1ebc525e912d23e58a0db0a54'


class NativeReductions:
    def __init__(self,neighbor_mode='basic',block_size=256,score_block_size=128):
        configure_cupy_cache()
        import cupy as cp
        self.cp = cp
        source = Path(__file__).with_name('kilosort_reductions.cu').read_text()
        self.module = cp.RawModule(code=source,options=('--std=c++11','--fmad=false'))
        self.neighbor_kernel = self.module.get_function('neighbor_max_f32')
        self.abs_kernel = self.module.get_function('abs_max_signed_f32')
        self.shared_neighbor_kernel = self.module.get_function('neighbor_max_shared_f32')
        self.warp_neighbor_kernel = self.module.get_function('neighbor_max_warp_f32')
        self.score_kernel = self.module.get_function('template_scores_shared_f32')
        if neighbor_mode not in ('basic','shared','warp') or block_size not in (64,128,256):
            raise ValueError('Invalid neighbor launch configuration')
        if score_block_size not in (64,128,256):
            raise ValueError('Invalid score block size')
        self.neighbor_mode,self.block_size,self.score_block_size=neighbor_mode,block_size,score_block_size
        self._validated = {}

    def _launch(self,kernel,tensor,arguments,count,grid=None,block=None):
        import torch
        # Launch on Torch's active stream. Raw pointer arguments avoid copies,
        # temporary CuPy arrays and transfers through CPU memory.
        with self.cp.cuda.Device(tensor.device.index):
            stream = self.cp.cuda.Stream.from_external(torch.cuda.current_stream(tensor.device))
            kernel(grid or ((count+255)//256,), block or (256,), arguments, stream=stream)

    def _check_indices(self,indices,limit,key):
        import torch
        try:
            version = indices._version
        except RuntimeError:
            version = None
        cached = self._validated.get(key)
        if version is None or cached is None or cached[0]() is not indices or cached[1:]!=(version,limit):
            lo,hi = torch.aminmax(indices)
            if int(lo)<0 or int(hi)>=limit:
                raise ValueError('Neighbor index is out of bounds')
            self._validated[key] = (weakref.ref(indices),version,limit)

    def neighbor_max(self,values,neighbors):
        import torch
        if (values.ndim!=2 or values.dtype!=torch.float32 or not values.is_cuda or
            neighbors.ndim!=2 or neighbors.dtype!=torch.int64 or neighbors.device!=values.device or
            neighbors.shape[1]!=values.shape[0] or not neighbors.shape[0] or not values.numel()):
            raise ValueError('Expected CUDA float32 [filter,time] and int64 [neighbor,filter] tensors')
        self._check_indices(neighbors,values.shape[0],'neighbors')
        output = torch.empty(values.shape,device=values.device,dtype=values.dtype)
        args = (np.uint64(values.data_ptr()),np.uint64(neighbors.data_ptr()),np.uint64(output.data_ptr()),
                *map(np.int64,[*values.shape,neighbors.shape[0],*values.stride(),*neighbors.stride()]))
        if self.neighbor_mode=='basic' or neighbors.shape[0]>256 or values.shape[0]>65535:
            self._launch(self.neighbor_kernel,values,args,values.numel())
        else:
            kernel = self.shared_neighbor_kernel if self.neighbor_mode=='shared' else self.warp_neighbor_kernel
            tile = self.block_size if self.neighbor_mode=='shared' else self.block_size//4
            self._launch(kernel,values,args,values.numel(),
                         grid=((values.shape[1]+tile-1)//tile,values.shape[0]),block=(self.block_size,))
        return output

    def template_scores(self,B,channels,weights,start,stop,arithmetic=1,diagnostic=False):
        """Experimental fused dot/reduction; accumulation can differ from cuBLAS.

        Shared-memory specialization applies only to the default 5x10 geometry;
        other valid Kilosort geometries fall back to the original Torch einsum.
        """
        import torch
        if (B.ndim!=3 or B.dtype!=torch.float32 or not B.is_cuda or
            channels.ndim!=2 or channels.dtype!=torch.int64 or channels.device!=B.device or
            weights.ndim!=3 or weights.dtype!=torch.float32 or weights.device!=B.device or
            weights.shape[1:]!=channels.shape or not 0<=start<stop<=B.shape[2] or
            not channels.numel() or not B.numel() or arithmetic not in (0,1)):
            raise ValueError('Invalid template-score tensors, time range or arithmetic mode')
        self._check_indices(channels,B.shape[0],'channels')
        if weights.shape[0]!=5 or channels.shape[0]!=10 or channels.shape[1]>65535:
            A=torch.einsum('ijk,jklm->iklm',weights,B[channels,:,start:stop])
            A=A.transpose(1,2).reshape(-1,channels.shape[1],stop-start)
            result=self.abs_max_signed(A)
            return (*result,A) if diagnostic else result
        shape=(channels.shape[1],stop-start)
        maxima=torch.empty(shape,device=B.device,dtype=B.dtype)
        indices=torch.empty(shape,device=B.device,dtype=torch.int64)
        raw=torch.empty((5*B.shape[1],*shape),device=B.device,dtype=B.dtype) if diagnostic else None
        args=(np.uint64(B.data_ptr()),np.uint64(channels.data_ptr()),np.uint64(weights.data_ptr()),
              np.uint64(maxima.data_ptr()),np.uint64(indices.data_ptr()),np.uint64(raw.data_ptr() if diagnostic else 0),
              *map(np.int64,[shape[0],B.shape[2],shape[1],start,B.shape[1],*B.stride(),*channels.stride(),*weights.stride()]),
              np.int32(arithmetic))
        block=self.score_block_size
        self._launch(self.score_kernel,B,args,maxima.numel(),grid=((shape[1]+block-1)//block,shape[0]),block=(block,))
        return (maxima,indices,raw) if diagnostic else (maxima,indices)

    def abs_max_signed(self,values):
        import torch
        if values.ndim!=3 or values.dtype!=torch.float32 or not values.is_cuda or not values.numel():
            raise ValueError('Expected a nonempty CUDA float32 [choice,filter,time] tensor')
        if values.shape[0]>=2**24:
            raise ValueError('Choice count exceeds exact float32 index representation in original Kilosort')
        maxima = torch.empty(values.shape[1:],device=values.device,dtype=values.dtype)
        indices = torch.empty(values.shape[1:],device=values.device,dtype=torch.int64)
        args = (np.uint64(values.data_ptr()),np.uint64(maxima.data_ptr()),np.uint64(indices.data_ptr()),
                *map(np.int64,[*values.shape,*values.stride()]))
        self._launch(self.abs_kernel,values,args,maxima.numel())
        return maxima,indices


def make_native_template_match(reductions,mode='both'):
    from kilosort import spikedetect
    if importlib.metadata.version('kilosort')!='4.1.7':
        raise RuntimeError('Native adapter validated only against Kilosort 4.1.7')
    source = inspect.getsource(spikedetect.template_match)
    if hashlib.sha256(source.encode()).hexdigest()!=EXPECTED_SOURCE_SHA256:
        raise RuntimeError('Kilosort template_match source changed; refusing an unverified substitution')
    if mode not in ('neighbor','both','shared','fused'):
        raise ValueError('Expected neighbor, both, shared or fused')
    source = source.replace('Amax = torch.max(Aa[iC2], 0)[0]',
                            'Amax = _native.neighbor_max(Aa, iC2)')
    if mode in ('both','shared','fused'):
        source = source.replace('Aa, imax = torch.max(A.abs(), 0)',
                                'Aa, imax = _native.abs_max_signed(A)')
        source = source.replace('        imax = (1+imax) * A[imax, ti.unsqueeze(-1), tj[:A.shape[-1]]].sign()',
                                '        # Native reduction already produced signed indices.')
    if mode in ('shared','fused'):
        reductions.neighbor_mode='shared'
    if mode=='fused':
        begin=source.index("        A = torch.einsum(")
        end=source.index('        # Native reduction already',begin)
        source=source[:begin]+('        Aa, imax = _native.template_scores(B, iC, weigh, nb*t, min(nb*(t+1), NT))\n')+source[end:]
    namespace = dict(vars(spikedetect),_native=reductions)
    exec(compile(source,'<Kilosort 4.1.7 with native reductions>','exec'),namespace)
    return namespace['template_match'],source


@contextmanager
def native_kilosort_reductions(mode='both'):
    from kilosort import spikedetect
    original = spikedetect.template_match
    reductions = NativeReductions()
    replacement,source = make_native_template_match(reductions,mode)
    spikedetect.template_match = replacement
    try:
        yield dict(mode=mode,original_source_sha256=EXPECTED_SOURCE_SHA256,
                   generated_source=source,implementation='CUDA C++ via NVRTC',
                   unchanged_template_arithmetic=mode!='fused')
    finally:
        spikedetect.template_match = original
