"""Second-stage CUDA fusion for learned-template matching in Kilosort 4.1.7."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import weakref

import numpy as np
from .kilosort_native import NativeReductions,native_kilosort_reductions

# Filled from the pinned installed implementation and checked before patching.
EXPECTED_LEARNED_SOURCE_SHA256 = 'b6adcafea581d8e80859695307a12c8c9e18f6c35edbc8530da2c502ad5c5f16'


class DeepCuda(NativeReductions):
    def __init__(self,parallel=False):
        super().__init__()
        source=Path(__file__).with_name('kilosort_deep.cu').read_text()
        self.deep_module=self.cp.RawModule(code=source,options=('--std=c++11','--fmad=false'))
        self.max_kernel=self.deep_module.get_function('learned_max_f32')
        self.parallel_max_kernel=self.deep_module.get_function('learned_max_parallel_f32')
        self.subtract_kernel=self.deep_module.get_function('subtract_correlations_f32')
        self.validation_kernel=self.deep_module.get_function('validate_events')
        self.parallel=parallel
        self.fallback_count=0
        self._waveforms=None
        self.waveform_builds=0

    def waveforms(self,U,W):
        import torch
        # Tensor versions invalidate the cache when templates change. Inference
        # tensors without versions are conservatively recomputed.
        try: versions=(U._version,W._version)
        except RuntimeError: versions=None
        cached=self._waveforms
        if versions is None or cached is None or cached[0]() is not U or cached[1]() is not W or cached[2]!=versions:
            waveform=torch.einsum('upc,pt->cut',U,W.contiguous()).contiguous()
            self._waveforms=(weakref.ref(U),weakref.ref(W),versions,waveform)
            self.waveform_builds+=1
        return self._waveforms[3]

    def learned_max(self,B,norm,edge):
        import torch
        if (B.ndim!=2 or B.dtype!=torch.float32 or not B.is_cuda or not B.numel() or
            norm.ndim!=1 or norm.shape[0]!=B.shape[0] or norm.device!=B.device or
            norm.dtype!=B.dtype or edge<0):
            raise ValueError('Expected CUDA float32 scores [unit,time], norms [unit], and nonnegative edge')
        maxima=torch.empty(B.shape[1],device=B.device,dtype=B.dtype)
        indices=torch.empty(B.shape[1],device=B.device,dtype=torch.int64)
        args=(np.uint64(B.data_ptr()),np.uint64(norm.data_ptr()),np.uint64(maxima.data_ptr()),np.uint64(indices.data_ptr()),
              *map(np.int64,[*B.shape,*B.stride(),norm.stride(0),edge]))
        if self.parallel:
            self._launch(self.parallel_max_kernel,B,args,B.shape[1],grid=((B.shape[1]+31)//32,),block=(128,))
        else:
            self._launch(self.max_kernel,B,args,B.shape[1])
        return maxima,indices

    def subtract_correlations(self,B,ctc,times,units,amps,radius):
        import torch
        if (B.ndim!=2 or B.dtype!=torch.float32 or not B.is_cuda or ctc.ndim!=3 or
            ctc.dtype!=B.dtype or ctc.device!=B.device or ctc.shape[0]!=B.shape[0] or
            ctc.shape[2]!=2*radius+1 or radius<0 or
            any(x.ndim!=1 or x.device!=B.device for x in (times,units,amps)) or
            times.dtype!=torch.int64 or units.dtype!=torch.int64 or amps.dtype!=B.dtype or
            not times.numel()==units.numel()==amps.numel()):
            raise ValueError('Invalid correlation-subtraction tensors')
        if not times.numel(): return
        # One small synchronization validates bounds and race-free ownership.
        flag=torch.empty((),device=B.device,dtype=torch.int32)
        validation_args=(np.uint64(times.data_ptr()),np.uint64(units.data_ptr()),np.uint64(flag.data_ptr()),
                         *map(np.int64,[times.numel(),B.shape[1],ctc.shape[1],radius,times.stride(0),units.stride(0)]))
        self._launch(self.validation_kernel,B,validation_args,1,grid=(1,),block=(256,))
        state=int(flag)
        if state&1: raise ValueError('Event window extends beyond score tensor')
        if state&4: raise ValueError('Unit index is out of bounds')
        if state&2:
            self.fallback_count+=1
            trange=torch.arange(-radius,radius+1,device=B.device)
            B[:,times[:,None]+trange]-=amps[:,None]*ctc[:,units,:]
            return
        args=(np.uint64(B.data_ptr()),np.uint64(ctc.data_ptr()),np.uint64(times.data_ptr()),
              np.uint64(units.data_ptr()),np.uint64(amps.data_ptr()),
              *map(np.int64,[B.shape[0],times.numel(),radius,*B.stride(),*ctc.stride(),times.stride(0),units.stride(0),amps.stride(0)]))
        self._launch(self.subtract_kernel,B,args,B.shape[0]*times.numel()*(2*radius+1))


def make_deep_matching(backend,mode='both'):
    from kilosort import template_matching
    source=inspect.getsource(template_matching.run_matching)
    if hashlib.sha256(source.encode()).hexdigest()!=EXPECTED_LEARNED_SOURCE_SHA256:
        raise RuntimeError('Kilosort learned-matching source changed; refusing unverified substitution')
    if mode not in ('max','subtract','both','all'): raise ValueError('Invalid deep matching mode')
    if mode in ('max','both','all'):
        start=source.index('        Cf = torch.relu(B)')
        stop=source.index('        Cmax  = max_pool1d',start)
        source=source[:start]+'        Cfmax, imax = _deep.learned_max(B, nm, nt)\n'+source[stop:]
    if mode in ('subtract','both','all'):
        old='            B[   :, iX[j::n] + trange]  -= amp[j::n] * ctc[:,iY[j::n,0],:]'
        assert source.count(old)==1
        source=source.replace(old,'            _deep.subtract_correlations(B, ctc, iX[j::n,0], iY[j::n,0], amp[j::n,0], nt)')
    if mode=='all':
        # Cache against the persistent PCA tensor, not the per-batch contiguous
        # copy, which otherwise expires and unnecessarily rebuilds the bank.
        source=source.replace("    W = ops['wPCA'].contiguous()","    W = ops['wPCA'].contiguous()\n    waveform_bank = _deep.waveforms(U, ops['wPCA'])")
        old="            Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n,0]], W)"
        assert source.count(old)==1
        source=source.replace(old,'            _deep.subtract_correlations(Xres, waveform_bank, iX[j::n,0], iY[j::n,0], amp[j::n,0], nt//2)')
    namespace=dict(vars(template_matching),_deep=backend)
    exec(compile(source,'<Kilosort 4.1.7 with learned CUDA fusion>','exec'),namespace)
    return namespace['run_matching'],source


@contextmanager
def deep_kilosort():
    from kilosort import template_matching
    with native_kilosort_reductions('fused') as info:
        original=template_matching.run_matching
        backend=DeepCuda(parallel=True)
        replacement,source=make_deep_matching(backend,'all')
        template_matching.run_matching=replacement
        try:
            yield dict(info,mode='deep',learned_generated_source=source,learned_source_sha256=EXPECTED_LEARNED_SOURCE_SHA256,
                       runtime_stats=lambda:dict(overlap_fallback_calls=backend.fallback_count,
                                                waveform_bank_builds=backend.waveform_builds,
                                                learned_mode='all_parallel'))
        finally:
            template_matching.run_matching=original
