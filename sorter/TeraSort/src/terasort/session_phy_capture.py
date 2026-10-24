"""Bounded measured curation features; no detector or fit decisions change."""
import numpy as np

PATCH_ROWS=256

def validate_basis(basis):
    basis=np.asarray(basis)
    if (basis.ndim!=2 or basis.shape[0]<21 or basis.shape[0]%2!=1 or basis.shape[1]!=3) or basis.dtype!=np.float32 or not np.isfinite(basis).all():
        raise ValueError('Invalid temporal Phy PCA basis')
    if not np.allclose(basis.T@basis,np.eye(3),rtol=1e-5,atol=1e-5):
        raise ValueError('Temporal Phy PCA basis is not orthonormal')

def curation_basis(models):
    from .session_novelty import seed_temporal_basis
    basis=seed_temporal_basis(models)
    for pc in range(3):
        if basis[np.argmax(np.abs(basis[:,pc])),pc]<0:basis[:,pc]*=-1
    validate_basis(basis)
    return basis

def project_signal(signal,times,channels,basis,*,units=None):
    """Project <=256 actual nt-sample, <=16-channel observations at a time."""
    validate_basis(basis)
    if (channels.ndim!=2 or (units is None and len(channels)!=len(times)) or not 1<=channels.shape[1]<=16
            or np.any(channels < -1) or np.any(channels>=signal.shape[1])):
        raise ValueError('Invalid measured Phy feature contacts')
    if units is not None and (len(units)!=len(times) or np.any(units<0) or np.any(units>=len(channels))):
        raise ValueError('Invalid measured Phy template IDs')
    for first in range(0,len(times),PATCH_ROWS):
        stop=min(len(times),first+PATCH_ROWS)
        t=np.asarray(times[first:stop],np.int64)
        c=channels[first:stop] if units is None else channels[units[first:stop]]
        indices=t[:,None]+np.arange(-(len(basis)//2),len(basis)//2+1,dtype=np.int64)[None]
        valid=(indices>=0)&(indices<len(signal))
        patches=signal[np.clip(indices,0,len(signal)-1)[:,:,None],np.maximum(c,0)[:,None,:]]
        patches*=valid[:,:,None]&(c[:,None,:]>=0)
        pc=np.einsum('tp,ntc->npc',basis,patches,optimize=True).astype(np.float32)
        if not np.isfinite(pc).all():raise ValueError('Nonfinite measured Phy PCA features')
        yield pc

def project_observations(observations,samples,basis,width):
    """Recover PCs only from stored raw INT16 snippets, never denoised proposals."""
    by_sample={row[0]:row for row in observations}
    result=np.empty((len(samples),3,width),np.float32)
    for i,sample in enumerate(samples):
        row=by_sample[int(sample)]
        raw=row[4].astype(np.float32)*row[5]
        result[i]=np.einsum('tp,tc->pc',basis,raw[:,:width])
    return result
