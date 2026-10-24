"""Opt-in, source-guarded CUDA assignment updates for bounded KS4 seed learning."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import logging
from pathlib import Path

import numpy as np

from .cuda_environment import configure_cupy_cache

_SOURCE_SHA256 = "e7c34840a221aa5531206b4bf2c70a0f0404ed9904239f608436270c30a9d9dc"


def _make_kmeans_update(reference):
    import torch
    configure_cupy_cache()
    import cupy as cp
    module = cp.RawModule(code=Path(__file__).with_name("seed_cluster_update.cu").read_text(),
                          options=("--std=c++11",))
    kernel = module.get_function("seed_cluster_update")

    def kmeans_plusplus(Xg, niter=200, seed=1, device=torch.device("cuda"), verbose=False):
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        if (not Xg.is_cuda or Xg.dtype != torch.float32 or Xg.device != requested
                or torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled()
                or Xg.shape[0] > 2**24 or verbose or not isinstance(niter, int) or niter < 0):
            return reference(Xg, niter=niter, seed=seed, device=device, verbose=verbose)
        vtot = torch.norm(Xg, 2, dim=1)**2
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_spikes = Xg.shape[0]
        vexp0 = torch.zeros(n_spikes, device=device)
        iclust = torch.zeros(n_spikes, dtype=torch.int, device=device)
        stream = cp.cuda.Stream.from_external(torch.cuda.current_stream(Xg.device))
        # The pinned original writes mu but never reads or returns it. Omit
        # that scratch bank and its scalar GPU indexing without changing RNG.
        for j in range(niter):
            v2 = torch.relu(vtot-vexp0)
            isamp = torch.multinomial(v2, 100)
            Xc = Xg[isamp]
            vexp = 2*Xg @ Xc.T - (Xc**2).sum(1)
            dexp = torch.relu(vexp-vexp0.unsqueeze(1))
            imax = torch.argmax(dexp.sum(0))
            with cp.cuda.Device(Xg.device.index), stream:
                # Borrow Torch buffers on its active stream; ownership stays
                # with Torch and no full feature/candidate array is copied.
                kernel(((n_spikes+255)//256,), (256,),
                    tuple(cp.from_dlpack(a) for a in (vexp, dexp, imax, vexp0, iclust))
                    + (np.int32(n_spikes), np.int32(100), np.int32(j)))
            del vexp, dexp
        return iclust
    return kmeans_plusplus


@contextmanager
def cuda_seed_cluster_updates(*, enabled=True):
    """Scope the patch to the verified KS4 function; restore even on failure."""
    if not enabled:
        yield False
        return
    import kilosort.clustering_qr as clustering
    original = clustering.kmeans_plusplus
    try:
        supported = (importlib.metadata.version("kilosort") == "4.1.7" and
            hashlib.sha256(inspect.getsource(original).encode()).hexdigest() == _SOURCE_SHA256)
    except (OSError, TypeError, importlib.metadata.PackageNotFoundError):
        supported = False
    if not supported:
        logging.getLogger(__name__).warning("Seed clustering source differs; using original KS4 updates")
        yield False
        return
    fast = _make_kmeans_update(original)
    clustering.kmeans_plusplus = fast
    try:
        yield True
    finally:
        clustering.kmeans_plusplus = original
