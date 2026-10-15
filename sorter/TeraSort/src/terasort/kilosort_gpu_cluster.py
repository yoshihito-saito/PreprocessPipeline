"""Bounded GPU acceleration for Kilosort's exact template-neighbor search."""

from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import logging

import numpy as np
import torch
from scipy.sparse import csr_matrix


_KILOSORT_VERSION = "4.1.7"
_NEIGH_MAT_SHA256 = "5047c8cda033ade0ec8edd86d29d0a9057261af428c133baf4b8e9e5ac6272fb"
_MAX_DISTANCE_BYTES = 256 * 1024**2
logger = logging.getLogger(__name__)


def _same_neighbor_sets(left, right):
    return np.array_equal(np.sort(left, axis=1), np.sort(right, axis=1))


def _gpu_neigh_mat_factory(module, reference, device):
    def neigh_mat(Xd, nskip=1, n_neigh=10, max_sub=25000):
        if nskip < 1 or n_neigh < 1:
            return reference(Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub)

        Xd_np = Xd.numpy() if torch.is_tensor(Xd) else np.asarray(Xd)
        if Xd_np.ndim != 2 or Xd_np.dtype != np.float32 or not np.isfinite(Xd_np).all():
            return reference(Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub)

        Xsub = Xd[::nskip]
        n1 = Xsub.shape[0]
        rev_idx = None
        if max_sub is not None and n1 > max_sub:
            idx, rev_idx = module.subsample_idx(n1, n1 - max_sub)
            Xsub = Xsub[idx]
        Xd_np = np.ascontiguousarray(Xd_np)
        Xsub_np = np.ascontiguousarray(Xsub.numpy() if torch.is_tensor(Xsub) else Xsub)
        n_samples, dim = Xd_np.shape
        n_nodes = Xsub_np.shape[0]
        if n_nodes <= n_neigh:
            return reference(Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub)

        try:
            kn = _search_gpu_with_cpu_guards(
                module, Xd_np, Xsub_np, n_neigh, device=device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU neighbor search ran out of memory; using Kilosort's CPU search")
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            return reference(Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub)

        dexp = np.ones(kn.shape, np.float32)
        rows = np.tile(np.arange(n_samples, dtype=np.int64)[:, np.newaxis],
                       (1, n_neigh)).ravel()
        M = csr_matrix((dexp.ravel(), (rows, kn.ravel())),
                       (n_samples, n_nodes))
        skip_idx = np.arange(0, n_samples, nskip)
        if rev_idx is not None:
            skip_idx = skip_idx[rev_idx]
        M[skip_idx, np.arange(n_nodes)] = 0
        return kn, M

    return neigh_mat


def _search_gpu_with_cpu_guards(module, queries, candidates, n_neigh, *, device):
    n_samples, _ = queries.shape
    n_nodes = candidates.shape[0]
    cpu_kn = np.empty((n_samples, n_neigh), dtype=np.int64)
    ambiguous = np.zeros(n_samples, dtype=bool)
    # Budget the distance matrix to at most half of 256 MiB, leaving headroom
    # for top-k workspaces and concurrent sort jobs.
    batch = max(1, min(1024, _MAX_DISTANCE_BYTES // (n_nodes * 4 * 2)))
    eps = torch.finfo(torch.float32).eps

    with torch.no_grad(), torch.cuda.device(device):
        candidate_gpu = torch.from_numpy(candidates).to(device)
        candidate_norm = (candidate_gpu * candidate_gpu).sum(dim=1)
        for start in range(0, n_samples, batch):
            stop = min(start + batch, n_samples)
            query_gpu = torch.from_numpy(queries[start:stop]).to(device)
            query_norm = (query_gpu * query_gpu).sum(dim=1, keepdim=True)
            distances = query_norm + candidate_norm.unsqueeze(0) - 2 * (query_gpu @ candidate_gpu.T)
            distances.clamp_min_(0)
            values, indices = torch.topk(
                distances, n_neigh + 1, dim=1, largest=False, sorted=True)
            cpu_kn[start:stop] = indices[:, :n_neigh].cpu().numpy()

            # Float32 accumulation can reorder almost-tied neighbors. Send
            # those rows to FAISS so their graph edges exactly match Kilosort.
            scale = query_norm.squeeze(1) + candidate_norm[indices[:, n_neigh]]
            tolerance = (8 * eps * queries.shape[1]) * scale.clamp_min(1)
            gap = values[:, n_neigh] - values[:, n_neigh - 1]
            ambiguous[start:stop] = (gap <= tolerance).cpu().numpy()

    ambiguous_rows = np.flatnonzero(ambiguous)
    if ambiguous_rows.size > n_samples // 10:
        raise _UseReferenceSearch("too many numerically ambiguous GPU neighbors")

    index = module.faiss.IndexFlatL2(queries.shape[1])
    index.add(candidates)
    if ambiguous_rows.size:
        _, exact = index.search(queries[ambiguous_rows], n_neigh)
        cpu_kn[ambiguous_rows] = exact

    safe_rows = np.flatnonzero(~ambiguous)
    if safe_rows.size:
        check_count = min(256, max(8, n_samples // 1024), safe_rows.size)
        check_rows = safe_rows[np.linspace(0, safe_rows.size - 1,
                                           num=check_count, dtype=np.int64)]
        _, exact = index.search(queries[check_rows], n_neigh)
        if not _same_neighbor_sets(cpu_kn[check_rows], exact):
            raise _UseReferenceSearch("GPU neighbor check differed from FAISS")

    logger.debug("GPU neighbor search: %d query rows; FAISS corrected %d borderline rows",
                 n_samples, ambiguous_rows.size)
    return cpu_kn


class _UseReferenceSearch(Exception):
    """Signal a numerical check failure that should safely retry on CPU."""


@contextmanager
def vectorized_kilosort_neighbors(device=None):
    """Use GPU search only when the installed Kilosort code is the verified build."""
    if importlib.metadata.version("kilosort") != _KILOSORT_VERSION or not torch.cuda.is_available():
        yield False
        return

    requested = torch.device("cuda" if device is None else device)
    if requested.type != "cuda":
        yield False
        return

    module = importlib.import_module("kilosort.clustering_qr")
    source = inspect.getsource(module.neigh_mat)
    if hashlib.sha256(source.encode()).hexdigest() != _NEIGH_MAT_SHA256:
        logger.warning("Kilosort neighbor search differs from the validated 4.1.7 source; "
                       "using the original implementation")
        yield False
        return

    original = module.neigh_mat
    optimized = _gpu_neigh_mat_factory(module, original, requested)

    def guarded(*args, **kwargs):
        try:
            return optimized(*args, **kwargs)
        except _UseReferenceSearch as exc:
            logger.warning("GPU neighbor validation failed (%s); retrying with FAISS CPU", exc)
            return original(*args, **kwargs)

    module.neigh_mat = guarded
    try:
        logger.info("TeraSort GPU clustering neighbor search enabled; near-tie rows are "
                    "checked with FAISS on the CPU")
        yield True
    finally:
        module.neigh_mat = original
