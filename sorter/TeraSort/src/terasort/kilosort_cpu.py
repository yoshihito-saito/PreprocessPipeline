"""Scoped CPU-side Kilosort accelerations with a pinned-source guard."""

from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import logging
import os

import torch


_KILOSORT_VERSION = "4.1.7"
_GET_DATA_CPU_SHA256 = "6c47db7f1a44f31a1a7d9fb1d250cae224889fe3d1f5f141d5c332708a0bcc8f"
logger = logging.getLogger(__name__)


@contextmanager
def bounded_faiss_threads(maximum=8):
    """Cap FAISS oversubscription during Kilosort, then restore its prior limit."""
    import faiss

    previous = faiss.omp_get_max_threads()
    available = os.cpu_count() or maximum
    selected = min(previous, maximum, available)
    if selected != previous:
        faiss.omp_set_num_threads(selected)
        logger.info("TeraSort limits FAISS to %d CPU threads for this sort", selected)
    try:
        yield selected
    finally:
        if selected != previous:
            faiss.omp_set_num_threads(previous)


def _make_vectorized_gather(reference):
    def get_data_cpu(ops, xy, iC, PID, tF, ycenter, xcenter, dmin=20,
                     dminx=32, ix=None, merge_dim=True):
        # Match Kilosort's input conversion and spatial-template selection.
        PID = torch.from_numpy(PID).long()
        y0 = ycenter
        x0 = xcenter
        if ix is None:
            ix = torch.logical_and(torch.abs(xy[1] - y0) < dmin,
                                   torch.abs(xy[0] - x0) < dminx)
        igood = ix[PID].nonzero()[:, 0]
        if len(igood) == 0:
            return None, None, None

        pid = PID[igood]
        data = tF[igood]
        nspikes, nchanraw, nfeatures = data.shape
        template_ids = ix.nonzero()[:, 0]
        template_channels = iC[:, ix]
        ichan, imap = torch.unique(template_channels, return_inverse=True)

        # Kilosort overwrites earlier source channels if a template repeats a
        # channel. Keep its exact behavior for that unusual probe mapping.
        ordered = torch.sort(template_channels, dim=0).values
        if ordered.shape[0] > 1 and torch.any(ordered[1:] == ordered[:-1]):
            return reference(ops, xy, iC, PID.numpy(), tF, ycenter, xcenter,
                             dmin=dmin, dminx=dminx, ix=ix,
                             merge_dim=merge_dim)

        template_map = imap.reshape(nchanraw, template_ids.numel())
        # Every event maps its raw feature channels directly into the union
        # of channels used by templates in this spatial neighborhood.
        local_template = torch.cumsum(ix.to(torch.int64), dim=0) - 1
        channel_map = template_map[:, local_template[pid]].T.contiguous()
        dd = torch.zeros((nspikes, ichan.numel(), nfeatures), dtype=data.dtype)
        rows = torch.arange(nspikes).unsqueeze(1)
        dd[rows, channel_map] = data

        if merge_dim:
            return dd.reshape(nspikes, -1), igood, ichan
        return dd, igood, ichan

    return get_data_cpu


@contextmanager
def vectorized_kilosort_gather():
    """Patch Kilosort's repeated CPU gather only for its verified 4.1.7 code."""
    if importlib.metadata.version("kilosort") != _KILOSORT_VERSION:
        yield False
        return

    clustering_qr = importlib.import_module("kilosort.clustering_qr")
    source = inspect.getsource(clustering_qr.get_data_cpu)
    if hashlib.sha256(source.encode()).hexdigest() != _GET_DATA_CPU_SHA256:
        logger.warning("Kilosort CPU gather differs from the validated 4.1.7 source; "
                       "using the original implementation")
        yield False
        return

    postprocessing = importlib.import_module("kilosort.postprocessing")
    original = clustering_qr.get_data_cpu
    fast = _make_vectorized_gather(original)
    postprocessing_original = postprocessing.get_data_cpu
    clustering_qr.get_data_cpu = fast
    postprocessing.get_data_cpu = fast
    try:
        logger.info("TeraSort vectorized Kilosort feature gather enabled")
        yield True
    finally:
        clustering_qr.get_data_cpu = original
        postprocessing.get_data_cpu = postprocessing_original
