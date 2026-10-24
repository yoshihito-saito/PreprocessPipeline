"""Direct-CUDA batched scorer for the experimental local template bank."""

from pathlib import Path
import numpy as np

from .cuda_environment import configure_cupy_cache


class CudaLocalMatcher:
    """Keep templates resident on one GPU; score only spatially eligible pairs.

    The immutable per-epoch template snapshots are refreshed only when their
    version changes. Candidate waveforms are transferred in bounded batches.
    """

    def __init__(self, bank):
        configure_cupy_cache()
        import cupy as cp
        self.cp = cp
        self.bank = bank
        self.ids = sorted(bank.units)
        self.index = {unit_id: i for i, unit_id in enumerate(self.ids)}
        if self.ids:
            width = len(bank.units[self.ids[0]].channels)
            if width > 32:
                raise ValueError("CUDA local matcher supports up to 32 contacts per template")
            if any(len(bank.units[u].channels) != width for u in self.ids):
                raise ValueError("CUDA templates need a fixed local width")
            self.template_channels = cp.asarray(np.stack(
                [bank.units[u].channels for u in self.ids]).astype(np.int32))
        else:
            self.template_channels = cp.empty((0, 0), cp.int32)
        self.templates = None
        self.versions = {}
        code = Path(__file__).with_name("streaming_match.cu").read_text()
        module = cp.RawModule(code=code, options=("--std=c++11", "--fmad=false"))
        self.score_kernel = module.get_function("score_pairs")
        self.best_kernel = module.get_function("best_two")

    def _refresh(self, snapshots):
        cp = self.cp
        if self.templates is None:
            values = np.stack([snapshots[u].waveform for u in self.ids])
            self.templates = cp.asarray(values, dtype=cp.float32)
            self.versions = {u: snapshots[u].version for u in self.ids}
            return
        changed = [u for u in self.ids if snapshots[u].version != self.versions[u]]
        if changed:
            indices = cp.asarray([self.index[u] for u in changed], dtype=cp.int32)
            values = cp.asarray(np.stack([snapshots[u].waveform for u in changed]),
                               dtype=cp.float32)
            self.templates[indices] = values
            for u in changed:
                self.versions[u] = snapshots[u].version

    def match(self, events, snapshots):
        if not events:
            return []
        if not self.ids:
            return [(-1, -1., -1., -1, None) for _ in events]
        cp = self.cp
        self._refresh(snapshots)
        width = len(events[0].channels)
        n_time = len(events[0].waveform)
        if width > 32 or width != self.templates.shape[2]:
            raise ValueError("Event and CUDA template widths differ or exceed 32")
        if any(e.waveform.shape != (n_time, width) for e in events):
            raise ValueError("Inconsistent event waveform shape")
        pair_events, pair_templates = [], []
        offsets = [0]
        for i, event in enumerate(events):
            ids = self.bank.candidate_ids(event)
            pair_events.extend([i] * len(ids))
            pair_templates.extend(self.index[unit_id] for unit_id in ids)
            offsets.append(len(pair_events))
        if not pair_events:
            return [(-1, -1., -1., -1, None) for _ in events]
        e_wave = cp.asarray(np.stack([e.waveform for e in events]), dtype=cp.float32)
        e_channels = cp.asarray(np.stack([e.channels for e in events]), dtype=cp.int32)
        pe = cp.asarray(pair_events, dtype=cp.int32)
        pu = cp.asarray(pair_templates, dtype=cp.int32)
        po = cp.asarray(offsets, dtype=cp.int32)
        scores = cp.empty(len(pair_events), dtype=cp.float32)
        amplitudes = cp.empty_like(scores)
        shifts = cp.empty(len(pair_events), dtype=cp.int32)
        self.score_kernel((len(pair_events),), (32,),
            (e_wave, e_channels, self.templates, self.template_channels,
             pe, pu, np.int32(len(pair_events)), np.int32(n_time), np.int32(width),
             scores, amplitudes, shifts))
        best_unit = cp.empty(len(events), cp.int32)
        best_score = cp.empty(len(events), cp.float32)
        runner = cp.empty(len(events), cp.float32)
        best_amp = cp.empty(len(events), cp.float32)
        best_shift = cp.empty(len(events), cp.int32)
        self.best_kernel((len(events),), (32,),
            (po, pu, scores, amplitudes, shifts, np.int32(len(events)),
             best_unit, best_score, runner, best_amp, best_shift))
        unit, score, second, amp, shift = map(cp.asnumpy,
            (best_unit, best_score, runner, best_amp, best_shift))
        result = []
        for i, event in enumerate(events):
            unit_id = self.ids[int(unit[i])] if unit[i] >= 0 else -1
            result.append(self.bank.finalize_assignment(
                event, unit_id, float(score[i]), float(second[i]),
                float(amp[i]), int(shift[i]), snapshots))
        return result
