"""Bounded, causal assignment of SCB waveform candidates to local templates.

This is an experimental CPU reference for the streaming path. It deliberately
keeps the original SCB detector events immutable and leaves uncertain events
unassigned. Templates are seeded by a separate calibration sorter; discovery
of new units and overlap deconvolution are outside this first pass.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event as ThreadEvent, Thread

import h5py
import numpy as np

from .candidates.bank import CandidateBank
from .matching.rolling import RollingTemplate


ASSIGNMENT_DTYPE = np.dtype([
    ("sample_index", "<i8"), ("anchor_sample_index", "<i8"),
    ("anchor_channel", "<u4"), ("amplitude_uv", "<f4"),
    ("snr", "<f4"), ("unit_id", "<i8"), ("score", "<f4"),
    ("runner_up", "<f4"), ("template_version", "<i4"),
    ("candidate_start", "<i8"), ("candidate_stop", "<i8"),
])


@dataclass(frozen=True)
class ConsolidatedEvent:
    sample_index: int  # earliest source detection in the group
    anchor_sample_index: int  # time of the highest-SNR detection
    anchor_channel: int
    amplitude_uv: float
    snr: float
    channels: np.ndarray
    waveform: np.ndarray
    candidate_rows: tuple[int, ...]


@dataclass
class _Group:
    first_sample: int
    anchor_sample: int
    anchor_channel: int
    amplitude_uv: float
    snr: float
    channels: np.ndarray
    waveform: np.ndarray
    rows: list[int] = field(default_factory=list)

    def finish(self):
        return ConsolidatedEvent(
            self.first_sample, self.anchor_sample, self.anchor_channel,
            self.amplitude_uv, self.snr, self.channels, self.waveform,
            tuple(self.rows))


def iter_consolidated(bank: CandidateBank, *, floor_snr: float,
                      radius_samples: int = 2, batch_size: int = 512,
                      max_pending: int = 4096,
                      start_sample=None, stop_sample=None):
    """Group nearby same-polarity channel detections without reading raw data.

    All original SCB rows remain addressable in ``candidate_rows``. This local
    grouping can fuse simultaneous neurons; it is not overlap deconvolution.
    A pathological burst fails explicitly instead of growing memory forever.
    """
    if bank.waveform_spec is None:
        raise ValueError("SCB 0.2 waveform cache required")
    if radius_samples < 0 or batch_size < 1 or max_pending < 1:
        raise ValueError("Invalid consolidation budget")
    pending: list[_Group] = []
    for batch in bank.iter_waveforms(floor_snr, batch_size=batch_size,
                                     start_sample=start_sample,
                                     stop_sample=stop_sample):
        for i, row in enumerate(batch.row_indices):
            event = batch.events[i]
            t, c = int(event["sample_index"]), int(event["channel_index"])
            still = []
            for group in pending:
                if t - group.first_sample > radius_samples:
                    yield group.finish()
                else:
                    still.append(group)
            pending = still
            channels = np.asarray(batch.channel_indices[i], np.int32)
            waveform = np.asarray(batch.waveforms_uv[i], np.float32)
            polarity = np.signbit(float(event["amplitude_uv"]))
            eligible = [g for g in pending
                        if (polarity == np.signbit(g.amplitude_uv)
                            and (c in g.channels or g.anchor_channel in channels))]
            if eligible:
                group = min(eligible, key=lambda g: (abs(t - g.anchor_sample), -g.snr))
                group.rows.append(int(row))
                if float(event["snr"]) > group.snr:
                    group.anchor_sample, group.anchor_channel = t, c
                    group.amplitude_uv, group.snr = (float(event["amplitude_uv"]),
                                                     float(event["snr"]))
                    group.channels, group.waveform = channels, waveform
            else:
                pending.append(_Group(t, t, c, float(event["amplitude_uv"]),
                                      float(event["snr"]), channels, waveform,
                                      [int(row)]))
                if len(pending) > max_pending:
                    raise OverflowError("Too many simultaneous event groups")
    for group in pending:
        yield group.finish()


def _shifted_similarity(template: np.ndarray, template_channels: np.ndarray,
                        event: ConsolidatedEvent, max_shift: int = 2):
    """Cosine, fitted amplitude and shift on common physical contacts."""
    shared, ti, ei = np.intersect1d(template_channels, event.channels,
                                    assume_unique=False, return_indices=True)
    keep = shared >= 0
    ti, ei = ti[keep], ei[keep]
    if not len(ti):
        return -1., np.nan, 0
    a = template[:, ti]
    b = event.waveform[:, ei]
    a_total = np.square(template[:, template_channels >= 0]).sum()
    b_total = np.nansum(np.square(event.waveform[:, event.channels >= 0]))
    if a_total <= 0 or b_total <= 0:
        return -1., np.nan, 0
    if np.square(a).sum() < .7 * a_total or np.nansum(np.square(b)) < .7 * b_total:
        return -1., np.nan, 0
    best = (-1., np.nan, 0)
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            left, right = a[:len(a)-shift or None], b[shift:]
        else:
            left, right = a[-shift:], b[:shift]
        valid = np.isfinite(right)
        if valid.mean() < .9:
            continue
        aa, bb = left[valid].astype(np.float64), right[valid].astype(np.float64)
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom <= 0:
            continue
        dot = float(aa @ bb)
        score = dot / denom
        if score > best[0]:
            best = (score, dot / float(aa @ aa), shift)
    return best


@dataclass
class _Unit:
    unit_id: int
    anchor_channel: int
    channels: np.ndarray
    rolling: RollingTemplate
    assigned_count: int = 0


class LocalTemplateBank:
    """Global unit IDs, at most ``slots_per_anchor`` templates per contact."""

    def __init__(self, units: list[_Unit], *, slots_per_anchor: int = 32,
                 score_floor: float = .88, min_margin: float = .03,
                 update_floor: float = .96, update_margin: float = .08,
                 max_matches_per_event: int = 512):
        if (slots_per_anchor < 1 or max_matches_per_event < 1
                or not 0 <= score_floor <= update_floor <= 1
                or not 0 <= min_margin <= update_margin <= 1):
            raise ValueError("Invalid local matching settings")
        self.units = {u.unit_id: u for u in units}
        if len(self.units) != len(units):
            raise ValueError("Duplicate global unit ID")
        self.by_anchor: dict[int, list[int]] = {}
        for unit in units:
            slot = self.by_anchor.setdefault(unit.anchor_channel, [])
            slot.append(unit.unit_id)
            if len(slot) > slots_per_anchor:
                raise OverflowError(f"Anchor {unit.anchor_channel} exceeds {slots_per_anchor} templates")
        self.slots_per_anchor = slots_per_anchor
        self.score_floor, self.min_margin = score_floor, min_margin
        self.update_floor, self.update_margin = update_floor, update_margin
        self.max_matches_per_event = max_matches_per_event

    @classmethod
    def from_dense_templates(cls, templates, channel_map, *, pre_samples: int,
                             sample_rate_hz: float, n_samples: int | None = None,
                             start_sample: int = 0,
                             window_seconds: int = 1800,
                             bin_seconds: int = 300, **kwargs):
        """Seed from a calibration sort's unwhitened time x channel templates."""
        templates = np.load(templates, mmap_mode="r") if isinstance(templates, (str, Path)) else np.asarray(templates)
        channel_map = np.asarray(channel_map, np.int32)
        if (templates.ndim != 3 or channel_map.ndim != 2
                or templates.shape[2] != len(channel_map)
                or templates.shape[1] != (n_samples if n_samples is not None
                                          else templates.shape[1])
                or not 0 <= pre_samples < templates.shape[1]):
            raise ValueError("Templates must be unit x waveform sample x recording channel")
        units = []
        for unit_id in range(len(templates)):
            dense = np.asarray(templates[unit_id], np.float32)
            if not np.isfinite(dense).all():
                raise ValueError("Nonfinite seed template")
            peak = np.max(np.abs(dense), axis=0)
            anchor = int(np.argmax(peak))
            if peak[anchor] <= 0:
                continue
            channels = channel_map[anchor].copy()
            local = np.zeros((dense.shape[0], len(channels)), np.float32)
            local[:, channels >= 0] = dense[:, channels[channels >= 0]]
            peak_time = int(np.argmax(np.abs(dense[:, anchor])))
            offset = pre_samples - peak_time
            local = np.roll(local, offset, axis=0)
            if offset > 0:
                local[:offset] = 0
            elif offset < 0:
                local[offset:] = 0
            rolling = RollingTemplate(local, sample_rate_hz=sample_rate_hz,
                                      coordinate_frame="recorded_channels:v1",
                                      start_sample=start_sample,
                                      window_seconds=window_seconds,
                                      bin_seconds=bin_seconds)
            units.append(_Unit(unit_id, anchor, channels, rolling))
        return cls(units, **kwargs)

    def snapshots(self):
        return {unit_id: unit.rolling.snapshot() for unit_id, unit in self.units.items()}

    def candidate_ids(self, event: ConsolidatedEvent):
        ids = sorted({unit_id for channel in event.channels if channel >= 0
                      for unit_id in self.by_anchor.get(int(channel), ())})
        if len(ids) > self.max_matches_per_event:
            raise OverflowError("Too many local templates for one event")
        return ids

    def assign(self, event: ConsolidatedEvent, snapshots):
        ids = self.candidate_ids(event)
        scored = []
        for unit_id in ids:
            unit = self.units[unit_id]
            score, amplitude, shift = _shifted_similarity(
                snapshots[unit_id].waveform, unit.channels, event)
            if np.isfinite(amplitude) and amplitude > 0:
                scored.append((score, unit_id, amplitude, shift))
        scored.sort(reverse=True)
        best = scored[0] if scored else (-1., -1, np.nan, 0)
        runner_up = scored[1][0] if len(scored) > 1 else -1.
        score, unit_id, amplitude, shift = best
        return self.finalize_assignment(event, unit_id, score, runner_up,
                                        amplitude, shift, snapshots)

    def finalize_assignment(self, event, unit_id, score, runner_up,
                            amplitude, shift, snapshots):
        margin = score - runner_up
        assigned = bool(unit_id >= 0 and score >= self.score_floor
                        and margin >= self.min_margin)
        clean = event.waveform.copy()
        clean[:, event.channels < 0] = 0
        update = bool(assigned and score >= self.update_floor
                      and margin >= self.update_margin and shift == 0
                      and .5 <= amplitude <= 2.
                      and event.anchor_channel == self.units[unit_id].anchor_channel
                      and np.array_equal(event.channels, self.units[unit_id].channels)
                      and np.isfinite(clean).all())
        return (unit_id if assigned else -1, float(score), float(runner_up),
                snapshots[unit_id].version if assigned else -1,
                clean / amplitude if update else None)


class AssignmentWriter:
    """Append bounded batches and publish a completed, separate HDF5 shard."""

    def __init__(self, output, *, source, config):
        self.output = Path(output)
        self.partial = self.output.with_suffix(self.output.suffix + ".partial")
        if self.output.exists() or self.partial.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial, "x")
        self.handle.attrs["complete"] = False
        self.handle.attrs["manifest_json"] = json.dumps({"format": "TSA", "version": "0.1",
            "source_bank": str(Path(source).resolve()), "config": config}, sort_keys=True)
        self.events = self.handle.create_dataset("events", shape=(0,), maxshape=(None,),
                         dtype=ASSIGNMENT_DTYPE, chunks=(4096,), compression="lzf", shuffle=True)
        self.members = self.handle.create_dataset("candidate_rows", shape=(0,), maxshape=(None,),
                          dtype="<i8", chunks=(16384,), compression="lzf", shuffle=True)
        self.count = 0
        self.candidate_count = 0

    def append(self, rows, member_lists):
        if len(rows) != len(member_lists):
            raise ValueError("Event/member count mismatch")
        start = len(self.members)
        members = np.concatenate([np.asarray(x, np.int64) for x in member_lists]) if rows else np.empty(0, np.int64)
        data = np.asarray(rows, ASSIGNMENT_DTYPE)
        lengths = np.fromiter((len(x) for x in member_lists), np.int64, len(rows))
        data["candidate_start"] = start + np.r_[0, np.cumsum(lengths)[:-1]]
        data["candidate_stop"] = data["candidate_start"] + lengths
        a, b = len(self.events), len(self.events) + len(data)
        self.events.resize((b,))
        self.events[a:b] = data
        self.members.resize((start + len(members),))
        self.members[start:] = members
        self.handle.flush()
        self.count += len(data)
        self.candidate_count += len(members)

    def finish(self, template_bank: LocalTemplateBank):
        units = [template_bank.units[i] for i in sorted(template_bank.units)]
        group = self.handle.create_group("final_templates")
        group.create_dataset("unit_id", data=np.asarray([u.unit_id for u in units], np.int64))
        group.create_dataset("anchor_channel", data=np.asarray([u.anchor_channel for u in units], np.int32))
        group.create_dataset("assigned_count", data=np.asarray([u.assigned_count for u in units], np.int64))
        if units:
            group.create_dataset("channels", data=np.stack([u.channels for u in units]))
            group.create_dataset("waveform", data=np.stack([u.rolling.snapshot().waveform for u in units]))
            group.create_dataset("version", data=np.asarray([u.rolling.version for u in units], np.int64))
        self.handle.attrs["complete"] = True
        self.handle.flush()
        self.handle.close()
        self.handle = None
        os.replace(self.partial, self.output)

    def abort(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _same_patch(a: ConsolidatedEvent, b: ConsolidatedEvent):
    return (a.anchor_channel in b.channels or b.anchor_channel in a.channels)


def _iter_epochs(events, *, bin_samples, max_epoch_events):
    """Bound each GPU batch; expose one lookahead event for update isolation."""
    epoch = []
    for event in events:
        boundary = bool(epoch and (
            event.sample_index // bin_samples != epoch[0].sample_index // bin_samples
            or (len(epoch) >= max_epoch_events
                and event.sample_index > epoch[-1].sample_index)))
        if boundary:
            yield epoch, event
            epoch = []
        epoch.append(event)
        if len(epoch) > max_epoch_events and event.sample_index == epoch[0].sample_index:
            raise OverflowError("Too many events at one sample for epoch budget")
    if epoch:
        yield epoch, None


@contextmanager
def _prefetched(iterable, *, depth=2):
    """Overlap SCB decoding/consolidation with GPU matching, with backpressure."""
    if depth < 1:
        raise ValueError("Positive prefetch depth required")
    queue = Queue(maxsize=depth)
    stop = ThreadEvent()
    sentinel = object()

    def send(value):
        while not stop.is_set():
            try:
                queue.put(value, timeout=.1)
                return
            except Full:
                pass

    def produce():
        try:
            for item in iterable:
                if stop.is_set():
                    break
                send((False, item))
        except BaseException as exc:
            send((True, exc))
        finally:
            send(sentinel)

    worker = Thread(target=produce, name="terasort-scb-prefetch", daemon=True)
    worker.start()

    def consume():
        while True:
            try:
                value = queue.get(timeout=.1)
            except Empty:
                if not worker.is_alive() and queue.empty():
                    raise RuntimeError("SCB prefetch stopped unexpectedly")
                continue
            if value is sentinel:
                return
            is_error, payload = value
            if is_error:
                raise payload
            yield payload

    try:
        yield consume()
    finally:
        stop.set()
        worker.join(timeout=5)


def _process_epoch(events, bank: LocalTemplateBank, writer: AssignmentWriter,
                   previous_stop: int, previous_event=None, next_event=None,
                   matcher=None):
    if not events:
        return previous_stop, 0, 0
    bin_samples = next(iter(bank.units.values())).rolling.bin_samples if bank.units else None
    first, last = events[0].sample_index, events[-1].sample_index
    if bin_samples is not None:
        bin_start = first // bin_samples * bin_samples
        if previous_stop < bin_start:
            for unit in bank.units.values():
                frozen = unit.rolling.snapshot()
                unit.rolling.commit(previous_stop, bin_start,
                                    coordinate_frame=frozen.coordinate_frame,
                                    expected_version=frozen.version)
            previous_stop = bin_start
    snapshots = bank.snapshots()
    rows, members = [], []
    updates: dict[int, list[np.ndarray]] = {}
    assigned_count = 0
    decisions = (matcher.match(events, snapshots) if matcher is not None else
                 [bank.assign(event, snapshots) for event in events])
    for i, event in enumerate(events):
        unit_id, score, runner, version, update = decisions[i]
        if unit_id >= 0:
            assigned_count += 1
            bank.units[unit_id].assigned_count += 1
        isolated = True
        for j in range(i - 1, -1, -1):
            if event.sample_index - events[j].sample_index > len(event.waveform):
                break
            if _same_patch(event, events[j]):
                isolated = False
                break
        if isolated:
            for j in range(i + 1, len(events)):
                if events[j].sample_index - event.sample_index > len(event.waveform):
                    break
                if _same_patch(event, events[j]):
                    isolated = False
                    break
        # At epoch edges, conservatively exclude any nearby event. A later
        # event beyond the immediate lookahead can still share this patch.
        if isolated and previous_event is not None:
            isolated = event.sample_index - previous_event.sample_index > len(event.waveform)
        if isolated and next_event is not None:
            isolated = next_event.sample_index - event.sample_index > len(event.waveform)
        if update is not None and isolated:
            updates.setdefault(unit_id, []).append(update)
        rows.append((event.sample_index, event.anchor_sample_index,
                     event.anchor_channel, event.amplitude_uv, event.snr,
                     unit_id, score, runner, version, 0, 0))
        members.append(event.candidate_rows)
    writer.append(rows, members)
    stop = last + 1
    for unit_id, unit in bank.units.items():
        frozen = snapshots[unit_id]
        accepted = updates.get(unit_id, [])
        mean = np.mean(accepted, axis=0).astype(np.float32) if accepted else None
        unit.rolling.commit(previous_stop, stop, mean_waveform=mean,
                            spike_count=len(accepted),
                            confidence=1. if accepted else 0.,
                            coordinate_frame=frozen.coordinate_frame,
                            expected_version=frozen.version)
    return stop, assigned_count, sum(map(len, updates.values()))


def sort_candidate_bank(source, output, *, templates=None, floor_snr=None,
                        radius_samples=2, max_epoch_events=2048,
                        slots_per_anchor=32, window_seconds=1800,
                        bin_seconds=300, backend="cpu",
                        start_sample=None, stop_sample=None):
    """Run a bounded first-pass sort from an SCB 0.2 shard.

    ``templates`` must be calibration templates in the same channel order and
    voltage preprocessing frame as the SCB waveforms. Omitting them measures
    consolidation and writes all events as unknown; it learns no units.
    """
    source = Path(source)
    scb = CandidateBank(source)
    if scb.waveform_spec is None:
        raise ValueError("SCB 0.2 waveform cache required")
    floor = scb.manifest["floor_snr"] if floor_snr is None else float(floor_snr)
    if not np.isfinite(floor) or floor < scb.manifest["floor_snr"]:
        raise ValueError("Requested floor is below the stored detection floor")
    if max_epoch_events < 1:
        raise ValueError("Positive epoch event cap required")
    if backend not in ("cpu", "cuda"):
        raise ValueError("Unknown matching backend")
    if (start_sample is not None and stop_sample is not None
            and start_sample >= stop_sample):
        raise ValueError("Empty or reversed source interval")
    rate = scb.manifest.get("sample_rate_hz")
    if not isinstance(rate, (float, int)) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("SCB sample rate required for rolling updates")
    first = int(start_sample if start_sample is not None else
                (scb.blocks[0]["start_sample"] if len(scb.blocks) else 0))
    seeded = (np.empty((0, scb.waveform_spec.n_samples, scb.n_channels), np.float32)
              if templates is None else templates)
    bank = LocalTemplateBank.from_dense_templates(
        seeded, scb.waveform_spec.channel_index,
        pre_samples=scb.waveform_spec.pre_samples,
        n_samples=scb.waveform_spec.n_samples, sample_rate_hz=rate,
        start_sample=first, window_seconds=window_seconds,
        bin_seconds=bin_seconds, slots_per_anchor=slots_per_anchor)
    if backend == "cuda":
        from .streaming_cuda import CudaLocalMatcher
        matcher = CudaLocalMatcher(bank)
    else:
        matcher = None
    config = {"floor_snr": floor, "radius_samples": radius_samples,
              "max_epoch_events": max_epoch_events,
              "slots_per_anchor": slots_per_anchor,
              "window_seconds": window_seconds, "bin_seconds": bin_seconds,
              "backend": backend,
              "start_sample": start_sample, "stop_sample": stop_sample,
              "seed_templates": str(Path(templates).resolve()) if isinstance(templates, (str, Path)) else None}
    writer = AssignmentWriter(output, source=source, config=config)
    previous_stop = first
    previous_event = None
    assigned = updated = 0
    try:
        events = iter_consolidated(scb, floor_snr=floor,
                                   radius_samples=radius_samples,
                                   start_sample=start_sample,
                                   stop_sample=stop_sample)
        epochs = _iter_epochs(events, bin_samples=round(bin_seconds * rate),
                              max_epoch_events=max_epoch_events)
        source_context = (_prefetched(epochs, depth=2) if backend == "cuda"
                          else nullcontext(epochs))
        with source_context as epoch_source:
            for epoch, next_event in epoch_source:
                previous_stop, a, u = _process_epoch(
                    epoch, bank, writer, previous_stop, previous_event, next_event,
                    matcher)
                assigned += a
                updated += u
                previous_event = epoch[-1]
        writer.finish(bank)
    except BaseException:
        writer.abort()
        raise
    return {"candidate_rows": writer.candidate_count, "consolidated_events": writer.count,
            "assigned_events": assigned, "template_updates": updated,
            "seed_units": len(bank.units), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental bounded SCB template assignment")
    parser.add_argument("--bank", required=True, help="Existing SCB 0.2 waveform bank")
    parser.add_argument("--output", required=True, help="New assignment shard; never overwritten")
    parser.add_argument("--templates", help="Calibration templates.npy, unit x time x channel")
    parser.add_argument("--floor-snr", type=float)
    parser.add_argument("--radius-samples", type=int, default=2)
    parser.add_argument("--max-epoch-events", type=int, default=2048)
    parser.add_argument("--slots-per-anchor", type=int, default=32)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--start-sample", type=int)
    parser.add_argument("--stop-sample", type=int)
    args = parser.parse_args(argv)
    result = sort_candidate_bank(args.bank, args.output, templates=args.templates,
        floor_snr=args.floor_snr, radius_samples=args.radius_samples,
        max_epoch_events=args.max_epoch_events,
        slots_per_anchor=args.slots_per_anchor, backend=args.backend,
        start_sample=args.start_sample, stop_sample=args.stop_sample)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
