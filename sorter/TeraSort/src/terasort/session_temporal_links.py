"""Bounded, ground-truth-free links between adjacent activity epochs.

This is an output-only identity view. It never changes spike rows, detector
events, fitted templates, or the source recording.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.spatial import cKDTree


RULE_VERSION = "adjacent_epoch_reciprocal_transition_v1"


def _atomic_json(path, payload):
    path = Path(path)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         prefix=path.name + ".", suffix=".partial",
                                         dir=path.parent, delete=False) as handle:
            name = handle.name
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


class TemporalIdentityLinker:
    """Keep two time bins and O(template count) best-link state."""

    def __init__(self, template_to_group, waveforms, channels, anchors,
                 positions, shanks, *, sample_rate, first_sample,
                 bin_seconds=5, min_crossing_spikes=8,
                 rate_crossing_ratio=2., max_anchor_distance_um=40.,
                 min_signed_cosine=.9, cosine_cache_size=32768):
        self.template_to_group = np.asarray(template_to_group, np.int32)
        self.waveforms = np.asarray(waveforms, np.float32)
        self.channels = np.asarray(channels, np.int32)
        self.anchors = np.asarray(anchors, np.int32)
        self.positions = np.asarray(positions, np.float64)
        self.shanks = np.asarray(shanks)
        self.bin_samples = round(float(sample_rate)*bin_seconds)
        self.first_sample = int(first_sample)
        self.min_crossing_spikes = int(min_crossing_spikes)
        self.rate_crossing_ratio = float(rate_crossing_ratio)
        self.max_anchor_distance_um = float(max_anchor_distance_um)
        self.min_signed_cosine = float(min_signed_cosine)
        self.cosine_cache_size = int(cosine_cache_size)
        n_templates = len(self.waveforms)
        if (self.waveforms.ndim != 3 or (self.waveforms.shape[1] < 21 or self.waveforms.shape[1] % 2 != 1)
                or self.channels.shape != (n_templates, self.waveforms.shape[2])
                or self.anchors.shape != (n_templates,)
                or self.template_to_group.shape != (n_templates,)
                or self.positions.ndim != 2 or self.positions.shape[1] != 2
                or self.shanks.shape != (len(self.positions),)
                or np.any(self.template_to_group < 0)
                or np.any(self.channels < -1)
                or np.any(self.channels >= len(self.positions))
                or np.any(self.anchors < 0)
                or np.any(self.anchors >= len(self.positions))
                or not np.isfinite(self.waveforms).all()
                or not np.isfinite(self.positions).all()
                or self.bin_samples < 1 or self.min_crossing_spikes < 1
                or self.rate_crossing_ratio <= 1
                or self.max_anchor_distance_um <= 0
                or not 0 < self.min_signed_cosine <= 1
                or self.cosine_cache_size < 1):
            raise ValueError("Invalid bounded temporal-link model or rule")
        self.n_groups = int(self.template_to_group.max()) + 1
        if set(np.unique(self.template_to_group)) != set(range(self.n_groups)):
            raise ValueError("Template group map must be compact")
        self.norms = np.linalg.norm(
            self.waveforms.reshape(n_templates, -1).astype(np.float64), axis=1)
        self.slots = [{int(channel): slot for slot, channel in enumerate(row)
                       if channel >= 0} for row in self.channels]
        self.trees = {}
        self.tree_channel_ids = {}
        for shank in np.unique(self.shanks):
            ids = np.flatnonzero(self.shanks == shank)
            self.tree_channel_ids[shank] = ids
            self.trees[shank] = cKDTree(self.positions[ids])
        self.neighbor_cache = {}
        self.cosine_cache = OrderedDict()
        self.current_bin = None
        self.current_groups = np.zeros(self.n_groups, np.int64)
        self.current_templates = np.zeros(n_templates, np.int64)
        self.previous = None
        self.best = {}
        self.observed_spikes = 0
        self.candidate_observations = 0
        self.last_sample = None

    def _neighbors(self, anchor):
        anchor = int(anchor)
        if anchor not in self.neighbor_cache:
            shank = self.shanks[anchor]
            indices = self.trees[shank].query_ball_point(
                self.positions[anchor], self.max_anchor_distance_um)
            self.neighbor_cache[anchor] = self.tree_channel_ids[shank][
                indices].tolist()
        return self.neighbor_cache[anchor]

    def _cosine(self, left, right):
        key = (min(int(left), int(right)), max(int(left), int(right)))
        cached = self.cosine_cache.get(key)
        if cached is not None:
            self.cosine_cache.move_to_end(key)
            return cached
        common = sorted(self.slots[left].keys() & self.slots[right].keys())
        similarity = 0.
        if common and self.norms[left]*self.norms[right] > 1e-9:
            li = np.fromiter((self.slots[left][c] for c in common), np.int32)
            ri = np.fromiter((self.slots[right][c] for c in common), np.int32)
            similarity = float(np.sum(
                self.waveforms[left, :, li].astype(np.float64) *
                self.waveforms[right, :, ri].astype(np.float64)) /
                (self.norms[left]*self.norms[right]))
        self.cosine_cache[key] = similarity
        if len(self.cosine_cache) > self.cosine_cache_size:
            self.cosine_cache.popitem(last=False)
        return similarity

    def _offer(self, edge):
        left, right = edge["left"], edge["right"]
        for group, other in ((left, right), (right, left)):
            old = self.best.get(group)
            if (old is None or edge["rank"] > old["rank"] or
                    (edge["rank"] == old["rank"] and
                     (other, edge["boundary_sample"]) <
                     (old["other"], old["boundary_sample"]))):
                self.best[group] = {**edge, "other": other}

    def _compare(self, before, after):
        if before[0] + 1 != after[0]:
            return
        ca, ta = before[1], before[2]
        cb, tb = after[1], after[2]
        falling = np.flatnonzero((ca >= self.min_crossing_spikes) &
                                 (cb*self.rate_crossing_ratio <= ca))
        rising = np.flatnonzero((cb >= self.min_crossing_spikes) &
                                (ca*self.rate_crossing_ratio <= cb))
        rising_by_anchor = defaultdict(list)
        for group in rising:
            rising_by_anchor[int(self.anchors[tb[group]])].append(int(group))
        for left in falling:
            left = int(left)
            left_template = int(ta[left])
            if left_template < 0:
                continue
            for anchor in self._neighbors(self.anchors[left_template]):
                for right in rising_by_anchor.get(anchor, ()):
                    if left == right:
                        continue
                    right_template = int(tb[right])
                    similarity = self._cosine(left_template, right_template)
                    if similarity < self.min_signed_cosine:
                        continue
                    ratio = min(ca[left]/max(1, cb[left]),
                                cb[right]/max(1, ca[right]))
                    rank = similarity + .02*np.log2(max(1., ratio))
                    a, b = sorted((left, right))
                    self._offer({"left": a, "right": b,
                                 "boundary_sample": self.first_sample +
                                     after[0]*self.bin_samples,
                                 "cosine": similarity,
                                 "anchor_distance_um": float(np.linalg.norm(
                                     self.positions[self.anchors[left_template]]-
                                     self.positions[self.anchors[right_template]])),
                                 "ratio": float(ratio), "rank": float(rank)})
                    self.candidate_observations += 1

    def _close_bin(self):
        if self.current_bin is None:
            return
        dominant = np.full(self.n_groups, -1, np.int32)
        for template in np.flatnonzero(self.current_templates):
            group = int(self.template_to_group[template])
            prior = dominant[group]
            if (prior < 0 or
                    self.current_templates[template] > self.current_templates[prior]):
                dominant[group] = int(template)
        finished = (self.current_bin, self.current_groups.copy(), dominant)
        if self.previous is not None:
            self._compare(self.previous, finished)
        self.previous = finished
        self.current_groups.fill(0)
        self.current_templates.fill(0)

    def observe(self, sample_indices, raw_template_ids):
        times = np.asarray(sample_indices, np.int64)
        raw = np.asarray(raw_template_ids, np.int64)
        if (times.ndim != 1 or raw.shape != times.shape or
                np.any(times < self.first_sample) or
                np.any(raw < 0) or np.any(raw >= len(self.waveforms)) or
                np.any(times[1:] < times[:-1]) or
                (len(times) and self.last_sample is not None and
                 times[0] < self.last_sample)):
            raise ValueError("Temporal-link spikes must be sorted, valid source rows")
        if not len(times):
            return
        self.last_sample = int(times[-1])
        bins = (times-self.first_sample)//self.bin_samples
        boundaries = np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(bins)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            bin_id = int(bins[start])
            if self.current_bin is None:
                self.current_bin = bin_id
            elif bin_id != self.current_bin:
                if bin_id < self.current_bin:
                    raise ValueError("Temporal-link shard order regressed")
                self._close_bin()
                self.current_bin = bin_id
            batch = raw[start:stop]
            self.current_templates += np.bincount(
                batch, minlength=len(self.waveforms))
            self.current_groups += np.bincount(
                self.template_to_group[batch], minlength=self.n_groups)
            self.observed_spikes += len(batch)

    def finish(self):
        self._close_bin()
        self.current_bin = None
        accepted = []
        parent = np.arange(self.n_groups, dtype=np.int32)
        for left, edge in sorted(self.best.items()):
            right = edge["other"]
            if left >= right or self.best.get(right, {}).get("other") != left:
                continue
            accepted.append({key: value for key, value in edge.items()
                             if key != "other"})
            parent[right] = left
        roots = np.asarray([parent[group] for group in range(self.n_groups)])
        _, groups = np.unique(roots, return_inverse=True)
        return groups.astype(np.int64), accepted

    def checkpoint(self):
        def summary(value):
            if value is None:
                return None
            index, counts, dominant = value
            active = np.flatnonzero(counts)
            return {"bin": int(index),
                    "groups": [[int(g), int(counts[g]), int(dominant[g])]
                               for g in active]}
        return {"version": 1, "current_bin": self.current_bin,
                "current_groups": [[int(i), int(v)] for i, v in enumerate(
                    self.current_groups) if v],
                "current_templates": [[int(i), int(v)] for i, v in enumerate(
                    self.current_templates) if v],
                "previous": summary(self.previous),
                "best": list(self.best.values()),
                "last_sample": self.last_sample,
                "observed_spikes": self.observed_spikes,
                "candidate_observations": self.candidate_observations}

    def restore(self, state):
        if state.get("version") != 1:
            raise ValueError("Unknown temporal-link checkpoint version")
        self.current_bin = state["current_bin"]
        self.current_groups.fill(0)
        self.current_templates.fill(0)
        for index, count in state["current_groups"]:
            self.current_groups[int(index)] = int(count)
        for index, count in state["current_templates"]:
            self.current_templates[int(index)] = int(count)
        prior = state["previous"]
        if prior is None:
            self.previous = None
        else:
            counts = np.zeros(self.n_groups, np.int64)
            dominant = np.full(self.n_groups, -1, np.int32)
            for group, count, template in prior["groups"]:
                counts[int(group)] = int(count)
                dominant[int(group)] = int(template)
            self.previous = (int(prior["bin"]), counts, dominant)
        self.best = {int(edge["left"]): edge for edge in state["best"]
                     if int(edge["other"]) == int(edge["right"])}
        self.best.update({int(edge["right"]): edge for edge in state["best"]
                          if int(edge["other"]) == int(edge["left"])})
        self.observed_spikes = int(state["observed_spikes"])
        self.candidate_observations = int(state["candidate_observations"])
        self.last_sample = state["last_sample"]


def build_temporal_link_view(source_root, output_root, *, probe_id,
                             first_sample=None, stop_sample=None,
                             resume=False, row_batch=65536):
    """Create a resumable sidecar map from immutable frozen-template shards."""
    source_root, output_root = Path(source_root), Path(output_root)
    source_run = json.loads((source_root / "run.json").read_text(encoding="utf-8"))
    config = source_run["config"]
    if (not config.get("freeze_templates") or config.get("novelty") != "off"
            or "template_linking" not in config):
        raise ValueError("Temporal links require frozen templates and an existing link map")
    shards = sorted((source_root / probe_id).glob("*.h5"))
    if not shards or row_batch < 1:
        raise ValueError("No completed probe shards or invalid row batch")
    with h5py.File(shards[0], "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError("Incomplete source shard")
        day_id = str(handle.attrs["day_id"])
        default_first = int(handle.attrs["start_sample"])
        template_to_group = handle["template_linking/template_to_unit"][:]
        model = handle["model_after"]
        waveforms = model["waveforms"][:]
        channels = model["channels"][:]
        anchors = model["anchors"][:]
        positions = handle["channel_positions_um"][:]
        shanks = handle["channel_shank"][:]
        model_version = int(model.attrs["version"])
    first = default_first if first_sample is None else int(first_sample)
    stop = None if stop_sample is None else int(stop_sample)
    if first < default_first or (stop is not None and stop <= first):
        raise ValueError("Invalid temporal-link source interval")
    model_hash = hashlib.sha256()
    for array in (template_to_group, waveforms, channels, anchors, positions, shanks):
        model_hash.update(np.asarray(array).tobytes())
    source_shards = []
    for path in shards:
        with h5py.File(path, "r") as handle:
            if (not handle.attrs.get("complete", False)
                    or str(handle.attrs["day_id"]) != day_id
                    or handle.attrs["config_sha256"] != source_run["config_sha256"]
                    or int(handle["model_after"].attrs["version"]) != model_version
                    or not np.array_equal(
                        handle["template_linking/template_to_unit"][:],
                        template_to_group)):
                raise ValueError("Temporal-link source shards must share a frozen day/model/map")
            start, end = int(handle.attrs["start_sample"]), int(handle.attrs["stop_sample"])
        if end <= first or (stop is not None and start >= stop):
            continue
        stat = path.stat()
        source_shards.append({"path": str(path.resolve()),
                              "start": start, "stop": end,
                              "size": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns})
    if not source_shards:
        raise ValueError("No source shards overlap the selected interval")
    if stop is None:
        stop = max(item["stop"] for item in source_shards)
    sample_rate = None
    for probe in source_run["source"]["probes"]:
        if probe["probe_id"] == probe_id:
            sample_rate = float(probe["sample_rate_hz"])
            break
    if sample_rate is None:
        raise ValueError("Probe is missing from the pinned source record")
    identity = {"rule_version": RULE_VERSION, "probe_id": probe_id,
                "day_id": day_id, "first_sample": first,
                "stop_sample": stop, "source_config_sha256": source_run["config_sha256"],
                "source_model_sha256": model_hash.hexdigest(),
                "source_shards": source_shards}
    if output_root.exists() and not resume:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=resume)
    checkpoint_path = output_root / "checkpoint.json"
    final_path = output_root / "temporal_links.json"
    linker = TemporalIdentityLinker(
        template_to_group, waveforms, channels, anchors, positions, shanks,
        sample_rate=sample_rate, first_sample=first)
    completed = 0
    if resume:
        if not checkpoint_path.exists():
            raise ValueError("No matching temporal-link checkpoint to resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["identity"] != identity:
            raise ValueError("Temporal-link source/configuration changed since checkpoint")
        completed = int(checkpoint["completed_shards"])
        if not 0 <= completed <= len(source_shards):
            raise ValueError("Invalid completed shard count")
        linker.restore(checkpoint["state"])
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            if completed != len(source_shards) or final.get("identity") != identity:
                raise ValueError("Completed temporal-link view has inconsistent state")
            return {"output": str(final_path),
                    "accepted_edges": len(final["accepted_edges"]),
                    "observed_spikes": final["observed_spikes"],
                    "output_units": len(set(final["base_group_to_temporal_unit"]))}
    for shard_number, item in enumerate(source_shards[completed:], start=completed):
        with h5py.File(item["path"], "r") as handle:
            spike_rows = handle["spikes"]
            export_mask = handle.get("spike_export_mask")
            previous_time = None
            for offset in range(0, len(spike_rows), row_batch):
                end = min(offset+row_batch, len(spike_rows))
                spikes = spike_rows[offset:end]
                if export_mask is not None:
                    spikes = spikes[export_mask[offset:end].astype(bool)]
                if not len(spikes):
                    continue
                times = spikes["sample_index"]
                if previous_time is not None and times[0] < previous_time:
                    raise ValueError("Source spikes are not sorted")
                previous_time = int(times[-1])
                selected = (times >= first) & (times < stop)
                linker.observe(times[selected], spikes["unit_id"][selected])
        _atomic_json(checkpoint_path, {"identity": identity,
                                       "completed_shards": shard_number+1,
                                       "state": linker.checkpoint()})
    groups, edges = linker.finish()
    payload = {"schema_version": 1, "rule_version": RULE_VERSION,
               "identity": identity, "accepted_edges": edges,
               "base_group_to_temporal_unit": groups.tolist(),
               "template_to_temporal_unit": groups[template_to_group].tolist(),
               "candidate_observations": linker.candidate_observations,
               "observed_spikes": linker.observed_spikes,
               "meaning": "output-only uncertain identity view; original shards are unchanged"}
    _atomic_json(final_path, payload)
    return {"output": str(final_path), "accepted_edges": len(edges),
            "observed_spikes": linker.observed_spikes,
            "output_units": int(len(np.unique(groups)))}
