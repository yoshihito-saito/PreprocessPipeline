"""Immutable session shards with a partial-coverage waveform cache and checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .candidates.bank import EVENT_DTYPE
from .candidates.waveforms import WaveformSpec, encode_waveforms, extract_waveforms
from .session_models import normalize_template_identity_map


CANDIDATE_DTYPE = np.dtype([
    ("sample_index", "<i8"), ("channel_index", "<u4"),
    ("amplitude_uv", "<f4"), ("snr", "<f4"),
    ("qc_flags", "u1"), ("unit_id", "<i8"),
    ("score", "<f4"), ("runner_up", "<f4"),
    ("fitted_amplitude", "<f4"), ("residual_gain", "<f4"),
    ("pass_index", "<i2"),
])
SPIKE_DTYPE = np.dtype([
    ("sample_index", "<i8"), ("channel_index", "<u4"),
    ("unit_id", "<i8"), ("score", "<f4"), ("runner_up", "<f4"),
    ("fitted_amplitude", "<f4"), ("residual_gain", "<f4"),
    ("pass_index", "<i2"), ("candidate_row", "<i8"),
    ("model_version", "<i4"),
])
QC_DTYPE = np.dtype([
    ("start_sample", "<i8"), ("stop_sample", "<i8"),
    ("interval_bad", "u1"), ("source_bytes_read", "<i8"),
])


def _dataset(handle, name, dtype, chunk=65_536):
    return handle.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype,
        chunks=(chunk,), compression="lzf", shuffle=True)


def _append(dataset, values):
    values = np.asarray(values, dtype=dataset.dtype)
    first = len(dataset)
    dataset.resize((first + len(values),) + dataset.shape[1:])
    if len(values):
        dataset[first:] = values
    return first


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def select_waveform_cache_rows(candidates, budget, *, mode="legacy",
                               template_link_map=None):
    """Choose bounded candidate rows without changing any assignment rows."""
    if mode not in ("legacy", "balanced", "recall") or budget < 0:
        raise ValueError("Invalid waveform cache mode or budget")
    budget = min(int(budget), len(candidates))
    if budget == 0:
        return np.empty(0, np.int64)
    clean = np.flatnonzero(candidates["qc_flags"] == 0)
    uncertain = clean[(candidates["unit_id"][clean] < 0) &
                      (candidates["snr"][clean] >=
                       (4.5 if mode == "recall" else 5.))]
    if mode == "recall":
        # Spread the fixed payload budget over weak unknown events instead of
        # spending the unknown quota on only the largest peaks. This changes
        # cached training evidence, never candidate metadata or assignments.
        def hashed_sample(pool, slots):
            slots = min(slots, len(pool))
            if not slots:
                return np.empty(0, np.int64)
            keys = (candidates["sample_index"][pool].astype(np.uint64) *
                    np.uint64(11400714819323198485) ^
                    candidates["channel_index"][pool].astype(np.uint64) *
                    np.uint64(7046029254386353131))
            return pool[np.argpartition(keys, slots-1)[:slots]]
        weak = uncertain[(candidates["snr"][uncertain] >= 4.5) &
                         (candidates["snr"][uncertain] < 8.)]
        high = uncertain[candidates["snr"][uncertain] >= 8.]
        priority = np.r_[hashed_sample(weak, int(budget*.3)),
                         hashed_sample(high, int(budget*.1))]
    else:
        uncertain_slots = budget//2 if mode == "legacy" else int(budget*.4)
        priority = uncertain[np.argsort(-candidates["snr"][uncertain],
                                        kind="stable")[:uncertain_slots]]
    selected = priority
    if mode in ("balanced", "recall"):
        available = np.setdiff1d(clean, selected, assume_unique=False)
        high = available[(candidates["unit_id"][available] >= 0) &
                         (candidates["score"][available] >= .85) &
                         ((candidates["score"][available] -
                           candidates["runner_up"][available]) >= .1) &
                         (candidates["fitted_amplitude"][available] >= .5) &
                         (candidates["fitted_amplitude"][available] <= 2.) &
                         (candidates["pass_index"][available] == 0)]
        high_slots = min(int(budget*.4), len(high), budget-len(selected))
        if high_slots:
            hashes = (candidates["sample_index"][high].astype(np.uint64) *
                      np.uint64(11400714819323198485) ^
                      candidates["channel_index"][high].astype(np.uint64))
            groups = candidates["unit_id"][high]
            if template_link_map is not None:
                groups = np.asarray(template_link_map)[groups]
            # One deterministic representative per active identity first.
            order = np.lexsort((hashes, groups))
            first = np.r_[True, groups[order][1:] != groups[order][:-1]]
            representatives = order[first]
            representatives = representatives[
                np.argsort(hashes[representatives], kind="stable")[:high_slots]]
            chosen = representatives
            remaining_slots = high_slots-len(chosen)
            if remaining_slots:
                remaining = np.setdiff1d(np.arange(len(high)), chosen,
                                         assume_unique=False)
                extra = remaining[np.argsort(hashes[remaining],
                                             kind="stable")[:remaining_slots]]
                chosen = np.r_[chosen, extra]
            selected = np.r_[selected, high[chosen]]
    remaining = np.setdiff1d(clean, selected, assume_unique=False)
    keep = min(budget-len(selected), len(remaining))
    if keep:
        keys = (candidates["sample_index"][remaining].astype(np.uint64) *
                np.uint64(11400714819323198485) ^
                candidates["channel_index"][remaining].astype(np.uint64))
        sampled = remaining[np.argpartition(keys, keep-1)[:keep]]
        selected = np.r_[selected, sampled]
    return np.sort(selected).astype(np.int64, copy=False)


class RunJournal:
    """A run directory is new, or a resume with the exact same source/config."""

    def __init__(self, root, source_record, config, *, resume=False):
        self.root = Path(root).expanduser().resolve()
        self.record = {"schema_version": 1, "source": source_record,
                       "config": config, "config_sha256": config_digest(config)}
        path = self.root / "run.json"
        if resume:
            if not path.is_file() or json.loads(path.read_text()) != self.record:
                raise ValueError("Resume source or configuration differs from run.json")
        else:
            if self.root.exists():
                raise FileExistsError(self.root)
            self.root.mkdir(parents=True)
            partial = path.with_suffix(".json.partial")
            partial.write_text(json.dumps(self.record, indent=2) + "\n",
                               encoding="utf-8")
            os.replace(partial, path)

    def shard_path(self, probe_id, start_sample, stop_sample):
        return self.root / probe_id / f"{start_sample:016d}-{stop_sample:016d}.h5"

    def completed(self, path):
        path = Path(path)
        if not path.is_file():
            return False
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete", False):
                raise ValueError(f"Published shard is incomplete: {path}")
            if handle.attrs.get("config_sha256") != self.record["config_sha256"]:
                raise ValueError(f"Shard configuration differs: {path}")
        return True

    def latest_model(self, probe_id, before_sample):
        folder = self.root / probe_id
        if not folder.is_dir():
            return None
        found = []
        for path in folder.glob("*.h5"):
            try:
                start, stop = map(int, path.stem.split("-"))
            except ValueError:
                raise ValueError(f"Unknown shard filename: {path}")
            if stop <= before_sample:
                found.append((stop, path))
        if not found:
            return None
        stop, path = max(found)
        self.completed(path)
        with h5py.File(path, "r") as handle:
            group = handle["model_after"]
            return stop, (group["waveforms"][:], group["channels"][:],
                          group["anchors"][:], group["assigned"][:],
                          int(group.attrs["version"]))


class ShardWriter:
    """SCB 0.3 candidate metadata, sparse waveform cache, TSA-like spikes."""

    def __init__(self, path, *, probe, day_id, start_sample, stop_sample,
                 config_sha256, waveform_scale_uv=1.,
                 template_link_map=None, template_link_metadata=None,
                 export_amplitude_min=None, phy_feature_basis=None, phy_feature_width=None, nt=61):
        from .session_models import validate_nt
        self.nt = validate_nt(nt)
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial, "x")
        h = self.handle
        h.attrs["complete"] = False
        h.attrs["format"] = "TeraSortSessionShard"
        h.attrs["version"] = 1
        h.attrs["candidate_format"] = "SCB0.3"
        h.attrs["candidate_coverage"] = "all threshold detections; waveforms selective"
        h.attrs["probe_id"] = probe.probe_id
        h.attrs["day_id"] = day_id
        h.attrs["start_sample"] = int(start_sample)
        h.attrs["stop_sample"] = int(stop_sample)
        h.attrs["config_sha256"] = config_sha256
        h.attrs["waveform_scale_uv"] = float(waveform_scale_uv)
        if template_link_map is not None:
            mapping = np.asarray(template_link_map, dtype=np.int64)
            if mapping.ndim != 1 or np.any(mapping < 0):
                raise ValueError("Invalid template link map")
            group = h.create_group("template_linking")
            group.create_dataset("template_to_unit", data=mapping, dtype="<i8",
                                 maxshape=(None,))
            group.attrs["metadata_json"] = json.dumps(
                template_link_metadata or {}, sort_keys=True)
        self.candidates = _dataset(h, "candidates", CANDIDATE_DTYPE)
        self.spikes = _dataset(h, "spikes", SPIKE_DTYPE)
        self.export_amplitude_min = export_amplitude_min
        if export_amplitude_min is not None:
            h.attrs["spike_export_filter_json"] = json.dumps({
                "algorithm": "matched_amplitude_scale_min_v1",
                "minimum": float(export_amplitude_min),
                "scope": "output_view_only",
            }, sort_keys=True)
        self.qc = _dataset(h, "qc", QC_DTYPE, chunk=1024)
        nchan = probe.n_channels
        self.noise = h.create_dataset("qc_noise_uv", (0, nchan), maxshape=(None, nchan),
                                      dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.flags = h.create_dataset("qc_channel_flags", (0, nchan),
                                      maxshape=(None, nchan), dtype="u1",
                                      chunks=(1, nchan), compression="lzf")
        self.saturated = h.create_dataset(
            "qc_saturated_fraction", (0, nchan), maxshape=(None, nchan),
            dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.flatline = h.create_dataset(
            "qc_flatline_fraction", (0, nchan), maxshape=(None, nchan),
            dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.zero = h.create_dataset(
            "qc_zero_fraction", (0, nchan), maxshape=(None, nchan),
            dtype="<f4", chunks=(1, nchan), compression="lzf")
        self.position = h.create_dataset("channel_positions_um",
                                         data=probe.geometry.astype(np.float32))
        self.shank = h.create_dataset("channel_shank", data=probe.shank.astype(np.int32))
        self.cache_rows = _dataset(h, "waveform_candidate_row", "<i8")
        self.template_link_map = (None if template_link_map is None else
                                  np.asarray(template_link_map, dtype=np.int64))
        self.cache_spec = None
        self.cache_data = None
        self.cached_bytes = 0
        self.waveform_scale_uv = waveform_scale_uv
        self.phy_features = None
        self.phy_feature_basis = phy_feature_basis
        if phy_feature_basis is not None:
            from .session_phy_capture import validate_basis
            validate_basis(phy_feature_basis)
            if not isinstance(phy_feature_width, int) or not 1 <= phy_feature_width <= 16:
                raise ValueError("Invalid Phy feature width")
            h.create_dataset("phy_feature_basis", data=phy_feature_basis)
            h.attrs["phy_feature_algorithm"] = "resident_preprocessed_temporal_pca3_v1"
            self.phy_features = h.create_dataset("spike_features", (0, 3, phy_feature_width),
                maxshape=(None, 3, phy_feature_width), dtype="<f4",
                chunks=(256, 3, phy_feature_width), compression="lzf", shuffle=True)

    def _ensure_waveforms(self, channel_map):
        if self.cache_data is not None:
            return
        self.cache_spec = WaveformSpec(channel_map, self.nt//2, self.nt//2,
                                       self.waveform_scale_uv, "int16")
        width = self.cache_spec.width
        self.handle.create_dataset("waveform_channel_index",
                                   data=self.cache_spec.channel_index)
        self.cache_data = self.handle.create_dataset(
            "waveform_data", shape=(0, self.nt, width),
            maxshape=(None, self.nt, width), dtype="<i2",
            chunks=(max(1, min(512, 1_048_576 // (self.nt * width * 2))), self.nt, width),
            compression="lzf", shuffle=True)
        self.cache_start = _dataset(self.handle, "waveform_valid_start", "<i4")
        self.cache_stop = _dataset(self.handle, "waveform_valid_stop", "<i4")

    def append_core(self, *, core, voltage_uv, quality, threshold_events,
                    matches, channel_map, cache_fraction=.05,
                    model_version=0, cache_mode="legacy", model_channels=None):
        """Append one core. Every threshold event gets metadata, no silent cap."""
        first = len(self.candidates)
        # A residual redetection may rediscover the same peak. Keep its first
        # observation, while retaining peaks that appeared only after fitting.
        events = np.asarray(threshold_events, dtype=np.float64).reshape(-1, 3)
        times = events[:, 0].astype(np.int64)
        contacts = events[:, 1].astype(np.int64)
        keep = ((times >= core.core_start - core.data_start) &
                (times < core.core_stop - core.data_start))
        events, times, contacts = events[keep], times[keep], contacts[keep]
        keys = times * voltage_uv.shape[1] + contacts
        _, first_observation = np.unique(keys, return_index=True)
        first_observation.sort()  # Preserve first-observation candidate order.
        times, contacts = times[first_observation], contacts[first_observation]
        keys = keys[first_observation]
        candidates = np.empty(len(times), CANDIDATE_DTYPE)
        candidates["unit_id"] = -1
        candidates["score"] = -1.
        candidates["runner_up"] = -1.
        candidates["fitted_amplitude"] = 0.
        candidates["residual_gain"] = 0.
        candidates["pass_index"] = -1
        # Add the 64-bit source offset only after converting local coordinates
        # to integers; long-session sample positions must never enter a float.
        candidates["sample_index"] = times + core.data_start
        candidates["channel_index"] = contacts
        candidates["amplitude_uv"] = voltage_uv[times, contacts]
        candidates["snr"] = events[first_observation, 2]
        candidates["qc_flags"] = quality.flags[contacts]
        match_keys = np.fromiter((
            (match.candidate_sample if match.candidate_sample is not None
             else match.source_sample) * voltage_uv.shape[1] + match.channel
            for match in matches), dtype=np.int64, count=len(matches))
        rows = np.full(len(matches), -1, np.int64)
        if len(keys):
            order = np.argsort(keys, kind="stable")
            positions = np.searchsorted(keys[order], match_keys)
            valid = positions < len(keys)
            valid[valid] &= keys[order[positions[valid]]] == match_keys[valid]
            rows[valid] = order[positions[valid]]
        spikes = np.empty(len(matches), SPIKE_DTYPE)
        for i, match in enumerate(matches):
            absolute = core.data_start + match.source_sample
            row = rows[i]
            spikes[i] = (absolute, match.channel, match.unit, match.score,
                         match.runner_up, match.amplitude, match.residual_gain,
                         match.pass_index, first + row if row >= 0 else -1,
                         model_version)
            if row >= 0:
                candidates[row]["unit_id"] = match.unit
                candidates[row]["score"] = match.score
                candidates[row]["runner_up"] = match.runner_up
                candidates[row]["fitted_amplitude"] = match.amplitude
                candidates[row]["residual_gain"] = match.residual_gain
                candidates[row]["pass_index"] = match.pass_index
        _append(self.candidates, candidates)
        _append(self.spikes, spikes)
        if self.phy_features is not None:
            from .session_phy_capture import project_signal
            if model_channels is None:
                raise ValueError("Measured Phy features require template channel indices")
            for values in project_signal(voltage_uv, spikes["sample_index"]-core.data_start,
                    np.asarray(model_channels)[:, :self.phy_features.shape[2]],
                    self.phy_feature_basis, units=spikes["unit_id"]):
                _append(self.phy_features, values)
        _append(self.qc, [(core.core_start, core.core_stop,
                           int(quality.interval_bad), core.source_bytes_read)])
        n = len(self.qc)
        self.noise.resize((n, len(quality.noise_uv)))
        self.noise[n-1] = quality.noise_uv
        self.flags.resize((n, len(quality.flags)))
        self.flags[n-1] = quality.flags
        for dataset, values in (
                (self.saturated, quality.saturated_fraction),
                (self.flatline, quality.flatline_fraction),
                (self.zero, quality.zero_fraction)):
            dataset.resize((n, len(values)))
            dataset[n-1] = values

        # The waveform payload budget is charged against raw core bytes.
        # The opt-in balanced mode reserves samples for confident assigned
        # identities, enabling later bounded empirical template sketches.
        self._ensure_waveforms(channel_map)
        row_bytes = self.nt * self.cache_spec.width * 2 + 16
        raw_core_bytes = (core.core_stop-core.core_start) * len(quality.flags) * 2
        budget = min(len(candidates),
                     int(cache_fraction * raw_core_bytes) // row_bytes)
        if budget:
            selected = select_waveform_cache_rows(
                candidates, budget, mode=cache_mode,
                template_link_map=self.template_link_map)
            if len(selected):
                event_rows = np.empty(len(selected), EVENT_DTYPE)
                for name in EVENT_DTYPE.names:
                    event_rows[name] = candidates[name][selected]
                values, starts, stops = extract_waveforms(
                    event_rows, voltage_uv, core.data_start, self.cache_spec,
                    core.data_start, core.data_start + len(voltage_uv))
                encoded = encode_waveforms(values, self.cache_spec)
                _append(self.cache_rows, first + selected)
                old = len(self.cache_data)
                self.cache_data.resize((old + len(selected), self.nt, self.cache_spec.width))
                self.cache_data[old:] = encoded
                _append(self.cache_start, starts)
                _append(self.cache_stop, stops)
                self.cached_bytes += len(selected) * row_bytes
        self.handle.flush()

    def _write_export_mask(self):
        """Build the optional view mask in bounded blocks after dense sorting."""
        if self.export_amplitude_min is None:
            return
        count = len(self.spikes)
        chunk = max(1, min(65_536, count))
        mask = self.handle.create_dataset(
            "spike_export_mask", shape=(count,), maxshape=(None,), dtype="u1",
            chunks=(chunk,), compression="lzf", shuffle=True)
        block_rows = 1_048_576
        for first in range(0, count, block_rows):
            stop = min(count, first + block_rows)
            amplitudes = self.spikes[first:stop]["fitted_amplitude"]
            mask[first:stop] = (amplitudes >= self.export_amplitude_min).astype(np.uint8)

    def finish(self, models, *, telemetry=None, adaptive_shift=None):
        if self.phy_features is not None and len(self.phy_features) != len(self.spikes):
            raise ValueError("Measured Phy feature rows do not match saved spikes")
        if self.template_link_map is not None:
            # Enrollment occurs at shard boundaries. Publish the new identities
            # with the checkpoint so the next shard/cache can address them.
            mapping = normalize_template_identity_map(
                self.template_link_map, len(models.waveforms))
            if len(mapping) != len(self.template_link_map):
                dataset = self.handle['template_linking/template_to_unit']
                dataset.resize((len(mapping),))
                dataset[:] = mapping
                self.template_link_map = mapping
        self._write_export_mask()
        group = self.handle.create_group("model_after")
        group.create_dataset("waveforms", data=models.waveforms,
                             compression="lzf")
        group.create_dataset("channels", data=models.channels)
        group.create_dataset("anchors", data=models.anchors)
        group.create_dataset("assigned", data=models.assigned)
        group.attrs["version"] = models.version
        if adaptive_shift is not None:
            adaptive_shift.save(self.handle.create_group("adaptive_shift"))
        if telemetry is not None:
            self.handle.attrs["telemetry_json"] = json.dumps(
                telemetry, sort_keys=True)
        self.handle.attrs["complete"] = True
        self.handle.flush()
        self.handle.close()
        self.handle = None
        os.replace(self.partial, self.path)

    def abort(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
