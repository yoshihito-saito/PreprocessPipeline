"""Experimental bounded motion-aligned identity links between frozen shards.

The view changes only output labels. Candidate/spike rows and raw sources stay
immutable. Each adjacent pair is processed independently in a bounded frame.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


RULE_VERSION = "adjacent_shard_motion_sketch_mutual_v3"
BIN_SECONDS = 5
MIN_SPIKES = 3
MIN_CONTACTS = 2
MAX_ANCHOR_DISTANCE_UM = 60.
MIN_COSINE = .85
MIN_SELF_IMPROVEMENT = .03


def _atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         prefix=path.name + ".", suffix=".partial",
                                         dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _source_identity(path):
    stat = path.stat()
    with h5py.File(path, "r") as h:
        if not h.attrs.get("complete", False):
            raise ValueError(f"Incomplete source shard: {path}")
        return {"path": str(path.resolve()),
                "start_sample": int(h.attrs["start_sample"]),
                "stop_sample": int(h.attrs["stop_sample"]),
                "day_id": str(h.attrs["day_id"]),
                "config_sha256": str(h.attrs["config_sha256"]),
                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _pair_histograms(paths, positions, *, sample_rate, row_batch):
    """Read assigned candidate metadata only; histogram memory is fixed by pair length."""
    depth = np.unique(positions[:, 1])
    if len(depth) < 2 or not np.allclose(np.diff(depth), np.diff(depth)[0]):
        raise ValueError("Motion view requires uniformly spaced contact depths")
    edges = np.geomspace(8., 350., 25)
    first = int(paths[0]["start_sample"])
    last = int(paths[-1]["stop_sample"])
    samples_per_bin = round(sample_rate * BIN_SECONDS)
    bins = (last-first+samples_per_bin-1)//samples_per_bin
    images = np.zeros((bins, len(depth), len(edges)-1), np.float64)
    observations = np.zeros(bins, np.int64)
    for item in paths:
        with h5py.File(item["path"], "r") as h:
            source = h["candidates"]
            for start in range(0, len(source), row_batch):
                rows = source[start:start+row_batch]
                valid = ((rows["unit_id"] >= 0) & (rows["qc_flags"] == 0) &
                         (rows["score"] >= .85))
                rows = rows[valid]
                if not len(rows):
                    continue
                bi = (rows["sample_index"]-first)//samples_per_bin
                di = np.searchsorted(depth, positions[rows["channel_index"], 1])
                ai = np.searchsorted(edges, abs(rows["amplitude_uv"]))-1
                inside = ((bi >= 0) & (bi < bins) & (ai >= 0) &
                          (ai < len(edges)-1))
                np.add.at(images, (bi[inside], di[inside], ai[inside]), 1)
                np.add.at(observations, bi[inside], 1)
    return np.asarray([gaussian_filter(image, (.7, .7)) for image in images]), observations, depth


def _estimate_motion(images, observations, depth_step_um):
    """Correlate each 5-second image to the pair's first bin with sub-bin fit."""
    reference = images[0]
    estimates = []
    quality = []
    for image, count in zip(images, observations):
        correlations = []
        for offset in range(-5, 6):
            if offset < 0:
                a, b = reference[-offset:], image[:offset]
            elif offset > 0:
                a, b = reference[:-offset], image[offset:]
            else:
                a, b = reference, image
            a, b = a-a.mean(), b-b.mean()
            denominator = np.linalg.norm(a)*np.linalg.norm(b)
            correlations.append(float(np.sum(a*b)/denominator)
                                if denominator > 1e-9 else 0.)
        best = int(np.argmax(correlations))
        fractional = 0.
        if 0 < best < len(correlations)-1:
            curvature = (correlations[best-1] - 2*correlations[best] +
                         correlations[best+1])
            if curvature < -1e-9:
                fractional = float(np.clip(
                    .5*(correlations[best-1]-correlations[best+1])/curvature,
                    -1., 1.))
        estimates.append((best-5+fractional)*depth_step_um)
        quality.append(float(correlations[best]))
        if count < 100 or best in (0, 10) or correlations[best] < .35:
            raise ValueError("Insufficient or boundary-limited motion evidence")
    return np.asarray(estimates), quality


def _sketch(path, mapping, positions, shifts, *, first_sample, sample_rate,
            row_batch):
    """Accumulate O(group × contact × waveform) sketches from cached rows."""
    groups = int(mapping.max())+1
    total = None
    count = np.zeros((groups, len(positions)), np.int32)
    observations = np.zeros(groups, np.int32)
    tree = cKDTree(positions)
    samples_per_bin = round(sample_rate*BIN_SECONDS)
    with h5py.File(path, "r") as h:
        cache_rows = h["waveform_candidate_row"]
        waveforms = h["waveform_data"]
        total = np.zeros((groups, waveforms.shape[1], len(positions)), np.float32)
        contacts = h["waveform_channel_index"][:]
        candidate_rows = h["candidates"]
        for start in range(0, len(cache_rows), row_batch):
            rows = cache_rows[start:start+row_batch]
            events = candidate_rows[rows]
            waves = waveforms[start:start+len(rows)]
            selected = np.flatnonzero(
                (events["unit_id"] >= 0) & (events["score"] >= .85) &
                ((events["score"]-events["runner_up"]) >= .1) &
                (events["fitted_amplitude"] >= .5) &
                (events["fitted_amplitude"] <= 2) &
                (events["pass_index"] == 0))
            for index in selected:
                event = events[index]
                group = int(mapping[event["unit_id"]])
                channels = contacts[event["channel_index"]]
                slots = np.flatnonzero(channels >= 0)
                if not len(slots):
                    continue
                bin_id = min(len(shifts)-1, max(0, int(
                    (event["sample_index"]-first_sample)//samples_per_bin)))
                shifted = positions[channels[slots]].copy()
                shifted[:, 1] -= shifts[bin_id]
                distances, target = tree.query(shifted, k=1)
                valid = distances <= 25.
                if not np.any(valid):
                    continue
                np.add.at(total[group], (np.arange(waveforms.shape[1])[:, None],
                                         target[valid][None, :]),
                          waves[index, :, slots[valid]].T)
                np.add.at(count[group], target[valid], 1)
                observations[group] += 1
    return total, count, observations


def _peak_positions(sketch):
    sums, counts, _ = sketch
    peaks = np.full(len(counts), -1, np.int32)
    for group in range(len(counts)):
        valid = counts[group] >= 2
        if not np.any(valid):
            continue
        mean = sums[group, :, valid]/counts[group, valid, None]
        peaks[group] = np.flatnonzero(valid)[np.argmax(np.sum(mean*mean, axis=1))]
    return peaks


def _waveform_similarity(left, right):
    result = -1.
    for lag in range(-3, 4):
        if lag < 0:
            a, b = left[-lag:], right[:lag]
        elif lag > 0:
            a, b = left[:-lag], right[lag:]
        else:
            a, b = left, right
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        if denominator > 1e-9:
            result = max(result, float(np.sum(a*b)/denominator))
    return result


def select_motion_links(early, late, positions):
    """Mutual-best sparse links exceeding both identities' self continuity."""
    sum_a, count_a, n_a = early
    sum_b, count_b, n_b = late
    peak_a, peak_b = _peak_positions(early), _peak_positions(late)
    eligible_b = np.flatnonzero((n_b >= MIN_SPIKES) & (peak_b >= 0))
    if not len(eligible_b):
        return [], 0
    tree = cKDTree(positions[peak_b[eligible_b]])
    pairs = []
    for left in np.flatnonzero((n_a >= MIN_SPIKES) & (peak_a >= 0)):
        neighbors = tree.query_ball_point(positions[peak_a[left]],
                                          MAX_ANCHOR_DISTANCE_UM)
        for neighbor in neighbors:
            right = int(eligible_b[neighbor])
            common = (count_a[left] >= 2) & (count_b[right] >= 2)
            if np.count_nonzero(common) < MIN_CONTACTS:
                continue
            a = sum_a[left][:, common]/count_a[left, common]
            b = sum_b[right][:, common]/count_b[right, common]
            score = _waveform_similarity(a, b)
            distance = float(np.linalg.norm(positions[peak_a[left]]-
                                            positions[peak_b[right]]))
            pairs.append((int(left), right, score, distance))
    self_score = {a:score for a,b,score,_ in pairs if a == b}
    best_a, best_b = {}, {}
    for a,b,score,distance in pairs:
        ka = (score, -distance, -b)
        kb = (score, -distance, -a)
        if a not in best_a or ka > best_a[a][0]:
            best_a[a] = (ka,b)
        if b not in best_b or kb > best_b[b][0]:
            best_b[b] = (kb,a)
    pair_by_ids = {(a,b):(score,distance) for a,b,score,distance in pairs}
    accepted = []
    for a,(_,b) in sorted(best_a.items()):
        if a == b or best_b.get(b,(None,None))[1] != a:
            continue
        score,distance = pair_by_ids[a,b]
        if (score < MIN_COSINE or
                score < max(self_score.get(a,-1.),self_score.get(b,-1.))+
                MIN_SELF_IMPROVEMENT):
            continue
        accepted.append({"early_group":a,"late_group":b,
                         "cosine":score,"distance_um":distance,
                         "early_self_cosine":self_score.get(a,-1.),
                         "late_self_cosine":self_score.get(b,-1.)})
    return accepted, len(pairs)


def build_motion_link_view(source_root, output_root, *, probe_id,
                           temporal_link_root=None, resume=False, row_batch=32768):
    """Build immutable per-shard labels with bounded adjacent-shard reads."""
    source_root, output_root = Path(source_root), Path(output_root)
    run = json.loads((source_root/"run.json").read_text(encoding="utf-8"))
    config = run["config"]
    if (not config.get("freeze_templates") or config.get("novelty") != "off"
            or config.get("waveform_cache_mode") !=
            "uncertain40_stratified_assigned40_random20_v1"
            or "template_linking" not in config):
        raise ValueError("Motion links require frozen templates and balanced waveform cache")
    if row_batch < 1:
        raise ValueError("Row batch must be positive")
    probe = next((p for p in run["source"]["probes"]
                  if p["probe_id"] == probe_id), None)
    if probe is None:
        raise ValueError("Probe not present in source run")
    sample_rate = float(probe["sample_rate_hz"])
    paths = sorted((source_root/probe_id).glob("*.h5"))
    if not paths:
        raise ValueError("No completed source shards")
    identities = [_source_identity(path) for path in paths]
    if len({item["day_id"] for item in identities}) != 1:
        raise ValueError("Motion links currently require one day")
    if any(item["config_sha256"] != run["config_sha256"] for item in identities):
        raise ValueError("Source shard configuration changed")
    if any(item["stop_sample"] != next_item["start_sample"]
           for item,next_item in zip(identities[:-1],identities[1:])):
        raise ValueError("Motion links currently require contiguous shards")
    if any((item["stop_sample"]-item["start_sample"])/sample_rate > 60.
           for item in identities):
        raise ValueError("Motion links require shards of at most 60 seconds")
    with h5py.File(paths[0], "r") as h:
        mapping = h["template_linking/template_to_unit"][:]
        positions = h["channel_positions_um"][:].astype(np.float64)
        shanks = h["channel_shank"][:]
        model_version = int(h["model_after"].attrs["version"])
        model_waveforms = h["model_after/waveforms"][:]
        model_channels = h["model_after/channels"][:]
        model_anchors = h["model_after/anchors"][:]
    if not len(mapping) or np.any(mapping < 0) or len(np.unique(shanks)) != 1:
        raise ValueError("Motion link model must have one shank and valid groups")
    for path in paths[1:]:
        with h5py.File(path, "r") as h:
            if (int(h["model_after"].attrs["version"]) != model_version or
                    not np.array_equal(h["template_linking/template_to_unit"][:],mapping) or
                    not np.array_equal(h["channel_positions_um"][:],positions) or
                    not np.array_equal(h["channel_shank"][:],shanks) or
                    not np.array_equal(h["model_after/waveforms"][:],model_waveforms) or
                    not np.array_equal(h["model_after/channels"][:],model_channels) or
                    not np.array_equal(h["model_after/anchors"][:],model_anchors)):
                raise ValueError("Frozen shard model, geometry, or link map changed")
    model_hash = hashlib.sha256()
    for value in (mapping,positions,shanks,model_waveforms,
                  model_channels,model_anchors):
        model_hash.update(np.asarray(value).tobytes())
    initial_map = np.arange(int(mapping.max())+1,dtype=np.int64)
    temporal_source = None
    if temporal_link_root is not None:
        temporal_path = Path(temporal_link_root)/"temporal_links.json"
        raw_temporal = temporal_path.read_bytes()
        temporal = json.loads(raw_temporal)
        temporal_identity = temporal["identity"]
        selected = temporal_identity["source_shards"]
        if (temporal_identity["probe_id"] != probe_id or
                temporal_identity["day_id"] != identities[0]["day_id"] or
                temporal_identity["first_sample"] != identities[0]["start_sample"] or
                temporal_identity["stop_sample"] != identities[-1]["stop_sample"] or
                len(selected) != len(identities) or
                any(a["path"] != b["path"] or a["start"] != b["start_sample"] or
                    a["stop"] != b["stop_sample"] or a["size"] != b["bytes"] or
                    a["mtime_ns"] != b["mtime_ns"]
                    for a,b in zip(selected,identities))):
            raise ValueError("Temporal base view does not match motion source shards")
        initial_map = np.asarray(temporal["base_group_to_temporal_unit"],
                                 dtype=np.int64)
        if initial_map.shape != (int(mapping.max())+1,):
            raise ValueError("Temporal base view has wrong group count")
        temporal_source = {"path":str(temporal_path.resolve()),
                           "sha256":hashlib.sha256(raw_temporal).hexdigest()}
    identity = {"schema_version":1,"rule_version":RULE_VERSION,
                "source_root":str(source_root.resolve()),
                "probe_id":probe_id,"day_id":identities[0]["day_id"],
                "config_sha256":run["config_sha256"],
                "model_sha256":model_hash.hexdigest(),
                "temporal_base_view":temporal_source,
                "source_shards":identities}
    journal_path = output_root/"motion_link_run.json"
    if resume:
        if not journal_path.is_file() or json.loads(journal_path.read_text()) != identity:
            raise ValueError("Motion link resume source or rule differs")
    else:
        if output_root.exists():
            raise FileExistsError(output_root)
        output_root.mkdir(parents=True)
        _atomic_json(journal_path, identity)
    folder = output_root/probe_id
    folder.mkdir(exist_ok=True)
    previous_map = initial_map.copy()
    accepted_total = 0
    for index,item in enumerate(identities):
        output = folder/(Path(item["path"]).stem+".json")
        if output.exists():
            if not resume:
                raise FileExistsError(output)
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved["source"] != item or saved["rule_version"] != RULE_VERSION:
                raise ValueError("Completed motion link shard changed")
            current_map = np.asarray(saved["base_group_to_unit"],np.int64)
            if current_map.shape != previous_map.shape:
                raise ValueError("Completed motion link map has wrong shape")
            previous_map = current_map
            accepted_total += len(saved["accepted_edges"])
            continue
        current_map = previous_map.copy()
        edges = []
        pair_count = 0
        shifts = []
        correlations = []
        motion_issue = None
        if index:
            pair = identities[index-1:index+1]
            images,observations,depth = _pair_histograms(
                pair,positions,sample_rate=sample_rate,row_batch=row_batch)
            try:
                shifts_array,correlations = _estimate_motion(
                    images,observations,float(depth[1]-depth[0]))
            except ValueError as exc:
                motion_issue = str(exc)
            else:
                shifts = shifts_array.tolist()
                early = _sketch(pair[0]["path"],mapping,positions,shifts_array,
                                first_sample=pair[0]["start_sample"],
                                sample_rate=sample_rate,row_batch=min(row_batch,512))
                late = _sketch(pair[1]["path"],mapping,positions,shifts_array,
                               first_sample=pair[0]["start_sample"],
                               sample_rate=sample_rate,row_batch=min(row_batch,512))
                edges,pair_count = select_motion_links(early,late,positions)
                for edge in edges:
                    current_map[edge["late_group"]] = previous_map[edge["early_group"]]
        payload = {"schema_version":1,"rule_version":RULE_VERSION,
                   "source":item,"base_group_to_unit":current_map.tolist(),
                   "accepted_edges":edges,"candidate_pairs":pair_count,
                   "motion_shift_um":shifts,"motion_correlation":correlations,
                   "motion_issue":motion_issue}
        _atomic_json(output,payload)
        previous_map = current_map
        accepted_total += len(edges)
    return {"output_root":str(output_root),"shards":len(identities),
            "accepted_edges":accepted_total}
