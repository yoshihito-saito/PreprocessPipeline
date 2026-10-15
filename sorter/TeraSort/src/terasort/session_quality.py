"""Auditable ground-truth gate for opt-in session sorting.

This evaluator is bounded by the benchmark dataset, not by a 100 TB archive.
Use the standard SpikeInterface comparison as an independent cross-check
before a publication claim.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment


def _one_to_one(left, right, tolerance):
    i = j = matched = 0
    while i < len(left) and j < len(right):
        delta = int(left[i]) - int(right[j])
        if abs(delta) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matched


def _collision_keys(times, labels, tolerance):
    """Events close to a spike from a different ground-truth unit."""
    order = np.argsort(times, kind="stable")
    times = np.asarray(times)[order]
    labels = np.asarray(labels)[order]
    collided = set()
    for i in range(1, len(times)):
        if times[i] - times[i-1] <= tolerance and labels[i] != labels[i-1]:
            collided.add((labels[i-1], int(times[i-1])))
            collided.add((labels[i], int(times[i])))
    return collided


def _group_times(times, labels):
    order = np.argsort(times, kind="stable")
    result = defaultdict(list)
    for time, label in zip(np.asarray(times)[order], np.asarray(labels)[order]):
        result[label].append(int(time))
    return {unit: np.asarray(values, np.int64) for unit, values in result.items()}


def score_sorting(gt_times, gt_labels, predicted_times, predicted_labels, *,
                  tolerance_samples, collision_tolerance_samples=None,
                  late_start_sample=None):
    """One-to-one unit matching with IoU and spike precision/recall/F-scores."""
    if tolerance_samples < 0:
        raise ValueError("Negative spike match tolerance")
    gt = _group_times(gt_times, gt_labels)
    predicted = _group_times(predicted_times, predicted_labels)
    collision_tolerance = (tolerance_samples if collision_tolerance_samples is None
                           else collision_tolerance_samples)
    collision_keys = _collision_keys(gt_times, gt_labels, collision_tolerance)
    duration_midpoint = (late_start_sample if late_start_sample is not None
                         else ((min(gt_times) + max(gt_times)) / 2
                               if len(gt_times) else 0))
    late_gt = {unit for unit, times in gt.items()
               if times[0] > duration_midpoint}
    rare_gt = {unit for unit, times in gt.items() if len(times) <= 50}
    gt_ids, pred_ids = list(gt), list(predicted)
    if not gt_ids or not pred_ids:
        return {"gt_units": len(gt_ids), "predicted_units": len(pred_ids),
                "recovered_units_iou_0p8": 0, "true_positive_spikes": 0,
                "spike_precision": 0., "spike_recall": 0., "spike_f1": 0.,
                "spike_f2": 0., "recovered_units_iou_0p5": 0,
                "unit_recall_iou_0p5": 0., "unit_recall_iou_0p8": 0.,
                "gt_units_with_50pct_spike_coverage": 0,
                "represented_gt_unit_ids_50pct": [],
                "recovered_gt_unit_ids_iou_0p5": [],
                "recovered_gt_unit_ids_iou_0p8": [],
                "split_gt_unit_ids": [],
                "merged_predicted_unit_ids": [],
                "split_gt_units": 0, "merged_predicted_units": 0,
                "collision_gt_spikes": len(collision_keys),
                "collision_recovered_spikes": 0,
                "rare_gt_units": len(rare_gt), "rare_recovered_units": 0,
                "late_gt_units": len(late_gt), "late_recovered_units": 0,
                "unit_matches": []}
    # A time sweep first discovers sparse unit pairs. No all-unit waveform
    # comparison or full recording x unit score tensor is built.
    gt_t = np.asarray(gt_times, np.int64)
    pred_t = np.asarray(predicted_times, np.int64)
    gt_l = np.asarray(gt_labels)
    pred_l = np.asarray(predicted_labels)
    gt_order = np.argsort(gt_t, kind="stable")
    pred_order = np.argsort(pred_t, kind="stable")
    gt_t, gt_l = gt_t[gt_order], gt_l[gt_order]
    pred_t, pred_l = pred_t[pred_order], pred_l[pred_order]
    possible = set()
    left = 0
    for time, label in zip(pred_t, pred_l):
        while left < len(gt_t) and gt_t[left] < time - tolerance_samples:
            left += 1
        k = left
        while k < len(gt_t) and gt_t[k] <= time + tolerance_samples:
            possible.add((gt_l[k], label))
            k += 1
    gi = {unit: index for index, unit in enumerate(gt_ids)}
    pi = {unit: index for index, unit in enumerate(pred_ids)}
    iou = np.zeros((len(gt_ids), len(pred_ids)), np.float32)
    matched = np.zeros_like(iou, dtype=np.int64)
    for gt_id, pred_id in possible:
        count = _one_to_one(gt[gt_id], predicted[pred_id],
                            tolerance_samples)
        union = len(gt[gt_id]) + len(predicted[pred_id]) - count
        iou[gi[gt_id], pi[pred_id]] = count / union if union else 0
        matched[gi[gt_id], pi[pred_id]] = count
    row, col = linear_sum_assignment(iou, maximize=True)
    links = [{
        "ground_truth_unit": str(gt_ids[i]),
        "predicted_unit": str(pred_ids[j]),
        "iou": float(iou[i, j]),
        "matched_spikes": int(matched[i, j]),
    } for i, j in zip(row, col) if iou[i, j] > 0]
    true_positive = sum(link["matched_spikes"] for link in links)
    # These are diagnostics rather than extra acceptance gates. A split has
    # substantial spike overlap with two output units; a merge is reciprocal.
    overlap_gt = matched / np.maximum(1, np.array(
        [len(gt[unit]) for unit in gt_ids], np.int64))[:, None]
    overlap_pred = matched / np.maximum(1, np.array(
        [len(predicted[unit]) for unit in pred_ids], np.int64))[None, :]
    split = np.sum(overlap_gt >= .2, axis=1) >= 2
    merged = np.sum(overlap_pred >= .2, axis=0) >= 2
    partially_recovered = {gt_ids[i] for i, j in zip(row, col)
                           if iou[i, j] >= .5}
    recovered = {gt_ids[i] for i, j in zip(row, col) if iou[i, j] >= .8}
    represented = {gt_ids[i] for i in range(len(gt_ids))
                   if np.max(overlap_gt[i]) >= .5}
    collision_recovered = 0
    for i, j in zip(row, col):
        if not matched[i, j]:
            continue
        left, right = gt[gt_ids[i]], predicted[pred_ids[j]]
        a = b = 0
        while a < len(left) and b < len(right):
            delta = int(left[a]) - int(right[b])
            if abs(delta) <= tolerance_samples:
                collision_recovered += (gt_ids[i], int(left[a])) in collision_keys
                a += 1
                b += 1
            elif delta < 0:
                a += 1
            else:
                b += 1
    return {
        "gt_units": len(gt_ids), "predicted_units": len(pred_ids),
        "recovered_units_iou_0p8": sum(link["iou"] >= .8 for link in links),
        "recovered_units_iou_0p5": sum(link["iou"] >= .5 for link in links),
        "unit_recall_iou_0p5": len(partially_recovered) / len(gt_ids),
        "unit_recall_iou_0p8": len(recovered) / len(gt_ids),
        # A neuron can be present but split or contaminated enough to miss
        # the stricter IoU recovery criterion. Count it separately rather
        # than hiding this distinction in aggregate spike recall.
        "gt_units_with_50pct_spike_coverage": len(represented),
        "represented_gt_unit_ids_50pct": sorted(map(str, represented)),
        "recovered_gt_unit_ids_iou_0p5": sorted(map(str, partially_recovered)),
        "recovered_gt_unit_ids_iou_0p8": sorted(map(str, recovered)),
        "true_positive_spikes": true_positive,
        "spike_precision": true_positive / len(pred_t) if len(pred_t) else 0.,
        "spike_recall": true_positive / len(gt_t) if len(gt_t) else 0.,
        "spike_f1": (2*true_positive/(len(pred_t)+len(gt_t))
                     if len(pred_t)+len(gt_t) else 0.),
        "spike_f2": (5*true_positive/(4*len(gt_t)+len(pred_t))
                     if 4*len(gt_t)+len(pred_t) else 0.),
        "split_gt_units": int(np.sum(split)),
        "merged_predicted_units": int(np.sum(merged)),
        "split_gt_unit_ids": sorted(str(gt_ids[i]) for i in np.flatnonzero(split)),
        "merged_predicted_unit_ids": sorted(str(pred_ids[i]) for i in np.flatnonzero(merged)),
        "collision_gt_spikes": len(collision_keys),
        "collision_recovered_spikes": collision_recovered,
        "rare_gt_units": len(rare_gt),
        "rare_recovered_units": len(recovered & rare_gt),
        "late_gt_units": len(late_gt),
        "late_recovered_units": len(recovered & late_gt),
        "unit_matches": links,
    }


def load_session_spikes(root, probe_id, *, first=0, stop=None,
                        resolve_template_links=False, export_filter=False,
                        view_root=None, temporal_link_root=None,
                        motion_link_root=None):
    if export_filter and view_root is not None:
        raise ValueError("Use either an in-shard export filter or a session view")
    if temporal_link_root is not None and motion_link_root is not None:
        raise ValueError("Motion view already includes its temporal base labels")
    folder = Path(root) / probe_id
    times, labels = [], []
    spans = []
    day_link_maps = {}
    temporal = None
    temporal_sources = {}
    if temporal_link_root is not None:
        temporal = json.loads((Path(temporal_link_root) /
                               "temporal_links.json").read_text(encoding="utf-8"))
        identity = temporal["identity"]
        if (temporal.get("schema_version") != 1 or
                identity["probe_id"] != probe_id or
                first < identity["first_sample"] or stop is None or
                stop > identity["stop_sample"] or stop <= first or
                not resolve_template_links):
            raise ValueError("Temporal link view does not contain quality interval or base map")
        temporal_sources = {item["path"]: item for item in identity["source_shards"]}
        if not temporal_sources:
            raise ValueError("Temporal link view has no source shards")
    seen_temporal_sources = set()
    motion_sources = {}
    if motion_link_root is not None:
        motion_run = json.loads((Path(motion_link_root) /
                                 "motion_link_run.json").read_text(encoding="utf-8"))
        if (motion_run.get("schema_version") != 1 or
                motion_run["probe_id"] != probe_id or
                motion_run["source_root"] != str(Path(root).resolve()) or
                not resolve_template_links):
            raise ValueError("Motion link view does not match source or base map")
        motion_sources = {item["path"]:item for item in motion_run["source_shards"]}
    seen_motion_sources = set()
    for path in sorted(folder.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete", False):
                raise ValueError(f"Incomplete shard: {path}")
            start = int(handle.attrs["start_sample"])
            end = int(handle.attrs["stop_sample"])
            if end <= first or (stop is not None and start >= stop):
                continue
            if temporal is not None:
                resolved = str(path.resolve())
                item = temporal_sources.get(resolved)
                stat = path.stat()
                if (item is None or item["start"] != start or item["stop"] != end or
                        item["size"] != stat.st_size or item["mtime_ns"] != stat.st_mtime_ns or
                        str(handle.attrs["day_id"]) != identity["day_id"]):
                    raise ValueError("Temporal link view source shard changed")
                seen_temporal_sources.add(resolved)
            motion_mapping = None
            if motion_link_root is not None:
                resolved = str(path.resolve())
                item = motion_sources.get(resolved)
                stat = path.stat()
                if (item is None or item["start_sample"] != start or
                        item["stop_sample"] != end or item["bytes"] != stat.st_size or
                        item["mtime_ns"] != stat.st_mtime_ns or
                        item["day_id"] != str(handle.attrs["day_id"])):
                    raise ValueError("Motion link source shard changed")
                payload = json.loads((Path(motion_link_root)/probe_id/
                                      (path.stem+".json")).read_text(encoding="utf-8"))
                if (payload["source"] != item or payload.get("schema_version") != 1 or
                        payload.get("rule_version") != motion_run["rule_version"]):
                    raise ValueError("Motion link shard map does not match source")
                motion_mapping = np.asarray(payload["base_group_to_unit"],
                                            dtype=np.int64)
                if np.any(motion_mapping < 0):
                    raise ValueError("Motion link shard map contains negative units")
                seen_motion_sources.add(resolved)
            spans.append((max(start, first), min(end, stop) if stop is not None else end))
            spikes = handle["spikes"][:]
            if export_filter and "spike_export_mask" in handle:
                export_mask = handle["spike_export_mask"][:]
                if export_mask.shape != (len(spikes),) or np.any(export_mask > 1):
                    raise ValueError(f"Invalid spike export mask in shard: {path}")
                spikes = spikes[export_mask.astype(bool)]
            if view_root is not None:
                from .session_view import read_view_mask
                spikes = spikes[read_view_mask(
                    view_root, probe_id, path, handle, rows=len(spikes))]
            keep = spikes["sample_index"] >= first
            if stop is not None:
                keep &= spikes["sample_index"] < stop
            spikes = spikes[keep]
            times.append(spikes["sample_index"])
            # Day/probe namespace prevents accidental cross-day identity.
            day_id = str(handle.attrs["day_id"])
            units = spikes["unit_id"]
            if resolve_template_links:
                if "template_linking/template_to_unit" not in handle:
                    raise ValueError(
                        f"Template links requested but absent from shard: {path}"
                    )
                mapping = handle["template_linking/template_to_unit"][:]
                previous = day_link_maps.get(day_id)
                if previous is not None and (len(mapping) < len(previous) or
                        not np.array_equal(previous, mapping[:len(previous)])):
                    raise ValueError(f"Inconsistent template map for day {day_id}")
                day_link_maps[day_id] = mapping
                if len(units) and (np.any(units < 0) or np.any(units >= len(mapping))):
                    raise ValueError(f"Shard spike references missing template map entry: {path}")
                units = mapping[units]
                if temporal is not None:
                    linked = np.asarray(temporal["base_group_to_temporal_unit"],
                                        dtype=np.int64)
                    if len(linked) != int(mapping.max()) + 1:
                        raise ValueError("Temporal link map has incompatible base groups")
                    units = linked[units]
                if motion_mapping is not None:
                    if len(motion_mapping) != int(mapping.max()) + 1:
                        raise ValueError("Motion link map has incompatible base groups")
                    units = motion_mapping[units]
            labels.extend(f"{probe_id}/{day_id}/{int(unit)}"
                          for unit in units)
    from .session_recovery import read_recovered, MAX_PER_DAY
    recovery_counts = {}
    for path in sorted(folder.glob('*.h5')):
        with h5py.File(path, 'r') as handle:
            recovered = read_recovered(handle)
            if not len(recovered):
                continue
            if view_root is not None or temporal_link_root is not None or motion_link_root is not None:
                raise ValueError('Derived link/filter views must explicitly include novelty recovery rows')
            day_id = str(handle.attrs['day_id'])
            recovery_counts[day_id] = recovery_counts.get(day_id, 0) + len(recovered)
            if recovery_counts[day_id] > MAX_PER_DAY:
                raise ValueError('Oversized retrospective recovery day')
            keep = recovered['sample_index'] >= first
            if stop is not None:
                keep &= recovered['sample_index'] < stop
            if export_filter and 'spike_export_filter_json' in handle.attrs:
                minimum = json.loads(handle.attrs['spike_export_filter_json'])['minimum']
                keep &= recovered['fitted_amplitude'] >= minimum
            recovered = recovered[keep]
            if not _inside_spans(recovered['sample_index'], spans).all():
                raise ValueError('Novelty recovery is outside completed source coverage')
            units = recovered['unit_id']
            if resolve_template_links:
                units = handle['template_linking/template_to_unit'][:][units]
            times.append(recovered['sample_index'])
            labels.extend(f'{probe_id}/{day_id}/{int(unit)}' for unit in units)
    if not spans:
        raise ValueError("No completed session shards in benchmark interval")
    if temporal is not None:
        expected = {path for path, item in temporal_sources.items()
                    if item["stop"] > first and item["start"] < stop}
        if seen_temporal_sources != expected:
            raise ValueError("Temporal link view source coverage changed")
    if motion_link_root is not None:
        expected = {path for path,item in motion_sources.items()
                    if item["stop_sample"] > first and
                    (stop is None or item["start_sample"] < stop)}
        if seen_motion_sources != expected:
            raise ValueError("Motion link view source coverage changed")
    return (np.concatenate(times) if times else np.empty(0, np.int64),
            np.asarray(labels, dtype=str), spans)


def _inside_spans(times, spans):
    """Mask to the exact disjoint source-clock intervals with completed shards."""
    times = np.asarray(times, np.int64)
    keep = np.zeros(times.shape, dtype=bool)
    for first, stop in spans:
        keep |= (times >= first) & (times < stop)
    return keep


def _merged_spans(spans):
    """Normalize adjacent intervals and reject duplicate or overlapping shards."""
    merged = []
    for first, stop in sorted((int(first), int(stop)) for first, stop in spans):
        if stop <= first:
            raise ValueError("Benchmark span must have positive duration")
        if merged and first < merged[-1][1]:
            raise ValueError("Overlapping completed shards in benchmark interval")
        if merged and first == merged[-1][1]:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((first, stop))
    return merged


def paired_neuron_recovery(session, baseline):
    """Name neurons gained or lost by the session route at each recovery level."""
    result = {}
    for level, field in (
            ("represented_50pct", "represented_gt_unit_ids_50pct"),
            ("iou_0p5", "recovered_gt_unit_ids_iou_0p5"),
            ("iou_0p8", "recovered_gt_unit_ids_iou_0p8")):
        ours = set(session[field])
        reference = set(baseline[field])
        result[level] = {
            "session_only_gt_unit_ids": sorted(ours - reference),
            "baseline_only_gt_unit_ids": sorted(reference - ours),
            "shared_gt_unit_count": len(ours & reference),
        }
    return result


def evaluate_case(case):
    required = {"name", "kind", "ground_truth", "session_output",
                "probe_id", "kilosort_dir"}
    if not required.issubset(case):
        raise ValueError(f"Benchmark case is missing {required - set(case)}")
    first = int(case.get("start_sample", 0))
    stop = case.get("stop_sample")
    stop = int(stop) if stop is not None else None
    if first < 0 or (stop is not None and stop <= first):
        raise ValueError("Invalid benchmark interval")
    if "expected_spans" in case and stop is None:
        raise ValueError("Expected spans require a finite stop_sample")
    with np.load(case["ground_truth"], allow_pickle=False) as data:
        gt_t = np.asarray(data["times"], np.int64)
        gt_l = np.asarray(data["labels"], np.int64)
        rate = float(data["sampling_frequency"])
    if gt_t.shape != gt_l.shape or rate <= 0:
        raise ValueError("Invalid ground-truth bundle")
    tolerance = max(1, round(rate * .0004))
    collision_tolerance = max(1, round(rate * .0006))
    resolve_template_links = bool(case.get("resolve_template_links", False))
    export_filter = bool(case.get("export_filter", False))
    view_root = case.get("session_view")
    temporal_link_root = case.get("temporal_link_view")
    motion_link_root = case.get("motion_link_view")
    pred_t, pred_l, spans = load_session_spikes(
        case["session_output"], case["probe_id"], first=first, stop=stop,
        resolve_template_links=resolve_template_links,
        export_filter=export_filter, view_root=view_root,
        temporal_link_root=temporal_link_root,
        motion_link_root=motion_link_root)
    spans = _merged_spans(spans)
    coverage_verified = stop is not None
    if coverage_verified:
        expected = case.get("expected_spans", [[first, stop]])
        if not isinstance(expected, list) or not expected:
            raise ValueError("Benchmark expected_spans must be a nonempty list")
        try:
            expected = _merged_spans(expected)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid benchmark expected_spans") from exc
        if (expected[0][0] != first or expected[-1][1] != stop or
                spans != expected):
            raise ValueError(
                f"Completed shards {spans} do not cover expected spans {expected}"
            )
    keep = _inside_spans(gt_t, spans)
    gt_t, gt_l = gt_t[keep], gt_l[keep]
    ks = Path(case["kilosort_dir"])
    ks_t = np.load(ks / "spike_times.npy", mmap_mode="r").ravel()
    ks_l = np.load(ks / "spike_clusters.npy", mmap_mode="r").ravel()
    keep = _inside_spans(ks_t, spans)
    result = {
        "name": case["name"], "kind": case["kind"],
        "coverage_verified": coverage_verified,
        "scored_spans": [list(span) for span in spans],
        "template_links_resolved": resolve_template_links,
        "export_filter_applied": export_filter,
        "session_view": str(view_root) if view_root is not None else None,
        "temporal_link_view": (str(temporal_link_root) if temporal_link_root
                               is not None else None),
        "motion_link_view": (str(motion_link_root) if motion_link_root
                             is not None else None),
        "tolerance_samples": tolerance,
        "collision_tolerance_samples": collision_tolerance,
        "session": score_sorting(gt_t, gt_l, pred_t, pred_l,
                                 tolerance_samples=tolerance,
                                 collision_tolerance_samples=collision_tolerance,
                                 late_start_sample=(min(a for a, _ in spans) +
                                                    max(b for _, b in spans)) / 2),
        "kilosort4": score_sorting(gt_t, gt_l, ks_t[keep], ks_l[keep],
                                  tolerance_samples=tolerance,
                                  collision_tolerance_samples=collision_tolerance,
                                  late_start_sample=(min(a for a, _ in spans) +
                                                     max(b for _, b in spans)) / 2),
    }
    result["paired_neuron_recovery"] = paired_neuron_recovery(
        result["session"], result["kilosort4"])
    return result


def evaluate_suite(suite_path):
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Quality suite needs cases")
    results = [evaluate_case(case) for case in cases]
    gt_units = sum(item["session"]["gt_units"] for item in results)
    if gt_units == 0 or any(item["session"]["gt_units"] == 0 for item in results):
        raise ValueError("Every quality case needs ground-truth units in completed spans")
    recovery_loss = sum(
        item["kilosort4"]["recovered_units_iou_0p8"] -
        item["session"]["recovered_units_iou_0p8"] for item in results
    ) / gt_units
    partial_recovery_loss = sum(
        item["kilosort4"]["recovered_units_iou_0p5"] -
        item["session"]["recovered_units_iou_0p5"] for item in results
    ) / gt_units
    def weighted(metric, route):
        # Dataset-level mean keeps the acceptance gate transparent even when
        # datasets differ greatly in duration and spike density.
        return float(np.mean([item[route][metric] for item in results]))
    precision_loss = weighted("spike_precision", "kilosort4") - weighted(
        "spike_precision", "session")
    recall_loss = weighted("spike_recall", "kilosort4") - weighted(
        "spike_recall", "session")
    f1_loss = weighted("spike_f1", "kilosort4") - weighted(
        "spike_f1", "session")
    f2_loss = weighted("spike_f2", "kilosort4") - weighted(
        "spike_f2", "session")
    kinds = {item["kind"] for item in results}
    ready = (len(results) >= 3 and
             {"base", "drift", "collision"}.issubset(kinds) and
             all(item.get("coverage_verified", False) for item in results))
    subgroup_losses = []
    for item in results:
        session, baseline = item["session"], item["kilosort4"]
        row = {
            "name": item["name"],
            "recovery_fraction_loss": (
                baseline["recovered_units_iou_0p8"] -
                session["recovered_units_iou_0p8"]) / session["gt_units"],
            "partial_recovery_fraction_loss": (
                baseline["recovered_units_iou_0p5"] -
                session["recovered_units_iou_0p5"]) / session["gt_units"],
            "precision_loss": baseline["spike_precision"] -
                              session["spike_precision"],
            "recall_loss": baseline["spike_recall"] -
                           session["spike_recall"],
        }
        for name, denominator, numerator in (
                ("collision_recall", "collision_gt_spikes",
                 "collision_recovered_spikes"),
                ("rare_unit_recovery", "rare_gt_units",
                 "rare_recovered_units"),
                ("late_unit_recovery", "late_gt_units",
                 "late_recovered_units")):
            count = session[denominator]
            row[name + "_loss"] = ((baseline[numerator] - session[numerator]) /
                                    count if count else None)
        subgroup_losses.append(row)
    passed = bool(ready and recovery_loss <= .02 and
                  partial_recovery_loss <= .02 and
                  precision_loss <= .02 and recall_loss <= .02 and
                  all(max(value for key, value in row.items()
                          if key.endswith("_loss") and value is not None) <= .05
                      for row in subgroup_losses))
    return {
        "quality_gate_passed": passed,
        "suite_complete": ready,
        "aggregate_recovery_fraction_loss": recovery_loss,
        "aggregate_partial_recovery_fraction_loss": partial_recovery_loss,
        "mean_precision_loss": precision_loss,
        "mean_recall_loss": recall_loss,
        "mean_f1_loss": f1_loss,
        "mean_f2_loss": f2_loss,
        "subgroup_losses": subgroup_losses,
        "cases": results,
    }
