"""Opt-in, bounded co-firing output masks for frozen-template sessions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .cofire_dedup import CofirePairLearner, CofireSuppressor
from .session_view import BLOCK_ROWS, _check_shard


SCHEMA_VERSION = 2
ALGORITHM = "cofire_dedup_v1"
MAX_HALO_EVENTS = 8192
SELECTION = {"min_cofires": 20, "min_cofire_fraction": .95,
             "max_distance_um": 250., "max_channel_index_distance": 12,
             "max_active_events": 8192, "max_pairs": 100_000}


def _hash_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _run_info(root: Path, probe_id: str) -> tuple[dict, dict]:
    run_path = root / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if not run.get("config", {}).get("freeze_templates", False):
        raise ValueError("Co-firing views currently require frozen templates")
    probes = [probe for probe in run["source"]["probes"]
              if probe["probe_id"] == probe_id]
    if len(probes) != 1:
        raise ValueError(f"Probe is missing or duplicated in run.json: {probe_id}")
    return run, probes[0]


def _shards(root: Path, probe_id: str) -> list[dict]:
    paths = sorted((root / probe_id).glob("*.h5"))
    if not paths:
        raise ValueError(f"No completed shards for probe {probe_id}: {root}")
    records = []
    for path in paths:
        with h5py.File(path, "r") as source:
            rows = _check_shard(source, path)
            records.append({"path": path, "start": int(source.attrs["start_sample"]),
                            "stop": int(source.attrs["stop_sample"]),
                            "day_id": str(source.attrs["day_id"]), "rows": rows})
    if any(record["stop"] <= record["start"] for record in records):
        raise ValueError("Source shard has an invalid sample span")
    if any(right["start"] < left["stop"]
           for left, right in zip(records, records[1:])):
        raise ValueError("Source shards overlap or are out of order")
    if len({record["day_id"] for record in records}) != 1:
        raise ValueError("A co-firing view requires one stable day-local template bank")
    return records


def _model_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with h5py.File(path, "r") as source:
        for name in ("waveforms", "channels", "anchors"):
            key = f"model_after/{name}"
            if key not in source:
                raise ValueError(f"Source shard lacks model bank {name}: {path}")
            dataset = source[key]
            digest.update(name.encode())
            digest.update(str(dataset.shape).encode())
            digest.update(dataset.dtype.str.encode())
            for first in range(0, len(dataset), 64):
                digest.update(np.asarray(dataset[first:first + 64]).tobytes())
    return digest.hexdigest()


def _good_pilot_rows(source: h5py.File, rows: np.ndarray) -> np.ndarray:
    if "qc" not in source or "interval_bad" not in source["qc"].dtype.names:
        raise ValueError("Pilot shard lacks interval QC")
    good = np.ones(len(rows), bool)
    for interval in source["qc"][:]:
        if interval["interval_bad"]:
            good &= ((rows["sample_index"] < interval["start_sample"])
                     | (rows["sample_index"] >= interval["stop_sample"]))
    return rows[good]


def _learn_policy(calibration_root: Path, probe_id: str, records: list[dict],
                  geometry, sample_rate: float, *, amplitude_min: float,
                  pilot_seconds: float, tolerance_samples: int,
                  model_sha256: str) -> tuple[dict, int, int]:
    pilot_start = records[0]["start"]
    pilot_stop = pilot_start + round(pilot_seconds * sample_rate)
    if pilot_stop > records[-1]["stop"]:
        raise ValueError("Calibration output does not cover the requested pilot")
    learner = CofirePairLearner(geometry, tolerance_samples=tolerance_samples,
                                amplitude_min=amplitude_min, **SELECTION)
    covered = pilot_start
    for record in records:
        if record["start"] >= pilot_stop:
            break
        if record["start"] != covered:
            raise ValueError("Calibration pilot contains a source gap")
        if _model_hash(record["path"]) != model_sha256:
            raise ValueError("Template bank changed inside calibration pilot")
        with h5py.File(record["path"], "r") as source:
            spikes = source["spikes"]
            for first in range(0, record["rows"], BLOCK_ROWS):
                rows = spikes[first:first + BLOCK_ROWS]
                rows = rows[(rows["sample_index"] >= pilot_start)
                            & (rows["sample_index"] < pilot_stop)]
                if len(rows):
                    learner.observe(_good_pilot_rows(source, rows))
        covered = min(record["stop"], pilot_stop)
    if covered != pilot_stop:
        raise ValueError("Calibration pilot is incomplete")
    policy = learner.policy()
    return policy.pair_winners, pilot_start, pilot_stop


def _halo(path: Path, *, side: str, boundary: int,
          amplitude_min: float) -> np.ndarray:
    """Read only rows within the requested boundary window."""
    parts = []
    total = 0
    with h5py.File(path, "r") as source:
        spikes = source["spikes"]
        dtype = spikes.dtype
        count = len(spikes)
        if side == "tail":
            stops = range(count, 0, -BLOCK_ROWS)
            for stop in stops:
                rows = spikes[max(0, stop - BLOCK_ROWS):stop]
                selected = rows[rows["sample_index"] >= boundary]
                total += len(selected)
                if total > MAX_HALO_EVENTS:
                    raise MemoryError("Boundary halo exceeds the event budget")
                parts.append(selected)
                if len(selected) < len(rows):
                    break
            parts.reverse()
        else:
            for first in range(0, count, BLOCK_ROWS):
                rows = spikes[first:first + BLOCK_ROWS]
                selected = rows[rows["sample_index"] <= boundary]
                total += len(selected)
                if total > MAX_HALO_EVENTS:
                    raise MemoryError("Boundary halo exceeds the event budget")
                parts.append(selected)
                if len(selected) < len(rows):
                    break
    if not parts:
        return np.empty(0, dtype=dtype)
    rows = np.concatenate(parts)
    if len(rows) > 1 and np.any(np.diff(rows["sample_index"]) < 0):
        raise ValueError(f"Boundary assignments are out of order: {path}")
    return rows[rows["fitted_amplitude"] >= amplitude_min]


def _valid_view(final: Path, record: dict, config_hash: str) -> bool:
    if not final.is_file():
        return False
    stat = record["path"].stat()
    with h5py.File(final, "r") as view:
        return (bool(view.attrs.get("complete", False))
                and int(view.attrs.get("schema_version", -1)) == SCHEMA_VERSION
                and str(view.attrs.get("config_sha256", "")) == config_hash
                and str(view.attrs.get("source_shard", "")) == record["path"].name
                and int(view.attrs.get("source_start_sample", -1)) == record["start"]
                and int(view.attrs.get("source_stop_sample", -1)) == record["stop"]
                and int(view.attrs.get("source_size_bytes", -1)) == stat.st_size
                and int(view.attrs.get("source_mtime_ns", -1)) == stat.st_mtime_ns
                and 0 <= int(view.attrs.get("kept_spikes", -1)) <= record["rows"]
                and "spike_mask" in view
                and view["spike_mask"].shape == (record["rows"],))


def _write_shard(record: dict, previous: dict | None, following: dict | None,
                 destination: Path, config: dict, config_hash: str) -> int:
    path = record["path"]
    amplitude_min = config["amplitude_min"]
    tolerance = config["tolerance_samples"]
    pair_winners = {(a, b): winner for a, b, winner in config["pair_winners"]}
    suppressor = CofireSuppressor(pair_winners, tolerance_samples=tolerance)
    partial = destination.with_name(destination.name + ".partial")
    if previous and record["start"] - previous["stop"] <= tolerance:
        rows = _halo(previous["path"], side="tail",
                     boundary=record["start"] - tolerance,
                     amplitude_min=amplitude_min)
        for row in rows:
            suppressor.observe(row["sample_index"], row["unit_id"], None)
    with h5py.File(path, "r") as source, h5py.File(partial, "w") as view:
        stat = path.stat()
        view.attrs["schema_version"] = SCHEMA_VERSION
        view.attrs["config_sha256"] = config_hash
        view.attrs["source_shard"] = path.name
        view.attrs["source_start_sample"] = record["start"]
        view.attrs["source_stop_sample"] = record["stop"]
        view.attrs["source_size_bytes"] = stat.st_size
        view.attrs["source_mtime_ns"] = stat.st_mtime_ns
        count = record["rows"]
        mask = view.create_dataset("spike_mask", shape=(count,), dtype="u1",
                                   chunks=(min(BLOCK_ROWS, count),) if count else None,
                                   compression="lzf" if count else None)
        spikes = source["spikes"]
        last_time = None
        for first in range(0, count, BLOCK_ROWS):
            stop = min(count, first + BLOCK_ROWS)
            rows = spikes[first:stop]
            eligible = rows["fitted_amplitude"] >= amplitude_min
            values = eligible.copy()
            for offset, row in enumerate(rows):
                time = int(row["sample_index"])
                if last_time is not None and time < last_time:
                    raise ValueError(f"Source assignments are out of order: {path}")
                last_time = time
                if not eligible[offset]:
                    continue
                victims = suppressor.observe(time, row["unit_id"], first + offset)
                for victim in victims:
                    if victim is None:
                        continue
                    if first <= victim < stop:
                        values[victim - first] = False
                    else:
                        mask[victim] = 0
            mask[first:stop] = values.astype(np.uint8)
        if following and following["start"] - record["stop"] <= tolerance:
            rows = _halo(following["path"], side="head",
                         boundary=record["stop"] - 1 + tolerance,
                         amplitude_min=amplitude_min)
            for row in rows:
                for victim in suppressor.observe(row["sample_index"],
                                                  row["unit_id"], None):
                    if victim is not None:
                        mask[victim] = 0
        kept = sum(int(np.count_nonzero(mask[first:first + BLOCK_ROWS]))
                   for first in range(0, count, BLOCK_ROWS))
        view.attrs["kept_spikes"] = kept
        view.attrs["complete"] = True
        view.flush()
    os.replace(partial, destination)
    return kept


def build_cofire_view(source_root, output_root, *, probe_id: str,
                      calibration_root=None, amplitude_min: float = .7,
                      pilot_seconds: float = 30., resume: bool = False) -> dict:
    """Learn on a bounded pilot and write immutable per-shard output masks."""
    source_root = Path(source_root).resolve()
    calibration_root = (Path(calibration_root).resolve()
                        if calibration_root is not None else source_root)
    output_root = Path(output_root).resolve()
    if output_root in (source_root, calibration_root):
        raise ValueError("The output view must differ from source and calibration roots")
    if not probe_id or Path(probe_id).name != probe_id or "/" in probe_id or "\\" in probe_id:
        raise ValueError("probe_id must be a single probe directory name")
    if (not np.isfinite(amplitude_min) or amplitude_min < 0
            or not np.isfinite(pilot_seconds) or pilot_seconds <= 0):
        raise ValueError("Invalid co-firing view thresholds")
    source_run, source_probe = _run_info(source_root, probe_id)
    calibration_run, calibration_probe = _run_info(calibration_root, probe_id)
    if (source_probe["sample_rate_hz"] != calibration_probe["sample_rate_hz"]
            or not np.array_equal(source_probe["geometry"],
                                  calibration_probe["geometry"])):
        raise ValueError("Calibration and source probe frames differ")
    source_records = _shards(source_root, probe_id)
    calibration_records = _shards(calibration_root, probe_id)
    if source_records[0]["day_id"] != calibration_records[0]["day_id"]:
        raise ValueError("Calibration and source day IDs differ")
    source_model_hash = _model_hash(source_records[0]["path"])
    if source_model_hash != _model_hash(calibration_records[0]["path"]):
        raise ValueError("Calibration and source template banks differ")
    tolerance = max(1, round(float(source_probe["sample_rate_hz"]) * .0004))
    winners, pilot_start, pilot_stop = _learn_policy(
        calibration_root, probe_id, calibration_records,
        source_probe["geometry"], float(source_probe["sample_rate_hz"]),
        amplitude_min=float(amplitude_min), pilot_seconds=float(pilot_seconds),
        tolerance_samples=tolerance, model_sha256=source_model_hash)
    config = {
        "schema_version": SCHEMA_VERSION, "algorithm": ALGORITHM,
        "source_root": str(source_root),
        "source_run_sha256": hashlib.sha256((source_root / "run.json").read_bytes()).hexdigest(),
        "calibration_root": str(calibration_root),
        "calibration_run_sha256": hashlib.sha256(
            (calibration_root / "run.json").read_bytes()).hexdigest(),
        "probe_id": probe_id, "day_id": source_records[0]["day_id"],
        "model_sha256": source_model_hash,
        "pilot_start_sample": pilot_start, "pilot_stop_sample": pilot_stop,
        "amplitude_min": float(amplitude_min), "tolerance_samples": tolerance,
        "cofire_selection": dict(SELECTION),
        "pair_winners": [[a, b, winner] for (a, b), winner in sorted(winners.items())],
    }
    config_hash = _hash_json(config)
    manifest = output_root / "view.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"Output view exists; pass resume: {output_root}")
        if not manifest.is_file():
            pending = output_root / "view.json.partial"
            if pending.is_file() and json.loads(pending.read_text(encoding="utf-8")) == config:
                os.replace(pending, manifest)
            elif not any(output_root.iterdir()):
                pending.write_text(json.dumps(config, indent=2) + "\n",
                                   encoding="utf-8")
                os.replace(pending, manifest)
            else:
                raise ValueError(f"Incomplete or unrelated co-firing view: {output_root}")
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Existing co-firing view has different settings: {output_root}")
    else:
        output_root.mkdir(parents=True)
        pending = output_root / "view.json.partial"
        pending.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(pending, manifest)
    destination = output_root / probe_id
    destination.mkdir(exist_ok=True)
    completed = skipped = seen = kept = 0
    for index, record in enumerate(source_records):
        final = destination / record["path"].name
        seen += record["rows"]
        if _valid_view(final, record, config_hash):
            with h5py.File(final, "r") as view:
                kept += int(view.attrs["kept_spikes"])
            skipped += 1
            continue
        if final.exists():
            raise ValueError(f"Existing view does not match source shard: {final}")
        if _model_hash(record["path"]) != source_model_hash:
            raise ValueError(f"Template bank changed inside source session: {record['path']}")
        previous = source_records[index - 1] if index else None
        following = source_records[index + 1] if index + 1 < len(source_records) else None
        kept += _write_shard(record, previous, following, final, config, config_hash)
        completed += 1
    return {"output_root": str(output_root), "shards_written": completed,
            "shards_skipped": skipped, "spikes_seen": seen, "spikes_kept": kept,
            "duplicate_pairs": len(winners), "pair_winners": config["pair_winners"]}


def read_cofire_view_mask(view_root, probe_id, source_path, source, *, rows: int):
    """Load a verified mask for the benchmark-sized quality evaluator."""
    view_root = Path(view_root).resolve()
    config = json.loads((view_root / "view.json").read_text(encoding="utf-8"))
    source_path = Path(source_path).resolve()
    source_root = source_path.parent.parent
    if (config.get("schema_version") != SCHEMA_VERSION
            or config.get("algorithm") != ALGORITHM
            or config.get("source_root") != str(source_root)
            or config.get("probe_id") != probe_id
            or config.get("source_run_sha256") != hashlib.sha256(
                (source_root / "run.json").read_bytes()).hexdigest()):
        raise ValueError("Co-firing view does not match source session")
    record = {"path": source_path, "start": int(source.attrs["start_sample"]),
              "stop": int(source.attrs["stop_sample"]), "rows": rows}
    final = view_root / probe_id / source_path.name
    if not _valid_view(final, record, _hash_json(config)):
        raise ValueError(f"Missing, incomplete, or stale co-firing view: {final}")
    with h5py.File(final, "r") as view:
        mask = view["spike_mask"][:]
    if np.any(mask > 1):
        raise ValueError(f"Invalid co-firing view mask: {final}")
    return mask.astype(bool)
