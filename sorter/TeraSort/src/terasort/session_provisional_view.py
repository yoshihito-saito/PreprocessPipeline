"""Bounded, non-destructive support masks for final rescue assignments."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .session_view import BLOCK_ROWS, _check_shard


ALGORITHM = "provisional_temporal_support_v1"
SCHEMA_VERSION = 1
MIN_SPIKES = 3
MIN_BINS = 2
BIN_SECONDS = 5
EPOCH_SECONDS = 30
MAX_SHARD_SECONDS = 300


def _config(source_root: Path) -> dict:
    path = source_root / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    settings = run["config"]
    if (settings.get("rescue_last_score_floor") is None or
            settings.get("rescue_passes") != 2):
        raise ValueError("Provisional support view requires two-tier rescue output")
    source_probes = run["source"]["probes"]
    probes = {item["probe_id"]: item["sample_rate_hz"]
              for item in source_probes}
    if (not probes or len(probes) != len(source_probes)
            or any(Path(name).name != name or "/" in name or "\\" in name
                   for name in probes)
            or any(not np.isfinite(rate) or rate <= 0
                   for rate in probes.values())):
        raise ValueError("Source run has invalid probe sample rates")
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "source_root": str(source_root.resolve()),
        "source_run_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rates_hz": probes,
        "final_pass_index": int(settings["residual_passes"]) + 1,
        "min_spikes": MIN_SPIKES,
        "min_bins": MIN_BINS,
        "bin_seconds": BIN_SECONDS,
        "epoch_seconds": EPOCH_SECONDS,
        "max_shard_seconds": MAX_SHARD_SECONDS,
    }


def _hash_config(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _valid_view(path: Path, source_path: Path, source: h5py.File,
                config_hash: str, rows: int) -> bool:
    if not path.is_file():
        return False
    stat = source_path.stat()
    with h5py.File(path, "r") as view:
        return (bool(view.attrs.get("complete", False))
                and int(view.attrs.get("schema_version", -1)) == SCHEMA_VERSION
                and str(view.attrs.get("config_sha256", "")) == config_hash
                and str(view.attrs.get("source_shard", "")) == source_path.name
                and int(view.attrs.get("source_start_sample", -1)) ==
                    int(source.attrs["start_sample"])
                and int(view.attrs.get("source_stop_sample", -1)) ==
                    int(source.attrs["stop_sample"])
                and int(view.attrs.get("source_size_bytes", -1)) == stat.st_size
                and int(view.attrs.get("source_mtime_ns", -1)) == stat.st_mtime_ns
                and 0 <= int(view.attrs.get("kept_spikes", -1)) <= rows
                and "spike_mask" in view and view["spike_mask"].shape == (rows,))


def _spans(source: h5py.File, config: dict, probe_id: str) -> tuple[int, int, int, int]:
    start = int(source.attrs["start_sample"])
    stop = int(source.attrs["stop_sample"])
    rate = float(config["sample_rates_hz"][probe_id])
    bin_samples = round(rate * BIN_SECONDS)
    epoch_samples = round(rate * EPOCH_SECONDS)
    if (stop <= start or stop - start > round(rate * MAX_SHARD_SECONDS)
            or bin_samples < 1 or epoch_samples < bin_samples
            or epoch_samples % bin_samples):
        raise ValueError("Provisional view requires valid shards of at most five minutes")
    return start, stop, bin_samples, epoch_samples


def _source_rows(source: h5py.File, first: int, stop: int):
    spikes = source["spikes"]
    rows = spikes[first:stop]
    if "spike_export_mask" in source:
        export = source["spike_export_mask"][first:stop]
        if len(export) != len(rows) or np.any(export > 1):
            raise ValueError("Invalid source export mask")
        base = export.astype(bool)
    else:
        base = np.ones(len(rows), bool)
    return rows, base


def _support(source: h5py.File, config: dict, probe_id: str, count: int):
    start, stop, bin_samples, epoch_samples = _spans(source, config, probe_id)
    if ("spike_export_mask" in source and
            source["spike_export_mask"].shape != (count,)):
        raise ValueError("Source export mask has wrong row count")
    mapping = source["template_linking/template_to_unit"][:]
    if mapping.ndim != 1 or np.any(mapping < 0):
        raise ValueError("Source shard has invalid template links")
    support_count = defaultdict(int)
    support_bins = defaultdict(int)
    previous_time = None
    final_pass = config["final_pass_index"]
    for first in range(0, count, BLOCK_ROWS):
        rows, base = _source_rows(source, first, min(count, first + BLOCK_ROWS))
        if len(rows):
            times = rows["sample_index"]
            if (np.any(times < start) or np.any(times >= stop) or
                    np.any(np.diff(times) < 0) or
                    (previous_time is not None and times[0] < previous_time)):
                raise ValueError("Source spike rows are outside shard or out of order")
            previous_time = int(times[-1])
        selected = rows[base & (rows["pass_index"] == final_pass)]
        if len(selected) and (np.any(selected["unit_id"] < 0) or
                              np.any(selected["unit_id"] >= len(mapping))):
            raise ValueError("Provisional spike references missing template link")
        for row in selected:
            offset = int(row["sample_index"]) - start
            epoch = offset // epoch_samples
            bin_id = (offset % epoch_samples) // bin_samples
            key = (epoch, int(mapping[row["unit_id"]]))
            support_count[key] = min(MIN_SPIKES, support_count[key] + 1)
            support_bins[key] |= 1 << bin_id
    eligible = {key for key, n in support_count.items()
                if n >= MIN_SPIKES and support_bins[key].bit_count() >= MIN_BINS}
    return eligible, mapping, start, epoch_samples


def _write_shard(source_path: Path, destination: Path, source: h5py.File,
                 config: dict, config_hash: str, probe_id: str, count: int) -> int:
    eligible, mapping, start, epoch_samples = _support(
        source, config, probe_id, count)
    partial = destination.with_name(destination.name + ".partial")
    stat = source_path.stat()
    with h5py.File(partial, "w") as view:
        view.attrs["schema_version"] = SCHEMA_VERSION
        view.attrs["config_sha256"] = config_hash
        view.attrs["source_shard"] = source_path.name
        view.attrs["source_start_sample"] = int(source.attrs["start_sample"])
        view.attrs["source_stop_sample"] = int(source.attrs["stop_sample"])
        view.attrs["source_size_bytes"] = stat.st_size
        view.attrs["source_mtime_ns"] = stat.st_mtime_ns
        mask = view.create_dataset(
            "spike_mask", shape=(count,), dtype="u1",
            chunks=(min(BLOCK_ROWS, count),) if count else None,
            compression="lzf" if count else None)
        kept = 0
        final_pass = config["final_pass_index"]
        for first in range(0, count, BLOCK_ROWS):
            stop = min(count, first + BLOCK_ROWS)
            rows, values = _source_rows(source, first, stop)
            provisional = np.flatnonzero(values & (rows["pass_index"] == final_pass))
            for index in provisional:
                row = rows[index]
                unit = int(row["unit_id"])
                if unit < 0 or unit >= len(mapping):
                    raise ValueError("Provisional spike references missing template link")
                epoch = (int(row["sample_index"]) - start) // epoch_samples
                values[index] = (epoch, int(mapping[unit])) in eligible
            mask[first:stop] = values.astype(np.uint8)
            kept += int(np.count_nonzero(values))
        view.attrs["kept_spikes"] = kept
        view.attrs["eligible_unit_epochs"] = len(eligible)
        view.attrs["complete"] = True
        view.flush()
    current_stat = source_path.stat()
    if (current_stat.st_size != stat.st_size or
            current_stat.st_mtime_ns != stat.st_mtime_ns):
        raise ValueError(f"Source shard changed while building view: {source_path}")
    os.replace(partial, destination)
    return kept


def read_provisional_view_mask(view_root, probe_id, source_path, source, *, rows):
    """Read a verified support mask for benchmark-sized quality evaluation."""
    root = Path(view_root)
    config = json.loads((root / "view.json").read_text(encoding="utf-8"))
    source_path = Path(source_path)
    if config != _config(source_path.parent.parent):
        raise ValueError("Provisional view configuration no longer matches source")
    destination = root / probe_id / source_path.name
    if not _valid_view(destination, source_path, source, _hash_config(config), rows):
        raise ValueError(f"Missing, incomplete, or stale provisional view: {destination}")
    with h5py.File(destination, "r") as view:
        values = view["spike_mask"][:]
    if np.any(values > 1):
        raise ValueError("Invalid provisional view mask")
    return values.astype(bool)


def build_provisional_support_view(source_root, output_root, *,
                                   probe_id=None, resume=False):
    """Write immutable per-shard masks with bounded reads and exact resume."""
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if source_root == output_root:
        raise ValueError("The view output must differ from the source session")
    config = _config(source_root)
    if probe_id is not None and (probe_id not in config["sample_rates_hz"] or
                                 Path(probe_id).name != probe_id):
        raise ValueError("Unknown or invalid probe ID")
    config_hash = _hash_config(config)
    manifest = output_root / "view.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(output_root)
        if not manifest.is_file():
            pending = output_root / "view.json.partial"
            if pending.is_file() and json.loads(
                    pending.read_text(encoding="utf-8")) == config:
                os.replace(pending, manifest)
            elif not any(output_root.iterdir()):
                pending.write_text(json.dumps(config, indent=2) + "\n",
                                   encoding="utf-8")
                os.replace(pending, manifest)
            else:
                raise ValueError("Incomplete or unrelated provisional view")
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing provisional view has different settings")
    else:
        output_root.mkdir(parents=True)
        pending = output_root / "view.json.partial"
        pending.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(pending, manifest)
    probes = ([probe_id] if probe_id else sorted(config["sample_rates_hz"]))
    completed = skipped = rows_seen = rows_kept = 0
    for probe in probes:
        paths = sorted((source_root / probe).glob("*.h5"))
        if not paths:
            raise ValueError(f"No completed shards for probe {probe}")
        destination_dir = output_root / probe
        destination_dir.mkdir(exist_ok=True)
        for source_path in paths:
            destination = destination_dir / source_path.name
            partial = destination.with_name(destination.name + ".partial")
            with h5py.File(source_path, "r") as source:
                count = _check_shard(source, source_path)
                if str(source.attrs.get("probe_id", probe)) != probe:
                    raise ValueError(f"Source shard has wrong probe ID: {source_path}")
                _spans(source, config, probe)
                rows_seen += count
                if _valid_view(destination, source_path, source, config_hash, count):
                    with h5py.File(destination, "r") as view:
                        rows_kept += int(view.attrs["kept_spikes"])
                    skipped += 1
                    continue
                if destination.exists():
                    raise ValueError(f"Stale provisional view: {destination}")
                if partial.exists():
                    if not resume:
                        raise FileExistsError(partial)
                    partial.unlink()
                rows_kept += _write_shard(source_path, destination, source,
                                          config, config_hash, probe, count)
                completed += 1
    return {"output_root": str(output_root), "shards_written": completed,
            "shards_skipped": skipped, "spikes_seen": rows_seen,
            "spikes_kept": rows_kept}
