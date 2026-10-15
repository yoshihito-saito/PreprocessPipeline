"""Bounded, non-destructive output views for completed session shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


SCHEMA_VERSION = 1
BLOCK_ROWS = 65_536


def _view_config(source_root: Path, amplitude_min: float) -> dict:
    run = source_root / "run.json"
    if not run.is_file():
        raise ValueError(f"Missing completed-session configuration: {run}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(source_root.resolve()),
        "source_run_sha256": hashlib.sha256(run.read_bytes()).hexdigest(),
        "amplitude_min": float(amplitude_min),
        "predicate": "fitted_amplitude >= amplitude_min",
    }


def _check_shard(source: h5py.File, path: Path) -> int:
    if not bool(source.attrs.get("complete", False)):
        raise ValueError(f"Incomplete source shard: {path}")
    if "spikes" not in source or "fitted_amplitude" not in source["spikes"].dtype.names:
        raise ValueError(f"Source shard lacks fitted amplitudes: {path}")
    return len(source["spikes"])


def _valid_view(view_path: Path, source_path: Path, source: h5py.File,
                config: dict, rows: int) -> bool:
    if not view_path.is_file():
        return False
    source_stat = source_path.stat()
    with h5py.File(view_path, "r") as view:
        return (
            bool(view.attrs.get("complete", False))
            and int(view.attrs.get("schema_version", -1)) == SCHEMA_VERSION
            and str(view.attrs.get("source_shard", "")) == source_path.name
            and str(view.attrs.get("source_run_sha256", "")) == config["source_run_sha256"]
            and float(view.attrs.get("amplitude_min", -1)) == config["amplitude_min"]
            and int(view.attrs.get("source_start_sample", -1)) == int(source.attrs["start_sample"])
            and int(view.attrs.get("source_stop_sample", -1)) == int(source.attrs["stop_sample"])
            and int(view.attrs.get("source_size_bytes", -1)) == source_stat.st_size
            and int(view.attrs.get("source_mtime_ns", -1)) == source_stat.st_mtime_ns
            and 0 <= int(view.attrs.get("kept_spikes", -1)) <= rows
            and "spike_mask" in view and view["spike_mask"].shape == (rows,)
        )


def read_view_mask(view_root, probe_id, source_path, source, *, rows=None):
    """Return a verified per-shard view mask for benchmark-sized reads."""
    view_root = Path(view_root)
    config = json.loads((view_root / "view.json").read_text(encoding="utf-8"))
    if config.get("algorithm") == "cofire_dedup_v1":
        from .session_cofire_view import read_cofire_view_mask

        return read_cofire_view_mask(view_root, probe_id, source_path, source,
                                     rows=len(source["spikes"]) if rows is None else rows)
    if config.get("algorithm") == "provisional_temporal_support_v1":
        from .session_provisional_view import read_provisional_view_mask

        return read_provisional_view_mask(
            view_root, probe_id, source_path, source,
            rows=len(source["spikes"]) if rows is None else rows)
    source_path = Path(source_path)
    source_root = source_path.parent.parent
    expected = _view_config(source_root, config["amplitude_min"])
    if config != expected:
        raise ValueError(f"View configuration does not match source session: {view_root}")
    count = _check_shard(source, source_path) if rows is None else rows
    view_path = view_root / probe_id / source_path.name
    if not _valid_view(view_path, source_path, source, config, count):
        raise ValueError(f"Missing, incomplete, or stale session view: {view_path}")
    with h5py.File(view_path, "r") as view:
        mask = view["spike_mask"][:]
    if np.any(mask > 1):
        raise ValueError(f"Invalid session view mask: {view_path}")
    return mask.astype(bool)


def build_session_view(source_root, output_root, *, amplitude_min, resume=False,
                       probe_id=None):
    """Write compact immutable masks, one per source shard, with bounded RAM.

    The original shards remain untouched. Resume skips only complete masks
    whose source coordinates and run configuration match.
    """
    source_root, output_root = Path(source_root), Path(output_root)
    if not np.isfinite(amplitude_min) or amplitude_min < 0:
        raise ValueError("amplitude_min must be finite and nonnegative")
    if source_root.resolve() == output_root.resolve():
        raise ValueError("The view output must differ from the source session")
    if probe_id is not None and (not probe_id or Path(probe_id).name != probe_id
                                 or "/" in probe_id or "\\" in probe_id):
        raise ValueError("probe_id must be a single probe directory name")
    config = _view_config(source_root, float(amplitude_min))
    manifest = output_root / "view.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"Output view exists; pass resume for a matching view: {output_root}")
        if not manifest.is_file():
            pending = output_root / "view.json.partial"
            if pending.is_file() and json.loads(pending.read_text(encoding="utf-8")) == config:
                os.replace(pending, manifest)
            elif not any(output_root.iterdir()):
                pending.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                os.replace(pending, manifest)
            else:
                raise ValueError(f"Incomplete or unrelated output view: {output_root}")
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Existing output view has different settings: {output_root}")
    else:
        output_root.mkdir(parents=True)
        pending = output_root / "view.json.partial"
        pending.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(pending, manifest)
    probes = [source_root / probe_id] if probe_id else sorted(
        path for path in source_root.iterdir() if path.is_dir()
        and any(path.glob("*.h5")))
    if not probes:
        raise ValueError(f"No source probes with completed shards: {source_root}")
    completed = skipped = rows_seen = rows_kept = 0
    for probe in probes:
        if not probe.is_dir():
            raise ValueError(f"Missing source probe: {probe}")
        paths = sorted(probe.glob("*.h5"))
        if not paths:
            raise ValueError(f"No completed source shards: {probe}")
        destination = output_root / probe.name
        destination.mkdir(exist_ok=True)
        for source_path in paths:
            final = destination / source_path.name
            partial = destination / (source_path.name + ".partial")
            with h5py.File(source_path, "r") as source:
                count = _check_shard(source, source_path)
                rows_seen += count
                if _valid_view(final, source_path, source, config, count):
                    with h5py.File(final, "r") as view:
                        rows_kept += int(view.attrs["kept_spikes"])
                    skipped += 1
                    continue
                if final.exists():
                    raise ValueError(f"Existing view does not match source shard: {final}")
                kept = 0
                with h5py.File(partial, "w") as view:
                    source_stat = source_path.stat()
                    view.attrs["schema_version"] = SCHEMA_VERSION
                    view.attrs["source_shard"] = source_path.name
                    view.attrs["source_run_sha256"] = config["source_run_sha256"]
                    view.attrs["amplitude_min"] = config["amplitude_min"]
                    view.attrs["source_start_sample"] = int(source.attrs["start_sample"])
                    view.attrs["source_stop_sample"] = int(source.attrs["stop_sample"])
                    view.attrs["source_size_bytes"] = source_stat.st_size
                    view.attrs["source_mtime_ns"] = source_stat.st_mtime_ns
                    if count:
                        mask = view.create_dataset(
                            "spike_mask", shape=(count,), dtype="u1",
                            chunks=(min(BLOCK_ROWS, count),),
                            compression="lzf", shuffle=True)
                    else:
                        mask = view.create_dataset("spike_mask", shape=(0,), dtype="u1")
                    spikes = source["spikes"]
                    for first in range(0, count, BLOCK_ROWS):
                        stop = min(count, first + BLOCK_ROWS)
                        values = (spikes[first:stop]["fitted_amplitude"] >= amplitude_min)
                        mask[first:stop] = values.astype(np.uint8)
                        kept += int(np.count_nonzero(values))
                    view.attrs["kept_spikes"] = kept
                    view.attrs["complete"] = True
                    view.flush()
                os.replace(partial, final)
                completed += 1
                rows_kept += kept
    return {"output_root": str(output_root), "shards_written": completed,
            "shards_skipped": skipped, "spikes_seen": rows_seen,
            "spikes_kept": rows_kept, "amplitude_min": float(amplitude_min)}
