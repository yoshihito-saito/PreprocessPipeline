from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from .models import AnalysisConfig, StageName, StageStatus
from .store import (
    RunStore,
    atomic_write_json,
    atomic_write_text,
    read_json,
    short_file_lock,
    utc_now,
)


_ACTIVE_RUN_STATES = {"pending", "submitted", "running", "lost"}
_MUTATING_STAGE_STATES = {"submitted", "running", "lost"}
_PHY_REQUIRED = ("params.py", "spike_times.npy", "spike_clusters.npy", "templates.npy")
_POST_REQUIRED = (
    "params.py",
    "spike_times.npy",
    "spike_clusters.npy",
    "cluster_group.tsv",
    "cluster_info.tsv",
    "quality_metrics.csv",
)

# These identifiers deliberately describe the producer/output contract, rather
# than the checkout state.  A dirty unrelated GUI edit must not invalidate a
# scientifically compatible output, while a deliberate producer-format change
# has an explicit way to do so.
OUTPUT_INVENTORY_SCHEMA = "execution-output-inventory-v1"
PRODUCER_SCHEMAS = {
    StageName.PREPROCESS.value: "preprocess-output-v2",
    StageName.SORTING.value: "sorting-output-v1",
    StageName.POSTPROCESS.value: "postprocess-output-v1",
}


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stage_fingerprints(analysis: Any) -> dict[str, str]:
    settings = analysis.settings if hasattr(analysis, "settings") else dict(analysis["settings"])
    artifacts = (
        analysis.artifact_sha256
        if hasattr(analysis, "artifact_sha256")
        else dict(analysis.get("artifact_sha256", {}))
    )
    preprocess = dict(settings.get("preprocess", {}))
    sorting_keys = {
        "run_sorter",
        "sorter",
        "sorter_path",
        "sorter_config_path",
        "sorter_partition_mode",
        "sorter_verbose",
        "cleanup_temp_wh",
    }
    runtime_keys = {"overwrite", "preprocess_worker_count", "sorter_worker_count", "matlab_path"}
    preprocess_payload = {
        key: value
        for key, value in preprocess.items()
        if key not in sorting_keys | runtime_keys
    }
    preprocess_payload["multi_day_enabled"] = settings.get("multi_day_enabled", False)
    preprocess_payload["multi_day_session_paths"] = settings.get("multi_day_session_paths", [])
    preprocess_payload["multi_day_selected_subepoch_paths"] = settings.get(
        "multi_day_selected_subepoch_paths", []
    )
    preprocess_payload["xml_sha256"] = artifacts.get("xml", "")
    sorting_payload = {
        key: preprocess.get(key)
        for key in sorted(sorting_keys)
        if key in preprocess
    }
    sorting_payload["sorter_config_sha256"] = artifacts.get("sorter_config", "")
    postprocess = dict(settings.get("postprocess", {}))
    for key in (
        "overwrite",
        "worker_count",
        "cell_explorer_sorting_folders",
    ):
        postprocess.pop(key, None)
    postprocess["chanmap_sha256"] = artifacts.get("chanmap", "")
    postprocess["xml_sha256"] = artifacts.get("xml", "")
    postprocess["sorting_input_identity"] = artifacts.get(
        "postprocess_sorting_input", ""
    )
    postprocess["dat_input_identity"] = artifacts.get("postprocess_dat_input", "")
    return {
        StageName.PREPROCESS.value: _hash_payload(preprocess_payload),
        StageName.SORTING.value: _hash_payload(sorting_payload),
        StageName.POSTPROCESS.value: _hash_payload(postprocess),
    }


def _prior_stage_fingerprints(session_dir: Path) -> dict[str, str] | None:
    path = session_dir / "preprocess_run.yaml"
    if not path.exists():
        return None
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        recorded = payload.get("stage_fingerprints")
        if isinstance(recorded, dict):
            return {str(key): str(value) for key, value in recorded.items()}
        analysis = payload.get("analysis")
        if isinstance(analysis, dict):
            return stage_fingerprints(analysis)
    except (OSError, TypeError, ValueError):
        return None
    return None


def _persistent_input_provenance_compatible(
    session_dir: Path, store: RunStore, settings: Any
) -> bool:
    prior_path = session_dir / "preprocess_run.yaml"
    current_path = store.run_dir / "snapshots" / "inputs.json"
    if not prior_path.exists() or not current_path.exists():
        return False
    try:
        import yaml

        prior_record = yaml.safe_load(prior_path.read_text(encoding="utf-8")) or {}
        prior_snapshot = (prior_record.get("provenance") or {}).get("inputs") or {}
        current_snapshot = read_json(current_path)
        return _input_snapshots_compatible(
            prior_snapshot,
            current_snapshot,
            settings,
            prior_cutoff_at=prior_record.get("started_at") or prior_record.get("created_at"),
        )
    except (OSError, TypeError, ValueError):
        return False


def _input_snapshots_compatible(
    prior_snapshot: dict[str, Any],
    current_snapshot: dict[str, Any],
    settings: Any,
    *,
    prior_cutoff_at: str | None = None,
) -> bool:
    prior_entries = {
        str(item.get("path")): item for item in prior_snapshot.get("inputs", [])
    }
    current_entries = {
        str(item.get("path")): item for item in current_snapshot.get("inputs", [])
    }
    from .controller import _input_provenance, _recursive_input_roots

    relevant = _recursive_input_roots(settings)
    relevant_roots = {str(Path(path).resolve()) for path in relevant if path is not None}
    if not relevant_roots:
        return False

    authoritative_xml = settings.resolved_xml_path()
    authoritative_xml = Path(authoritative_xml).resolve() if authoritative_xml else None

    def unused_multiday_xml(path: str) -> bool:
        # Explicit XML supplies all Intan channel/rate metadata during staging.
        # Per-recording amplifier XML is not read in that workflow.
        return bool(
            getattr(settings, "multi_day_enabled", False)
            and authoritative_xml is not None
            and Path(path).name.lower() == "amplifier.xml"
            and Path(path).resolve() != authoritative_xml
        )

    def same_content(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return bool(
            left.get("is_dir") is False and right.get("is_dir") is False
            and left.get("exists") is True and right.get("exists") is True
            and left.get("size") == right.get("size")
            and left.get("sha256") and left.get("sha256") == right.get("sha256")
        )

    def relevant_entries(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(item.get("path")): item
            for item in snapshot.get("inputs", [])
            if not unused_multiday_xml(str(item.get("path"))) and any(
                str(item.get("path")) == root
                or str(item.get("path")).startswith(root + os.sep)
                for root in relevant_roots
            )
        }

    def timestamp_ns(value: Any) -> int | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1_000_000_000)
        except (OverflowError, TypeError, ValueError):
            return None

    # The immutable current snapshot is captured during Run creation.  Re-scan
    # immediately before trusting it so an input added, removed, or changed
    # while the Run waited for its worker cannot authorize stale output reuse.
    live_snapshot = _input_provenance(settings)
    stored_current = relevant_entries(current_snapshot)
    live_current = relevant_entries(live_snapshot)
    if not stored_current or stored_current.keys() != live_current.keys():
        return False
    current_cutoff_ns = timestamp_ns(
        current_snapshot.get("scan_started_at")
        or current_snapshot.get("recorded_at")
    )
    live_ctimes: dict[str, int] = {}
    metadata_keys = ("exists", "size", "mtime_ns", "is_dir")
    for path, stored in stored_current.items():
        live = live_current[path]
        if same_content(stored, live):
            live_ctimes[path] = live["ctime_ns"]
            continue
        if stored.get("sha256") and stored.get("sha256") != live.get("sha256"):
            return False
        if any(stored.get(key) != live.get(key) for key in metadata_keys):
            return False
        live_ctime = live.get("ctime_ns")
        if not isinstance(live_ctime, int):
            return False
        stored_ctime = stored.get("ctime_ns")
        if isinstance(stored_ctime, int):
            if stored_ctime != live_ctime:
                return False
        elif current_cutoff_ns is None or live_ctime > current_cutoff_ns:
            return False
        live_ctimes[path] = live_ctime

    relevant_paths = set(stored_current)
    prior_relevant_paths = set(relevant_entries(prior_snapshot))
    if not prior_relevant_paths or not prior_relevant_paths.issubset(relevant_paths):
        return False
    if not all(
        path in prior_entries
        and path in current_entries
        and (same_content(prior_entries[path], current_entries[path]) or all(
            prior_entries[path].get(key) == current_entries[path].get(key)
            for key in metadata_keys
        ))
        and (not prior_entries[path].get("sha256") or prior_entries[path].get("sha256") == current_entries[path].get("sha256"))
        for path in prior_relevant_paths
    ):
        return False

    newly_tracked_paths = relevant_paths - prior_relevant_paths
    # Input provenance was expanded after Persistent Runs were already in use.
    # The immutable prior Run/scan-start cutoff proves that a path omitted by a
    # legacy schema—and any legacy entry lacking ctime—already existed in its
    # current form before that Run began.
    recorded_at = (
        prior_cutoff_at
        or prior_snapshot.get("scan_started_at")
    )
    recorded_ns = timestamp_ns(recorded_at)
    current_recorded_ns = timestamp_ns(
        current_snapshot.get("scan_started_at")
        or current_snapshot.get("recorded_at")
    )
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    needs_legacy_cutoff = bool(newly_tracked_paths) or any(
        not isinstance(prior_entries[path].get("ctime_ns"), int)
        for path in prior_relevant_paths
    )
    if needs_legacy_cutoff and (
        recorded_ns is None
        or current_recorded_ns is None
        or recorded_ns > current_recorded_ns
        or recorded_ns > now_ns
    ):
        return False
    for path in prior_relevant_paths:
        if same_content(prior_entries[path], current_entries[path]):
            continue
        prior_ctime = prior_entries[path].get("ctime_ns")
        if isinstance(prior_ctime, int):
            if live_ctimes[path] != prior_ctime:
                return False
        elif recorded_ns is None or live_ctimes[path] > recorded_ns:
            return False
    for path in newly_tracked_paths:
        entry = current_entries[path]
        if (
            entry.get("exists") is not True
            or entry.get("is_dir") is not False
            or not isinstance(entry.get("mtime_ns"), int)
            or entry["mtime_ns"] > recorded_ns
        ):
            return False
        if live_ctimes[path] > recorded_ns:
            return False
    return True


def _legacy_preprocess_compatible(session_dir: Path, settings: Any) -> bool:
    path = session_dir / "preprocessSession_params.json"
    if not path.exists():
        return False
    try:
        prior = read_json(path)
        current = _json_safe(asdict(settings.to_preprocess_config()))
    except (OSError, TypeError, ValueError):
        return False
    excluded = {
        "basepath",
        "localpath",
        "output_dir",
        "chanmap_mat_path",
        "xml_path",
        "sorter",
        "sorter_path",
        "sorter_config_path",
        "sorter_partition_mode",
        "matlab_path",
        "matlab_max_workers",
        "sorter_verbose",
        "cleanup_temp_wh",
        "highamp_n_jobs",
        "overwrite",
        "job_kwargs",
        "save_params_json",
        "save_manifest_json",
        "save_log_mat",
    }
    comparable_keys = set(current) - excluded
    if not comparable_keys.issubset(prior):
        return False
    return all(_json_safe(prior[key]) == _json_safe(current[key]) for key in comparable_keys)


def _legacy_sorting_compatible(session_dir: Path, store: RunStore, settings: Any) -> bool:
    expected = store.load_analysis().artifact_sha256.get("sorter_config")
    params_path = session_dir / "preprocessSession_params.json"
    candidates = [
        path
        for output in session_dir.glob("Kilosort*")
        if output.is_dir()
        and "_spi" not in output.name
        and ".preserved-" not in output.name
        for path in output.glob("sorter_config_source.*")
        if path.is_file()
    ]
    if not expected or not candidates or not params_path.exists():
        return False
    try:
        prior = read_json(params_path)
        current = _json_safe(asdict(settings.to_preprocess_config()))
        for key in ("sorter", "sorter_partition_mode"):
            if key not in prior or _json_safe(prior[key]) != current.get(key):
                return False
        return any(hashlib.sha256(path.read_bytes()).hexdigest() == expected for path in candidates)
    except (OSError, TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _inventory_entry(path: Path, role: str) -> dict[str, str]:
    return {"path": str(path.expanduser().resolve()), "role": role}


def preprocess_output_inventory(result: Any, config: Any) -> dict[str, Any]:
    """Return every preprocess product requested by this immutable config.

    The inventory is intentionally path-oriented: it is both the completion
    checkpoint and the evidence that a later Run must revalidate before reuse.
    """
    output_dir = Path(result.local_output_dir)
    basename = str(result.basename)
    entries: list[dict[str, Any]] = []

    def add(
        path: Path | None,
        role: str,
        *,
        required: bool = True,
        binary: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if path is None:
            if required:
                entries.append({"path": "", "role": role})
            return None
        entry = _inventory_entry(Path(path), role)
        if binary is not None:
            entry["binary"] = binary
        entries.append(entry)
        return entry

    add(result.dat_path, "dat")
    add(result.session_mat_path, "session_mat")
    add(result.mergepoints_mat_path, "mergepoints_mat")
    add(output_dir / f"{basename}.xml", "xml")
    add(output_dir / "chanMap.mat", "chanmap")
    if bool(getattr(config, "save_raw", False)):
        add(output_dir / f"{basename}_raw.dat", "raw_dat")
    copied_rhd = output_dir / f"{basename}.rhd"
    if copied_rhd.is_file():
        add(copied_rhd, "rhd")
    if bool(getattr(config, "make_lfp", False)):
        add(result.lfp_path, "lfp")
    for index, path in enumerate(result.analog_event_paths):
        add(path, f"analog_event_{index:03d}")
    for index, path in enumerate(result.digital_event_paths):
        add(path, f"digital_event_{index:03d}")
    for name, path in sorted(result.intermediate_dat_paths.items()):
        path = Path(path)
        if path.suffix.lower() != ".dat":
            add(path, f"intermediate_{name}")
            continue
        layout_path = path.with_name(f"{path.name}.layout.json")
        fallback_channels = 1
        if name == "analogin":
            fallback_channels = max(1, len(getattr(config, "source_adc_channels", []) or []))
        binary: dict[str, Any] = {
            "dtype": str(getattr(config, "dtype", "int16")),
            "n_channels": fallback_channels,
        }
        if layout_path.is_file():
            try:
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
                binary.update(
                    {
                        "dtype": str(layout.get("dtype") or binary["dtype"]),
                        "n_channels": int(layout["num_channels"]),
                        "expected_frames": (
                            int(sum(int(value) for value in layout["sample_counts"]))
                            if isinstance(layout.get("sample_counts"), list)
                            else None
                        ),
                    }
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # The JSON entry below makes a malformed layout fail during
                # completion rather than silently weakening binary validation.
                pass
            layout_entry = add(layout_path, f"intermediate_{name}_layout")
            assert layout_entry is not None
            layout_entry["sha256"] = hashlib.sha256(layout_path.read_bytes()).hexdigest()
        add(path, f"intermediate_{name}", binary=binary)
    chan_coords = output_dir / f"{basename}.chanCoords.channelInfo.mat"
    if chan_coords.is_file():
        add(chan_coords, "chancoords")
    if bool(getattr(config, "state_score", False)):
        # State scoring has a fixed dependency graph.  Derive the requested
        # leaves from configuration (rather than merely trusting a partial
        # returned list), while respecting the optional SleepScoreLFP MAT.
        from src.preprocess.state_scoring import _expected_state_score_outputs

        *_, expected_state = _expected_state_score_outputs(
            basepath=output_dir,
            basename=basename,
            save_lfp_mat=bool(getattr(config, "state_save_lfp_mat", True)),
        )
        for index, path in enumerate(expected_state):
            add(path, f"state_score_{index:03d}")
    chanmap_path = output_dir / "chanMap.mat"
    layout_sha256 = ""
    if chanmap_path.is_file():
        layout_sha256 = hashlib.sha256(chanmap_path.read_bytes()).hexdigest()
    return {
        "schema": OUTPUT_INVENTORY_SCHEMA,
        "producer_schema": PRODUCER_SCHEMAS[StageName.PREPROCESS.value],
        "metadata": {
            "n_channels": int(result.n_channels),
            "dtype": str(getattr(config, "dtype", "int16")),
            "dat_frames": int(sum(int(value) for value in result.subsession_sample_counts)),
            # LFP is resampled, so its frame count is not the raw-data count;
            # it must still consist of complete channel frames.
            "lfp_n_channels": int(result.n_channels),
            "channel_layout_sha256": layout_sha256,
        },
        "entries": entries,
    }


def validate_output_inventory(inventory: dict[str, Any], *, label: str) -> list[str]:
    if inventory.get("schema") != OUTPUT_INVENTORY_SCHEMA:
        raise RuntimeError(f"{label} has an unsupported output inventory schema")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{label} has no recorded output inventory")
    metadata = inventory.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise RuntimeError(f"{label} has invalid output inventory metadata")
    metadata = metadata or {}
    validated: list[str] = []

    def fail(role: str, path: Path, detail: str) -> None:
        raise RuntimeError(f"{label} validation failed for {role}: {path} ({detail})")

    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(f"{label} has an invalid output inventory entry")
        path_text = str(entry.get("path") or "")
        role = str(entry.get("role") or "output")
        path = Path(path_text) if path_text else None
        if path is None or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{label} validation failed for {role}: {path_text or '<missing path>'}")
        expected_sha256 = str(entry.get("sha256") or "")
        if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            fail(role, path, "content fingerprint changed")
        suffix = path.suffix.lower()
        binary = entry.get("binary") if isinstance(entry.get("binary"), dict) else None
        if role in {"dat", "raw_dat", "lfp"} or binary is not None:
            try:
                if binary is not None:
                    n_channels = int(binary.get("n_channels") or 1)
                    itemsize = np.dtype(str(binary["dtype"])).itemsize
                    expected_frames = binary.get("expected_frames")
                else:
                    n_channels = int(metadata["n_channels" if role in {"dat", "raw_dat"} else "lfp_n_channels"])
                    itemsize = np.dtype(str(metadata["dtype"])).itemsize
                    expected_frames = metadata.get("dat_frames") if role in {"dat", "raw_dat"} else None
            except (KeyError, TypeError, ValueError) as exc:
                # Older inventories had no shape metadata; retain their
                # compatibility path while current inventories fail closed.
                if metadata:
                    fail(role, path, "missing binary shape metadata")
                n_channels = 0
                itemsize = 0
            if n_channels and itemsize:
                frame_bytes = n_channels * itemsize
                if path.stat().st_size % frame_bytes:
                    fail(role, path, f"size is not aligned to {frame_bytes}-byte frames")
                if expected_frames is not None:
                    if path.stat().st_size // frame_bytes != int(expected_frames):
                        fail(role, path, "frame count differs from merged subsessions")
        elif suffix == ".xml":
            try:
                ET.parse(path)
            except (OSError, ET.ParseError) as exc:
                fail(role, path, f"invalid XML: {exc}")
        elif suffix == ".mat":
            try:
                from src.preprocess.io import validate_mat_output
            except Exception as exc:
                fail(role, path, f"invalid MAT payload: {exc}")
            required_key = {
                "session_mat": "session",
                "mergepoints_mat": "MergePoints",
                "chanmap": "chanMap",
            }.get(role)
            name = path.name
            if required_key is None:
                if ".EMGFromLFP." in name:
                    required_key = "EMGFromLFP"
                elif ".SleepScoreLFP." in name:
                    required_key = "SleepScoreLFP"
                elif ".SleepStateEpisodes." in name:
                    required_key = "SleepStateEpisodes"
                elif ".SleepState." in name:
                    required_key = "SleepState"
                elif ".chanCoords." in name:
                    required_key = "chanCoords"
                elif ".artifactTTL.events." in name:
                    required_key = "artifactTTL"
                elif ".artifactHigh.events." in name:
                    required_key = "artifactHigh"
            if required_key is not None:
                try:
                    validate_mat_output(path, required_key, load_payload=False)
                except ValueError as exc:
                    fail(role, path, f"invalid MAT payload: {exc}")
            else:
                try:
                    from scipy.io import loadmat

                    loaded = loadmat(path, simplify_cells=True)
                except Exception as exc:
                    fail(role, path, f"invalid MAT payload: {exc}")
                if not any(not str(key).startswith("__") for key in loaded):
                    fail(role, path, "MAT payload has no public data")
        elif suffix == ".npy":
            try:
                np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                fail(role, path, f"invalid NumPy array: {exc}")
        elif suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                fail(role, path, f"invalid JSON: {exc}")
            if role.endswith("_layout"):
                try:
                    if int(payload["schema_version"]) != 1:
                        raise ValueError("unsupported schema_version")
                    np.dtype(str(payload["dtype"]))
                    n_channels = int(payload["num_channels"])
                    if n_channels <= 0:
                        raise ValueError("num_channels must be positive")
                    sampling_frequency = payload.get("sampling_frequency")
                    if sampling_frequency is not None and float(sampling_frequency) <= 0:
                        raise ValueError("sampling_frequency must be positive")
                    sample_counts = payload["sample_counts"]
                    if not isinstance(sample_counts, list) or any(
                        int(value) < 0 for value in sample_counts
                    ):
                        raise ValueError("sample_counts must be a nonnegative list")
                    n_epochs = len(sample_counts)
                    for field in (
                        "source_sample_counts",
                        "source_sampling_frequencies",
                        "source_num_channels",
                        "source_channel_indices",
                        "destination_channel_indices",
                    ):
                        values = payload.get(field)
                        if values is not None and (
                            not isinstance(values, list) or len(values) != n_epochs
                        ):
                            raise ValueError(f"{field} must align with sample_counts")
                    destinations = payload.get("destination_channel_indices") or []
                    sources = payload.get("source_channel_indices") or []
                    for index, mapping in enumerate(destinations):
                        if mapping is None:
                            continue
                        mapped = [int(value) for value in mapping]
                        if len(set(mapped)) != len(mapped) or any(
                            value < 0 or value >= n_channels for value in mapped
                        ):
                            raise ValueError("destination mapping is outside output layout")
                        if index < len(sources) and sources[index] is not None:
                            if len(mapped) != len(sources[index]):
                                raise ValueError("source/destination mapping lengths differ")
                except (KeyError, TypeError, ValueError) as exc:
                    fail(role, path, f"invalid sidecar layout: {exc}")
        elif suffix in {".jpg", ".jpeg"}:
            header = path.read_bytes()[:2]
            tail = b""
            with path.open("rb") as handle:
                handle.seek(-2, os.SEEK_END)
                tail = handle.read(2)
            if header != b"\xff\xd8" or tail != b"\xff\xd9":
                fail(role, path, "invalid JPEG signature")
        if role == "chanmap" and metadata.get("channel_layout_sha256"):
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != metadata["channel_layout_sha256"]:
                fail(role, path, "channel-layout fingerprint changed")
        validated.append(str(path.resolve()))
    return validated


def settings_for_store(store: RunStore):
    from src.preprocess.gui.config_model import PipelineGuiSettings

    return PipelineGuiSettings.from_json(json.dumps(store.load_analysis().settings))


def session_output_dir(settings: Any) -> Path:
    output = settings.local_output_dir
    if output is None:
        raise ValueError("The session output directory cannot be resolved")
    return Path(output).expanduser().resolve()


_BINARY_PARTIAL_TOKEN = re.compile(r"^(?:[0-9a-fA-F]{12}|[0-9a-fA-F]{32})$")


def cleanup_preprocess_binary_partials(
    session_dir: Path | str,
    basename: str,
) -> dict[str, Any]:
    """Remove producer-owned incomplete binary outputs from one session root.

    This intentionally handles only binary files whose canonical targets are
    owned by preprocessing.  It never recurses and never removes a canonical
    output, so unrelated user files and completed data remain untouched.
    """

    root = Path(session_dir).expanduser().resolve()
    clean_basename = str(basename).strip()
    if not clean_basename or Path(clean_basename).name != clean_basename:
        raise ValueError(f"Invalid preprocess basename for partial cleanup: {basename!r}")
    if not root.exists():
        return {"removed": [], "removed_bytes": 0, "errors": []}
    if not root.is_dir():
        raise NotADirectoryError(f"Preprocess session output is not a directory: {root}")

    target_names = {
        f"{clean_basename}.dat",
        f"{clean_basename}_raw.dat",
        f"{clean_basename}.lfp",
        "analogin.dat",
        "digitalin.dat",
        "auxiliary.dat",
        "supply.dat",
        "time.dat",
    }

    def owned_partial(name: str) -> bool:
        folded = name.casefold()
        for target in target_names:
            target_folded = target.casefold()
            for prefix in (
                f"{target_folded}.partial-",
                f".{target_folded}.partial-",
            ):
                if folded.startswith(prefix):
                    return _BINARY_PARTIAL_TOKEN.fullmatch(name[len(prefix) :]) is not None
        return False

    removed: list[str] = []
    errors: list[dict[str, str]] = []
    removed_bytes = 0
    for candidate in root.iterdir():
        if not owned_partial(candidate.name):
            continue
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        try:
            size = int(candidate.lstat().st_size)
            candidate.unlink()
        except OSError as exc:
            errors.append({"path": str(candidate), "error": str(exc)})
        else:
            removed.append(str(candidate))
            removed_bytes += size
    return {
        "removed": removed,
        "removed_bytes": removed_bytes,
        "errors": errors,
    }


_PREPROCESS_CONTRACT_NAME = ".preprocess-output-contract.json"


def _preprocess_contract_identity(store: RunStore, session_dir: Path) -> dict[str, str]:
    try:
        inputs = read_json(store.run_dir / "snapshots" / "inputs.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Missing immutable acquisition provenance for preprocess output contract") from exc
    return {
        "schema": "preprocess-output-contract-v1",
        "producer_schema": PRODUCER_SCHEMAS[StageName.PREPROCESS.value],
        "session_dir": str(session_dir.resolve()),
        "stage_fingerprint": str(store.load_run().get("stage_fingerprints", {}).get("preprocess") or ""),
        "input_provenance_sha256": _hash_payload({"inputs": inputs.get("inputs", [])}),
    }


def _known_preprocess_outputs(session_dir: Path, basename: str) -> list[Path]:
    names = (
        f"{basename}.dat", f"{basename}.lfp", f"{basename}.session.mat",
        f"{basename}.MergePoints.events.mat", f"{basename}.xml",
        f"{basename}.chanCoords.channelInfo.mat", f"{basename}_raw.dat", f"{basename}.rhd",
        "chanMap.mat", "analogin.dat", "digitalin.dat", "auxiliary.dat", "supply.dat", "time.dat",
        "digitalIn.events.mat", "preprocessSession_manifest.json", "preprocessSession_params.json",
    )
    paths = [session_dir / name for name in names]
    # Explicit producer-owned MAT products only.  In particular, do not glob
    # every ``<basename>.*.mat``: CellExplorer/user behavior and cell-info
    # products share that convention but are outside preprocess authority.
    paths.extend(
        session_dir / name
        for name in (
            f"{basename}.analogInput.behavior.mat",
            f"{basename}.pulses.events.mat",
            f"{basename}.artifactTTL.events.mat",
            f"{basename}.artifactHigh.events.mat",
            f"{basename}.EMGFromLFP.LFP.mat",
            f"{basename}.SleepScoreLFP.LFP.mat",
            f"{basename}.SleepState.states.mat",
            f"{basename}.SleepStateEpisodes.states.mat",
        )
    )
    paths.extend(session_dir.glob("*DigitalIn.events.mat"))
    paths.extend(session_dir.glob("*.dat.layout.json"))
    for directory in ("StateScoreFigures", "pulses", "Pulses"):
        path = session_dir / directory
        if path.exists():
            paths.append(path)
    # A path may match both an explicit name and a glob.  Sorting largest
    # paths first prevents moving a child after its parent directory.
    unique = {path.resolve() for path in paths if path.exists()}
    return sorted(unique, key=lambda path: (len(path.parts), str(path)), reverse=True)


def _trusted_preprocess_evidence(
    store: RunStore,
    settings: Any,
    identity: dict[str, str],
    *,
    required_prior_input_sha256: str = "",
) -> bool:
    """Accept legacy partial outputs only when immutable prior evidence agrees."""
    session_dir = Path(identity["session_dir"])
    record_path = session_dir / "preprocess_run.yaml"
    if record_path.exists():
        try:
            import yaml

            record = yaml.safe_load(record_path.read_text(encoding="utf-8")) or {}
            if (record.get("stage_fingerprints") or {}).get("preprocess") != identity["stage_fingerprint"]:
                return False
            prior_inputs = ((record.get("provenance") or {}).get("inputs") or {})
            if required_prior_input_sha256 and _hash_payload(
                {"inputs": prior_inputs.get("inputs", [])}
            ) != required_prior_input_sha256:
                return False
            current_inputs = read_json(store.run_dir / "snapshots" / "inputs.json")
            return _input_snapshots_compatible(
                prior_inputs,
                current_inputs,
                settings,
                prior_cutoff_at=record.get("started_at") or record.get("created_at"),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    for previous in _previous_persistent_stores(store):
        try:
            previous_run = previous.load_run()
            if str(previous_run.get("stage_fingerprints", {}).get("preprocess") or "") != identity["stage_fingerprint"]:
                continue
            prior_inputs = read_json(previous.run_dir / "snapshots" / "inputs.json")
            if required_prior_input_sha256 and _hash_payload(
                {"inputs": prior_inputs.get("inputs", [])}
            ) != required_prior_input_sha256:
                continue
            current_inputs = read_json(store.run_dir / "snapshots" / "inputs.json")
            if _input_snapshots_compatible(
                prior_inputs,
                current_inputs,
                settings,
                prior_cutoff_at=previous_run.get("created_at"),
            ):
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return False


def prepare_preprocess_output_contract(
    store: RunStore, settings: Any, *, overwrite: bool
) -> Path:
    """Commit or verify the pre-mutation authority for canonical preprocess outputs."""
    session_dir = session_output_dir(settings)
    session_dir.mkdir(parents=True, exist_ok=True)
    identity = _preprocess_contract_identity(store, session_dir)
    contract_path = session_dir / _PREPROCESS_CONTRACT_NAME
    basename = str(getattr(settings, "basename", session_dir.name))
    existing_outputs = _known_preprocess_outputs(session_dir, basename)
    existing: dict[str, Any] | None = None
    if contract_path.exists():
        try:
            existing = read_json(contract_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if not overwrite:
                raise RuntimeError(f"Invalid preprocess output contract: {contract_path}") from exc
    if existing == identity:
        return contract_path
    if isinstance(existing, dict) and existing.get("producer_schema") == "legacy-preprocess-v1":
        legacy_identity = dict(existing)
        legacy_identity.pop("producer_schema", None)
        current_identity = dict(identity)
        current_identity.pop("producer_schema", None)
        if legacy_identity == current_identity:
            return contract_path
    if (
        isinstance(existing, dict)
        and existing.get("schema") == identity["schema"]
        and existing.get("session_dir") == identity["session_dir"]
        and existing.get("stage_fingerprint") == identity["stage_fingerprint"]
        and existing.get("producer_schema")
        in {"legacy-preprocess-v1", identity["producer_schema"]}
        and _trusted_preprocess_evidence(
            store,
            settings,
            identity,
            required_prior_input_sha256=str(
                existing.get("input_provenance_sha256") or ""
            ),
        )
    ):
        migrated_identity = dict(identity)
        migrated_identity["producer_schema"] = str(existing["producer_schema"])
        atomic_write_json(contract_path, migrated_identity)
        return contract_path
    if existing is None and existing_outputs and _trusted_preprocess_evidence(store, settings, identity):
        legacy_identity = dict(identity)
        legacy_identity["producer_schema"] = "legacy-preprocess-v1"
        atomic_write_json(contract_path, legacy_identity)
        return contract_path
    if existing is not None or existing_outputs:
        if not overwrite:
            raise RuntimeError(
                "Existing preprocess outputs have no compatible persistent output contract; "
                f"refusing mutation with overwrite=False: {session_dir}"
            )
        # Do not relabel stale output as current.  Preserve every known
        # canonical product in a recoverable sibling before publishing the new
        # contract; the producer can then create a coherent replacement set.
        backup = session_dir / ".preprocess-output-backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        backup.mkdir(parents=True, exist_ok=False)
        moved: list[tuple[Path, Path]] = []
        contract_publish_attempted = False
        try:
            for path in existing_outputs:
                target = backup / path.name
                shutil.move(str(path), str(target))
                moved.append((path, target))
            if contract_path.exists():
                target = backup / contract_path.name
                shutil.move(str(contract_path), str(target))
                moved.append((contract_path, target))
            # Contract publication is the commit record for this migration and
            # therefore belongs to the same rollback boundary as the moves.
            contract_publish_attempted = True
            atomic_write_json(contract_path, identity)
            return contract_path
        except BaseException:
            # Never leave a partially migrated canonical session behind.
            if contract_publish_attempted and contract_path.exists():
                contract_path.unlink()
            for original, moved_path in reversed(moved):
                if moved_path.exists():
                    shutil.move(str(moved_path), str(original))
            try:
                backup.rmdir()
                backup.parent.rmdir()
            except OSError:
                pass
            raise
    atomic_write_json(contract_path, identity)
    return contract_path


def _claim_path(pipeline_root: Path, session_dir: Path) -> Path:
    del pipeline_root
    return session_dir.resolve() / ".pipeline-active-run.json"


def _claim_blocks_new_run(claim: dict[str, Any]) -> bool:
    if claim.get("kind") == "manual":
        try:
            pid = int(claim.get("owner_pid") or 0)
            if pid <= 0:
                return True
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except (OSError, TypeError, ValueError):
            return True
    run_dir_text = str(claim.get("run_dir") or "").strip()
    if not run_dir_text:
        return True
    run_dir = Path(run_dir_text)
    if not (run_dir / "run.json").exists():
        created_text = str(claim.get("created_at") or "")
        try:
            created = datetime.fromisoformat(created_text)
            return (datetime.now(timezone.utc) - created).total_seconds() < 60.0
        except (TypeError, ValueError):
            return True
    try:
        store = RunStore(run_dir)
        state = store.derive_state()
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    for stage in StageName:
        for attempt in store.list_attempt_numbers(stage):
            submitted = store.read_attempt_fact(stage, attempt, "submitted.json")
            observation = store.latest_observation(stage, attempt) or {}
            backend_status = observation.get("status") or {}
            backend_conclusive_terminal = bool(
                backend_status.get("terminal", False)
            ) and (
                backend_status.get("successful") is not None
                or str(backend_status.get("state") or "").lower()
                in {"cancelled", "canceled"}
            )
            cancel_confirmed = (
                store.read_attempt_fact(stage, attempt, "cancel_confirmed.json")
                is not None
            )
            worker_terminal = any(
                store.read_attempt_fact(stage, attempt, name) is not None
                for name in ("result.json", "failure.json")
            )
            if submitted is not None:
                if (
                    not backend_conclusive_terminal
                    and not cancel_confirmed
                    and not worker_terminal
                ):
                    return True
            started = store.read_attempt_fact(stage, attempt, "started.json")
            if started is not None and not any(
                store.read_attempt_fact(stage, attempt, name) is not None
                for name in ("result.json", "failure.json", "cancel_confirmed.json")
            ) and not backend_conclusive_terminal:
                return True
            intent = store.read_attempt_fact(stage, attempt, "submission_intent.json")
            submission_failure = store.read_attempt_fact(
                stage, attempt, "submission_failure.json"
            )
            if intent is not None and submitted is None and submission_failure is None:
                return True
    if str(state.get("status", "unknown")) in _ACTIVE_RUN_STATES:
        return True
    return any(
        str(view.get("status", "unknown")) in _MUTATING_STAGE_STATES
        for view in (state.get("stages") or {}).values()
        if isinstance(view, dict) and view.get("enabled")
    )


def acquire_session_claim(
    *, pipeline_root: Path, session_dir: Path, run_id: str, run_dir: Path
) -> Path:
    claim_path = _claim_path(pipeline_root, session_dir)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        previous_run_dir = ""
        previous_run_dirs: list[str] = []
        if claim_path.exists():
            claim = read_json(claim_path)
            if _claim_blocks_new_run(claim):
                raise RuntimeError(
                    "Another persistent Run still owns this session output: "
                    f"{claim.get('run_id', 'unknown')} ({claim.get('run_dir', 'unknown')})"
                )
            if claim.get("kind", "run") == "run":
                previous_run_dir = str(claim.get("run_dir") or "")
                previous_run_dirs = [previous_run_dir] if previous_run_dir else []
                previous_run_dirs.extend(
                    str(value)
                    for value in claim.get("previous_run_dirs", [])
                    if str(value).strip()
                )
                legacy_previous = str(claim.get("previous_run_dir") or "")
                if legacy_previous:
                    previous_run_dirs.append(legacy_previous)
                previous_run_dirs = list(dict.fromkeys(previous_run_dirs))
        atomic_write_json(
            claim_path,
            {
                "schema_version": 1,
                "kind": "run",
                "run_id": run_id,
                "run_dir": str(run_dir.resolve()),
                "session_dir": str(session_dir.resolve()),
                "previous_run_dir": previous_run_dir,
                "previous_run_dirs": previous_run_dirs,
                "created_at": utc_now(),
            },
        )
    return claim_path


def abandon_session_claim(*, claim_path: Path, run_id: str) -> None:
    """Remove an unsubmitted claim while preserving a recoverable prior Run pointer."""

    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        if not claim_path.exists():
            return
        claim = read_json(claim_path)
        if claim.get("kind", "run") != "run" or claim.get("run_id") != run_id:
            return
        previous_dir = Path(str(claim.get("previous_run_dir") or ""))
        if (previous_dir / "run.json").exists():
            previous_run = read_json(previous_dir / "run.json")
            lineage = [
                str(value)
                for value in claim.get("previous_run_dirs", [])
                if str(value).strip()
            ]
            prior_lineage = (
                lineage[1:]
                if lineage and Path(lineage[0]).resolve() == previous_dir.resolve()
                else []
            )
            atomic_write_json(
                claim_path,
                {
                    "schema_version": 1,
                    "kind": "run",
                    "run_id": previous_run.get("run_id", previous_dir.name),
                    "run_dir": str(previous_dir.resolve()),
                    "session_dir": str(claim.get("session_dir") or claim_path.parent),
                    "previous_run_dir": prior_lineage[0] if prior_lineage else "",
                    "previous_run_dirs": prior_lineage,
                    "created_at": previous_run.get("created_at", utc_now()),
                },
            )
        else:
            claim_path.unlink(missing_ok=True)


def acquire_manual_session_claim(*, session_dir: Path, owner: str) -> str:
    claim_path = _claim_path(Path(), session_dir)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    token = uuid.uuid4().hex
    with short_file_lock(guard, timeout=60.0):
        previous_claim: dict[str, Any] | None = None
        if claim_path.exists():
            existing = read_json(claim_path)
            if _claim_blocks_new_run(existing):
                raise RuntimeError("A persistent Run or manual application still owns this session")
            if existing.get("kind", "run") == "run":
                previous_claim = existing
        atomic_write_json(
            claim_path,
            {
                "schema_version": 1,
                "kind": "manual",
                "owner": owner,
                "owner_pid": os.getpid(),
                "token": token,
                "previous_claim": previous_claim,
                "session_dir": str(session_dir.resolve()),
                "created_at": utc_now(),
            },
        )
    return token


def release_manual_session_claim(*, session_dir: Path, token: str) -> None:
    claim_path = _claim_path(Path(), session_dir)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        if not claim_path.exists():
            return
        claim = read_json(claim_path)
        if claim.get("kind") == "manual" and claim.get("token") == token:
            previous = claim.get("previous_claim")
            if isinstance(previous, dict) and previous.get("run_dir"):
                atomic_write_json(claim_path, previous)
            else:
                claim_path.unlink(missing_ok=True)


def update_manual_session_claim_pid(*, session_dir: Path, token: str, child_pid: int) -> None:
    if child_pid <= 0:
        raise ValueError("A positive child PID is required for a manual session lease")
    claim_path = _claim_path(Path(), session_dir)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        claim = read_json(claim_path)
        if claim.get("kind") != "manual" or claim.get("token") != token:
            raise RuntimeError("Manual session lease changed before child launch completed")
        claim["owner_pid"] = int(child_pid)
        claim["child_started_at"] = utc_now()
        atomic_write_json(claim_path, claim)


def release_session_claim(store: RunStore) -> None:
    claim_text = str(store.load_run().get("session_claim_path") or "").strip()
    if not claim_text:
        return
    claim_path = Path(claim_text)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        if not claim_path.exists():
            return
        claim = read_json(claim_path)
        if claim.get("run_id") == store.load_run().get("run_id"):
            claim_path.unlink(missing_ok=True)


def active_session_claim(*, pipeline_root: Path, session_dir: Path) -> dict[str, Any] | None:
    claim_path = _claim_path(pipeline_root, session_dir)
    if not claim_path.exists():
        return None
    claim = read_json(claim_path)
    return claim if _claim_blocks_new_run(claim) else None


def _validate_phy_output(output_dir: Path) -> list[str]:
    missing = [str(output_dir / name) for name in _PHY_REQUIRED if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError("sorting output is incomplete: " + ", ".join(missing))
    spike_times = np.load(output_dir / "spike_times.npy", mmap_mode="r", allow_pickle=False)
    spike_clusters = np.load(output_dir / "spike_clusters.npy", mmap_mode="r", allow_pickle=False)
    templates = np.load(output_dir / "templates.npy", mmap_mode="r", allow_pickle=False)
    if spike_times.reshape(-1).shape[0] != spike_clusters.reshape(-1).shape[0]:
        raise RuntimeError("sorting output has inconsistent spike_times/spike_clusters lengths")
    if templates.ndim < 2 or (output_dir / "params.py").stat().st_size == 0:
        raise RuntimeError("sorting output has invalid templates or params.py")
    return [str((output_dir / name).resolve()) for name in _PHY_REQUIRED]


def _validate_post_output(output_dir: Path) -> list[str]:
    required = [output_dir / name for name in _POST_REQUIRED]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("postprocess output is incomplete: " + ", ".join(missing))
    spike_times = np.load(output_dir / "spike_times.npy", mmap_mode="r", allow_pickle=False)
    spike_clusters = np.load(
        output_dir / "spike_clusters.npy", mmap_mode="r", allow_pickle=False
    )
    if spike_times.reshape(-1).shape[0] != spike_clusters.reshape(-1).shape[0]:
        raise RuntimeError("postprocess output has inconsistent spike array lengths")
    return [str(path.resolve()) for path in required]


def _legacy_preprocess_result(session_dir: Path, settings: Any) -> tuple[dict[str, Any], list[str]] | None:
    manifest_path = session_dir / "preprocessSession_manifest.json"
    manifest: dict[str, Any]
    final_record = session_dir / "preprocess_run.yaml"
    if final_record.exists():
        try:
            import yaml

            payload = yaml.safe_load(final_record.read_text(encoding="utf-8")) or {}
            stage_result = (payload.get("stages") or {}).get("preprocess") or {}
            stage_result = stage_result.get("result") or stage_result
            manifest = dict((stage_result.get("outputs") or {}).get("preprocess_result") or {})
        except (OSError, TypeError, ValueError):
            return None
    elif manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        return None
    if not manifest:
        return None
    basename = str(manifest.get("basename") or session_dir.name)
    dat_path = Path(str(manifest.get("dat_path") or session_dir / f"{basename}.dat"))
    required = [
        dat_path,
        Path(str(manifest.get("session_mat_path") or session_dir / f"{basename}.session.mat")),
        Path(
            str(
                manifest.get("mergepoints_mat_path")
                or session_dir / f"{basename}.MergePoints.events.mat"
            )
        ),
        session_dir / f"{basename}.xml",
        session_dir / "chanMap.mat",
    ]
    config = settings.to_preprocess_config()
    if bool(config.make_lfp):
        required.append(Path(str(manifest.get("lfp_path") or session_dir / f"{basename}.lfp")))
    # Manifest fields are the authoritative inventory for sidecars and
    # artifact-event checkpoints from legacy/direct runs.  Validate every
    # recorded output instead of treating the primary dat/MAT pair as proof of
    # completion.
    for field in ("analog_event_paths", "digital_event_paths", "state_score_paths", "state_score_figure_paths"):
        required.extend(Path(str(value)) for value in manifest.get(field, []) if str(value).strip())
    required.extend(
        Path(str(value))
        for value in dict(manifest.get("intermediate_dat_paths", {})).values()
        if str(value).strip()
    )
    if bool(config.state_score):
        try:
            from src.preprocess.state_scoring import _expected_state_score_outputs

            *_, expected_state = _expected_state_score_outputs(
                basepath=session_dir,
                basename=basename,
                save_lfp_mat=bool(config.state_save_lfp_mat),
            )
            required.extend(expected_state)
        except (ImportError, AttributeError):
            return None
    if any(not path.exists() or (path.is_file() and path.stat().st_size == 0) for path in required):
        return None
    n_channels = int(manifest.get("n_channels") or 0)
    if n_channels < 1 or dat_path.stat().st_size % (n_channels * np.dtype("int16").itemsize):
        return None
    sample_counts = [int(value) for value in manifest.get("subsession_sample_counts", [])]
    if sample_counts:
        actual_frames = dat_path.stat().st_size // (n_channels * np.dtype("int16").itemsize)
        if actual_frames != sum(sample_counts):
            return None
    source_paths = [Path(str(value)).expanduser() for value in manifest.get("subsession_paths", [])]
    if not source_paths or len(source_paths) != len(sample_counts):
        return None
    if any(
        not path.exists()
        or not path.is_file()
        or count <= 0
        or path.stat().st_size % (count * np.dtype("int16").itemsize) != 0
        or path.stat().st_size // (count * np.dtype("int16").itemsize) < n_channels
        for path, count in zip(source_paths, sample_counts, strict=True)
    ):
        return None
    source = settings.preprocess_source_path or Path(str(manifest.get("basepath") or session_dir))
    result = {
        "basepath": str(Path(source).resolve()),
        "basename": basename,
        "local_output_dir": str(session_dir.resolve()),
        "dat_path": str(dat_path.resolve()),
        "lfp_path": str(manifest.get("lfp_path") or "") or None,
        "session_mat_path": str(required[1].resolve()),
        "mergepoints_mat_path": str(required[2].resolve()),
        "analog_event_paths": list(manifest.get("analog_event_paths", [])),
        "digital_event_paths": list(manifest.get("digital_event_paths", [])),
        "intermediate_dat_paths": dict(manifest.get("intermediate_dat_paths", {})),
        "n_channels": n_channels,
        "sr": float(manifest.get("sr") or 0.0),
        "sr_lfp": manifest.get("sr_lfp"),
        "bad_channels_0based": [int(value) for value in manifest.get("bad_channels_0based", [])],
        "bad_channels_1based": [int(value) for value in manifest.get("bad_channels_1based", [])],
        "subsession_paths": list(manifest.get("subsession_paths", [])),
        "subsession_sample_counts": sample_counts,
        "sorter": None,
        "sorter_output_dir": None,
        "sorter_output_dirs": [],
        "sorter_partition_manifest_path": None,
        "state_score_paths": list(manifest.get("state_score_paths", [])),
        "state_score_figure_paths": list(manifest.get("state_score_figure_paths", [])),
    }
    return result, [str(path.resolve()) for path in required]


def _sorting_outputs(session_dir: Path) -> tuple[list[Path], Path | None]:
    from src.sorting_manifest import all_partitions_skipped
    manifest_path = session_dir / "sorter_partition_manifest.json"
    outputs: list[Path] = []
    if all_partitions_skipped(manifest_path):
        return [], manifest_path
    if manifest_path.exists():
        payload = read_json(manifest_path)
        for item in payload.get("partitions", []):
            if not isinstance(item, dict) or item.get("status") not in {None, "completed"}:
                continue
            value = str(item.get("output_folder") or "").strip()
            if value:
                candidate = Path(value).expanduser().resolve()
                if (
                    candidate.parent == session_dir.resolve()
                    and candidate.is_dir()
                    and candidate.name.startswith("Kilosort")
                    and "_spi" not in candidate.name
                    and ".preserved-" not in candidate.name
                ):
                    outputs.append(candidate)
    if not outputs and not manifest_path.exists():
        outputs = sorted(
            (
                path.resolve()
                for path in session_dir.glob("Kilosort*")
                if path.is_dir()
                and "_spi" not in path.name
                and ".preserved-" not in path.name
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:1]
    return outputs, manifest_path if manifest_path.exists() else None


def _write_adopted_result(
    store: RunStore,
    *,
    stage: StageName,
    attempt: int,
    outputs: dict[str, Any],
    validated_paths: list[str],
    source: str,
) -> None:
    now = utc_now()
    inventory = {
        "schema": OUTPUT_INVENTORY_SCHEMA,
        "producer_schema": PRODUCER_SCHEMAS[stage.value],
        "entries": [
            _inventory_entry(Path(path), "validated_output") for path in validated_paths
        ],
    }
    outputs = dict(outputs)
    outputs["output_inventory"] = inventory
    store.write_attempt_fact(
        stage,
        attempt,
        "adopted.json",
        {"adopted_at": now, "source": source, "validated_paths": validated_paths},
    )
    store.write_attempt_fact(
        stage,
        attempt,
        "result.json",
        {
            "schema_version": 1,
            "stage": stage.value,
            "attempt": attempt,
            "status": StageStatus.COMPLETED.value,
            "analysis_sha256": store.load_analysis().sha256,
            "started_at": now,
            "finished_at": now,
            "adopted": True,
            "outputs": _json_safe(outputs),
            "producer_schema": PRODUCER_SCHEMAS[stage.value],
            "validation": {
                "passed": True,
                "validated_paths": validated_paths,
                "output_inventory": inventory,
            },
        },
    )


def _previous_persistent_store(store: RunStore) -> RunStore | None:
    stores = _previous_persistent_stores(store)
    return stores[0] if stores else None


def _previous_persistent_stores(store: RunStore) -> list[RunStore]:
    """Return immutable Run lineage, with the current claim as a legacy fallback."""

    run = store.load_run()
    candidates: list[str] = []
    candidates.extend(
        str(value) for value in run.get("previous_run_dirs", []) if str(value).strip()
    )
    previous = str(run.get("previous_run_dir") or "").strip()
    if previous:
        candidates.append(previous)
    claim_text = str(run.get("session_claim_path") or "").strip()
    try:
        if claim_text and Path(claim_text).exists():
            claim = read_json(Path(claim_text))
            # A claim is mutable.  Consult it only while it still names this
            # Run; persisted lineage above remains valid after a later Run
            # replaces the claim.
            if Path(str(claim.get("run_dir") or "")).resolve() == store.run_dir:
                candidates.extend(
                    str(value)
                    for value in claim.get("previous_run_dirs", [])
                    if str(value).strip()
                )
                legacy_previous = str(claim.get("previous_run_dir") or "").strip()
                if legacy_previous:
                    candidates.append(legacy_previous)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    session_dir = str(run.get("session_output_dir") or "").strip()
    result: list[RunStore] = []
    seen: set[Path] = {store.run_dir}
    for value in candidates:
        path = Path(value).expanduser().resolve()
        if path in seen or not (path / "run.json").exists():
            continue
        try:
            prior = RunStore(path)
            prior_run = prior.load_run()
            if session_dir and str(prior_run.get("session_output_dir") or "") != session_dir:
                continue
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        seen.add(path)
        result.append(prior)
    return result


def _completed_previous_result(
    previous: RunStore, stage: StageName
) -> tuple[dict[str, Any], list[str]] | None:
    view = previous.derive_state()["stages"][stage.value]
    if view.get("status") != StageStatus.COMPLETED.value:
        return None
    attempt = view.get("selected_attempt")
    if attempt is None:
        return None
    result = previous.read_attempt_fact(stage, int(attempt), "result.json")
    if (
        result is None
        or result.get("status") != StageStatus.COMPLETED.value
        or result.get("analysis_sha256") != previous.load_analysis().sha256
        or (result.get("validation") or {}).get("passed") is not True
    ):
        return None
    try:
        validated = _revalidate_previous_outputs(stage, result)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return result, validated


def _revalidate_previous_outputs(stage: StageName, result: dict[str, Any]) -> list[str]:
    producer_schema = result.get("producer_schema")
    expected_producer = PRODUCER_SCHEMAS[stage.value]
    allowed_producers = {expected_producer}
    if stage == StageName.PREPROCESS:
        allowed_producers.add("preprocess-output-validated-legacy-resume-v1")
    if producer_schema is not None and producer_schema not in allowed_producers:
        raise RuntimeError(
            f"previous {stage.value} output was produced by incompatible schema "
            f"{producer_schema!r}"
        )
    inventory = (result.get("validation") or {}).get("output_inventory")
    if inventory is None:
        inventory = (result.get("outputs") or {}).get("output_inventory")
    if inventory is not None:
        if inventory.get("producer_schema") not in allowed_producers:
            raise RuntimeError(f"previous {stage.value} output inventory has incompatible producer schema")
        inventory_validated = validate_output_inventory(
            inventory, label=f"previous {stage.value} output"
        )
        if stage == StageName.PREPROCESS:
            return inventory_validated
    outputs = dict(result.get("outputs") or {})
    if stage == StageName.PREPROCESS:
        pre = dict(outputs.get("preprocess_result") or {})
        output_dir = Path(str(pre.get("local_output_dir") or ""))
        basename = str(pre.get("basename") or output_dir.name)
        dat_path = Path(str(pre.get("dat_path") or output_dir / f"{basename}.dat"))
        required = [
            dat_path,
            Path(str(pre.get("session_mat_path") or output_dir / f"{basename}.session.mat")),
            Path(
                str(
                    pre.get("mergepoints_mat_path")
                    or output_dir / f"{basename}.MergePoints.events.mat"
                )
            ),
            output_dir / f"{basename}.xml",
            output_dir / "chanMap.mat",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("previous preprocess output is incomplete")
        n_channels = int(pre.get("n_channels") or 0)
        if n_channels < 1 or dat_path.stat().st_size % (n_channels * np.dtype("int16").itemsize):
            raise RuntimeError("previous preprocess dat has an invalid frame shape")
        sample_counts = [int(value) for value in pre.get("subsession_sample_counts", [])]
        if sample_counts:
            frames = dat_path.stat().st_size // (n_channels * np.dtype("int16").itemsize)
            if frames != sum(sample_counts):
                raise RuntimeError("previous preprocess dat has a stale sample count")
        return [str(path.resolve()) for path in required]
    if stage == StageName.SORTING:
        from src.sorting_manifest import all_partitions_skipped
        output_dirs = [
            Path(value).expanduser().resolve()
            for value in outputs.get("sorter_output_dirs", [])
            if str(value).strip()
        ]
        skipped_manifest = outputs.get("sorter_partition_manifest_path")
        if not output_dirs and all_partitions_skipped(Path(skipped_manifest) if skipped_manifest else None):
            return [str(Path(skipped_manifest).resolve())]
        if not output_dirs:
            raise RuntimeError("previous sorting result has no output directories")
        validated = [item for path in output_dirs for item in _validate_phy_output(path)]
        manifest_text = str(outputs.get("sorter_partition_manifest_path") or "").strip()
        if manifest_text:
            manifest_path = Path(manifest_text)
            manifest = read_json(manifest_path)
            completed = {
                str(Path(item.get("output_folder", "")).resolve())
                for item in manifest.get("partitions", [])
                if isinstance(item, dict)
                and item.get("status") == StageStatus.COMPLETED.value
                and item.get("output_folder")
            }
            if completed != {str(path) for path in output_dirs}:
                raise RuntimeError("previous sorter manifest disagrees with output directories")
            validated.append(str(manifest_path.resolve()))
        return validated
    post_results = outputs.get("postprocess_results", [])
    output_dirs = [
        Path(str(item.get("output_folder"))).resolve()
        for item in post_results
        if isinstance(item, dict) and item.get("output_folder")
    ]
    if not output_dirs:
        if outputs.get("skip_reason") == "no_active_channels":
            from src.sorting_manifest import all_partitions_skipped
            skipped_manifest = outputs.get("sorter_partition_manifest_path")
            if skipped_manifest and all_partitions_skipped(Path(skipped_manifest)):
                return [str(Path(skipped_manifest).resolve())]
            paths = [Path(value) for value in outputs.get("validated_paths", [])]
            if paths and all(path.is_file() for path in paths):
                return [str(path.resolve()) for path in paths]
        raise RuntimeError("previous postprocess result has no output directories")
    return [item for path in output_dirs for item in _validate_post_output(path)]


def adopt_existing_outputs(store: RunStore) -> dict[str, str]:
    """Adopt deeply validated legacy outputs for overwrite-disabled Stages."""

    settings = settings_for_store(store)
    session_dir = session_output_dir(settings)
    enabled = [StageName(value) for value in store.load_run().get("enabled_stages", [])]
    decisions: dict[str, str] = {stage.value: "run" for stage in enabled}
    current_fingerprints = stage_fingerprints(store.load_analysis())
    prior_fingerprints = _prior_stage_fingerprints(session_dir)
    previous = _previous_persistent_store(store)
    previous_fingerprints = (
        dict(previous.load_run().get("stage_fingerprints") or {}) if previous is not None else {}
    )
    if previous is not None and not previous_fingerprints:
        previous_fingerprints = stage_fingerprints(previous.load_analysis())

    def previous_compatible(stage: StageName) -> bool:
        if previous is None:
            return False
        if previous_fingerprints.get(stage.value) != current_fingerprints[stage.value]:
            return False
        if stage != StageName.PREPROCESS:
            return True
        try:
            prior_inputs = read_json(previous.run_dir / "snapshots" / "inputs.json")
            current_inputs = read_json(store.run_dir / "snapshots" / "inputs.json")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return _input_snapshots_compatible(
            prior_inputs,
            current_inputs,
            settings,
            prior_cutoff_at=previous.load_run().get("created_at"),
        )

    def compatible(stage: StageName) -> bool:
        if prior_fingerprints is not None:
            matches = prior_fingerprints.get(stage.value) == current_fingerprints[stage.value]
            if stage == StageName.PREPROCESS:
                matches = matches and _persistent_input_provenance_compatible(
                    session_dir, store, settings
                )
            return matches
        if stage == StageName.PREPROCESS:
            return _legacy_preprocess_compatible(session_dir, settings)
        if stage == StageName.SORTING:
            return _legacy_sorting_compatible(session_dir, store, settings)
        # Legacy postprocess folders do not contain a complete, canonical copy
        # of the scientific postprocess settings. Shape validation alone is not
        # evidence of parameter compatibility.
        return False

    preprocess_result: dict[str, Any] | None = None
    preprocess_attempt = store.list_attempt_numbers(StageName.PREPROCESS)
    if (
        StageName.PREPROCESS in enabled
        and preprocess_attempt
        and not settings.preprocess.overwrite
        and previous_compatible(StageName.PREPROCESS)
        and previous is not None
    ):
        adopted_previous = _completed_previous_result(previous, StageName.PREPROCESS)
        if adopted_previous is not None:
            prior_result, validated = adopted_previous
            _write_adopted_result(
                store,
                stage=StageName.PREPROCESS,
                attempt=preprocess_attempt[-1],
                outputs=dict(prior_result.get("outputs") or {}),
                validated_paths=validated,
                source=f"persistent Run {previous.run_dir.name}",
            )
            preprocess_result = dict(
                (prior_result.get("outputs") or {}).get("preprocess_result") or {}
            )
            decisions[StageName.PREPROCESS.value] = "reuse"
    if (
        StageName.PREPROCESS in enabled
        and preprocess_attempt
        and decisions[StageName.PREPROCESS.value] != "reuse"
        and not settings.preprocess.overwrite
        and compatible(StageName.PREPROCESS)
    ):
        adopted = _legacy_preprocess_result(session_dir, settings)
        if adopted is not None:
            preprocess_result, validated = adopted
            _write_adopted_result(
                store,
                stage=StageName.PREPROCESS,
                attempt=preprocess_attempt[-1],
                outputs={
                    "preprocess_result": preprocess_result,
                    "xml_path": str((session_dir / f"{session_dir.name}.xml").resolve()),
                    "chanmap_path": str((session_dir / "chanMap.mat").resolve()),
                    "dtype": "int16",
                    "validated_paths": validated,
                },
                validated_paths=validated,
                source="legacy preprocess manifest",
            )
            decisions[StageName.PREPROCESS.value] = "reuse"

    sorting_attempt = store.list_attempt_numbers(StageName.SORTING)
    sorting_outputs: list[Path] = []
    if (
        StageName.SORTING in enabled
        and sorting_attempt
        and not settings.preprocess.overwrite
        and previous_compatible(StageName.SORTING)
        and previous is not None
        and (
            StageName.PREPROCESS not in enabled
            or decisions.get(StageName.PREPROCESS.value) == "reuse"
        )
    ):
        adopted_previous = _completed_previous_result(previous, StageName.SORTING)
        if adopted_previous is not None:
            prior_result, validated = adopted_previous
            _write_adopted_result(
                store,
                stage=StageName.SORTING,
                attempt=sorting_attempt[-1],
                outputs=dict(prior_result.get("outputs") or {}),
                validated_paths=validated,
                source=f"persistent Run {previous.run_dir.name}",
            )
            sorting_outputs = [
                Path(value).resolve()
                for value in (prior_result.get("outputs") or {}).get(
                    "sorter_output_dirs", []
                )
            ]
            decisions[StageName.SORTING.value] = "reuse"
    if (
        StageName.SORTING in enabled
        and sorting_attempt
        and decisions[StageName.SORTING.value] != "reuse"
        and not settings.preprocess.overwrite
        and compatible(StageName.SORTING)
        and (
            StageName.PREPROCESS not in enabled
            or decisions.get(StageName.PREPROCESS.value) == "reuse"
        )
    ):
        candidates, manifest_path = _sorting_outputs(session_dir)
        from src.sorting_manifest import all_partitions_skipped
        try:
            validated = [item for path in candidates for item in _validate_phy_output(path)]
        except (OSError, RuntimeError, ValueError):
            candidates = []
        if candidates or all_partitions_skipped(manifest_path):
            sorting_outputs = candidates
            if manifest_path is not None:
                validated.append(str(manifest_path.resolve()))
            sorting_preprocess = dict(preprocess_result or {})
            sorting_preprocess.update(
                {
                    "sorter": settings.preprocess.sorter,
                    "sorter_output_dir": str(candidates[0]) if len(candidates) == 1 else None,
                    "sorter_output_dirs": [str(path) for path in candidates],
                    "sorter_partition_manifest_path": (
                        str(manifest_path.resolve()) if manifest_path is not None else None
                    ),
                }
            )
            _write_adopted_result(
                store,
                stage=StageName.SORTING,
                attempt=sorting_attempt[-1],
                outputs={
                    "preprocess_result": sorting_preprocess,
                    "sorter_output_dir": str(candidates[0]) if len(candidates) == 1 else "",
                    "sorter_output_dirs": [str(path) for path in candidates],
                    "sorter_partition_manifest_path": (
                        str(manifest_path.resolve()) if manifest_path is not None else ""
                    ),
                    "validated_paths": validated,
                },
                validated_paths=validated,
                source="validated existing Kilosort output",
            )
            decisions[StageName.SORTING.value] = "reuse"

    post_attempt = store.list_attempt_numbers(StageName.POSTPROCESS)
    if (
        StageName.POSTPROCESS in enabled
        and post_attempt
        and not settings.postprocess.overwrite
        and previous_compatible(StageName.POSTPROCESS)
        and previous is not None
        and (
            StageName.SORTING not in enabled
            or decisions.get(StageName.SORTING.value) == "reuse"
        )
    ):
        adopted_previous = _completed_previous_result(previous, StageName.POSTPROCESS)
        if adopted_previous is not None:
            prior_result, validated = adopted_previous
            _write_adopted_result(
                store,
                stage=StageName.POSTPROCESS,
                attempt=post_attempt[-1],
                outputs=dict(prior_result.get("outputs") or {}),
                validated_paths=validated,
                source=f"persistent Run {previous.run_dir.name}",
            )
            decisions[StageName.POSTPROCESS.value] = "reuse"
    if (
        StageName.POSTPROCESS in enabled
        and post_attempt
        and decisions[StageName.POSTPROCESS.value] != "reuse"
        and not settings.postprocess.overwrite
        and compatible(StageName.POSTPROCESS)
        and (
            StageName.SORTING not in enabled
            or decisions.get(StageName.SORTING.value) == "reuse"
        )
    ):
        if not sorting_outputs:
            sorting_outputs, _ = _sorting_outputs(session_dir)
        post_outputs = [path.parent / f"{path.name}_spi" for path in sorting_outputs]
        try:
            validated = [item for output in post_outputs for item in _validate_post_output(output)]
        except (OSError, RuntimeError, ValueError):
            validated = []
        if post_outputs and validated:
            _write_adopted_result(
                store,
                stage=StageName.POSTPROCESS,
                attempt=post_attempt[-1],
                outputs={
                    "postprocess_results": [
                        {
                            "sorting_phy_folder": str(sorting),
                            "output_folder": str(output),
                            "metrics_csv_path": str(output / "quality_metrics.csv"),
                        }
                        for sorting, output in zip(sorting_outputs, post_outputs, strict=True)
                    ],
                    "validated_paths": validated,
                },
                validated_paths=validated,
                source="validated existing postprocess output",
            )
            decisions[StageName.POSTPROCESS.value] = "reuse"

    atomic_write_json(
        store.run_dir / "resume_plan.json",
        {"created_at": utc_now(), "session_dir": str(session_dir), "stages": decisions},
    )
    store.rebuild_state()
    return decisions


def _selected_result(store: RunStore, stage: StageName) -> dict[str, Any] | None:
    view = store.derive_state()["stages"][stage.value]
    attempt = view.get("selected_attempt")
    if attempt is None:
        return None
    return store.read_attempt_fact(stage, int(attempt), "result.json")


def _selected_stage_summary(store: RunStore, stage: StageName) -> dict[str, Any] | None:
    view = store.derive_state()["stages"][stage.value]
    attempt = view.get("selected_attempt")
    if attempt is None:
        return None
    attempt_number = int(attempt)
    submitted = store.read_attempt_fact(stage, attempt_number, "submitted.json")
    return {
        "attempt": attempt_number,
        "status": view.get("status"),
        "spec": store.load_attempt_spec(stage, attempt_number).to_dict(),
        "submitted": submitted,
        "terminal_observation": store.latest_observation(stage, attempt_number),
        "result": store.read_attempt_fact(stage, attempt_number, "result.json"),
    }


def _conclusive_success(store: RunStore) -> bool:
    state = store.derive_state()
    if state.get("status") != StageStatus.COMPLETED.value:
        return False
    for stage in StageName:
        view = state["stages"][stage.value]
        if not view.get("enabled"):
            continue
        attempt = view.get("selected_attempt")
        if attempt is None:
            return False
        submitted = store.read_attempt_fact(stage, int(attempt), "submitted.json")
        if submitted is None:
            continue
        observation = store.latest_observation(stage, int(attempt))
        status = (observation or {}).get("status") or {}
        if not (status.get("terminal") is True and status.get("successful") is True):
            return False
    return True


def _log_sources(store: RunStore, sorting_dirs: list[Path]) -> list[tuple[str, Path]]:
    values: list[tuple[str, Path]] = []
    controller_log = store.run_dir / "controller.log"
    if controller_log.exists():
        values.append(("controller", controller_log))
    for stage in StageName:
        for attempt in store.list_attempt_numbers(stage):
            attempt_dir = store.attempt_dir(stage, attempt)
            for name in ("stdout.log", "stderr.log"):
                path = attempt_dir / name
                if path.exists():
                    values.append((f"{stage.value}/attempt-{attempt:03d}/{name}", path))
    for output in sorting_dirs:
        for name in ("kilosort.log", "matlab_run.log", "spikeinterface_log.json"):
            path = output / name
            if path.exists():
                values.append((f"{output.name}/{name}", path))
    return values


_WARNING_RE = re.compile(r"(?:^|\b)(?:warning|warn)(?:\b|:)", re.IGNORECASE)
_ERROR_RE = re.compile(
    r"(?:traceback \(most recent call last\)|\b(?:fatal|exception|segmentation fault|out of memory)\b|\berror:)",
    re.IGNORECASE,
)
_NEGATED_WARNING_RE = re.compile(
    r"(?:without|no|zero|0)\s+(?:detected\s+)?warnings?\b", re.IGNORECASE
)


def _diagnostics(sources: list[tuple[str, Path]]) -> tuple[list[str], list[str]]:
    warnings_found: list[str] = []
    errors_found: list[str] = []
    seen_warning_lines: set[str] = set()
    seen_error_lines: set[str] = set()
    for label, path in sources:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rendered = f"{label}: {stripped[:500]}"
            if _ERROR_RE.search(stripped):
                if stripped not in seen_error_lines:
                    errors_found.append(rendered)
                    seen_error_lines.add(stripped)
            elif _WARNING_RE.search(stripped) and not _NEGATED_WARNING_RE.search(stripped):
                if stripped not in seen_warning_lines:
                    warnings_found.append(rendered)
                    seen_warning_lines.add(stripped)
    return warnings_found, errors_found


def _sorting_dirs_from_results(store: RunStore) -> list[Path]:
    result = _selected_result(store, StageName.SORTING) or {}
    return [
        Path(value).expanduser().resolve()
        for value in (result.get("outputs") or {}).get("sorter_output_dirs", [])
        if str(value).strip()
    ]


def _write_pipeline_log(
    *,
    store: RunStore,
    destination: Path,
    sources: list[tuple[str, Path]],
    warnings_found: list[str],
    errors_found: list[str],
) -> None:
    state = store.derive_state()
    run = store.load_run()
    lines = [
        f"RUN {run.get('run_id')} status=completed",
        f"started_at={run.get('created_at', '')}",
        f"finished_at={utc_now()}",
        f"backend={run.get('resolved_backend', '')}",
        f"warnings_detected={len(warnings_found)} errors_detected={len(errors_found)}",
        "",
    ]
    for stage in StageName:
        view = state["stages"][stage.value]
        if view.get("enabled"):
            result = _selected_result(store, stage) or {}
            submitted = store.read_attempt_fact(
                stage, int(view["selected_attempt"]), "submitted.json"
            )
            jobs = [str(job.get("job_id")) for job in (submitted or {}).get("jobs", [])]
            observation = store.latest_observation(stage, int(view["selected_attempt"])) or {}
            telemetry = (observation.get("status") or {}).get("telemetry") or {}
            lines.append(
                f"STAGE {stage.value} status={view.get('status')} "
                f"attempt={view.get('selected_attempt')} "
                f"started_at={result.get('started_at', '')} "
                f"finished_at={result.get('finished_at', '')} "
                f"jobs={','.join(jobs) or '-'} "
                f"elapsed={telemetry.get('elapsed', '-')} "
                f"max_rss={telemetry.get('max_rss', '-')}"
            )
    for label, path in sources:
        if path.name in {"kilosort.log", "matlab_run.log", "spikeinterface_log.json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"[log read failed: {exc}]\n"
        lines.extend(("", f"===== {label} =====", text.rstrip()))
    if warnings_found:
        lines.extend(("", "===== DETECTED WARNINGS =====", *warnings_found))
    if errors_found:
        lines.extend(("", "===== DETECTED ERRORS =====", *errors_found))
    atomic_write_text(destination, "\n".join(lines).rstrip() + "\n")


def _remove_success_helpers(store: RunStore, sorting_dirs: list[Path]) -> None:
    for output in sorting_dirs:
        matlab_log = output / "matlab_run.log"
        kilosort_log = output / "kilosort.log"
        if matlab_log.exists() and kilosort_log.exists():
            try:
                if hashlib.sha256(matlab_log.read_bytes()).digest() == hashlib.sha256(
                    kilosort_log.read_bytes()
                ).digest():
                    matlab_log.unlink()
            except OSError:
                pass
        for name in (
            "spikeinterface_log.json",
            "spikeinterface_params.json",
            "spikeinterface_recording.json",
            "run_kilosort.sh",
            "sorter_config_source.yaml",
            "sorter_config_source.yml",
            "sorter_config_source.json",
        ):
            (output / name).unlink(missing_ok=True)
    state = store.derive_state()
    has_historical_attempts = False
    for stage in StageName:
        selected = state["stages"][stage.value].get("selected_attempt")
        attempts = store.list_attempt_numbers(stage)
        has_historical_attempts = has_historical_attempts or any(
            selected is None or attempt != int(selected) for attempt in attempts
        )
        if selected is None:
            continue
        attempt_dir = store.attempt_dir(stage, int(selected))
        for name in ("stdout.log", "stderr.log"):
            (attempt_dir / name).unlink(missing_ok=True)
    if not has_historical_attempts:
        atomic_write_text(
            store.run_dir / "controller.log", "Run finalized; see session preprocess.log.\n"
        )


def finalize_successful_run(store: RunStore) -> dict[str, Any] | None:
    """Write the compact final record once all enabled jobs conclusively succeed."""

    final_path = store.run_dir / "final.json"
    if final_path.exists():
        record = read_json(final_path)
        if _conclusive_success(store):
            release_session_claim(store)
        return record
    if not _conclusive_success(store):
        return None
    settings = settings_for_store(store)
    session_dir = session_output_dir(settings)
    session_dir.mkdir(parents=True, exist_ok=True)
    prior_record: dict[str, Any] = {}
    prior_record_path = session_dir / "preprocess_run.yaml"
    if prior_record_path.exists():
        try:
            import yaml

            prior_record = yaml.safe_load(prior_record_path.read_text(encoding="utf-8")) or {}
        except (OSError, TypeError, ValueError):
            prior_record = {}
    sorting_dirs = _sorting_dirs_from_results(store)
    if not sorting_dirs:
        sorting_dirs = [
            Path(value).expanduser().resolve()
            for value in prior_record.get("sorting_output_dirs", [])
            if str(value).strip()
        ]
    sources = _log_sources(store, sorting_dirs)
    warnings_found, errors_found = _diagnostics(sources)
    stage_results = dict(prior_record.get("stages") or {})
    enabled_stage_names: list[str] = []
    for stage in StageName:
        if store.derive_state()["stages"][stage.value].get("enabled"):
            stage_results[stage.value] = _selected_stage_summary(store, stage)
            enabled_stage_names.append(stage.value)
    fingerprints = dict(prior_record.get("stage_fingerprints") or {})
    current_fingerprints = stage_fingerprints(store.load_analysis())
    for stage_name in enabled_stage_names:
        fingerprints[stage_name] = current_fingerprints[stage_name]
    current_analysis = store.load_analysis().to_dict()
    prior_analysis = prior_record.get("analysis") or {}
    # A postprocess-only Run intentionally has a narrow analysis snapshot.
    # Publishing it wholesale would erase the preprocess/sorting parameters
    # that produced the retained upstream outputs and break faithful session
    # recovery.  Preserve that stage-owned portion of the prior record while
    # recording the current postprocess settings.
    if (
        StageName.PREPROCESS.value not in enabled_stage_names
        and isinstance(prior_analysis, dict)
        and isinstance(prior_analysis.get("settings"), dict)
    ):
        prior_settings = prior_analysis["settings"]
        current_settings = dict(current_analysis.get("settings") or {})
        if isinstance(prior_settings.get("preprocess"), dict):
            current_settings["preprocess"] = dict(prior_settings["preprocess"])
        current_analysis["settings"] = current_settings
        prior_artifacts = prior_analysis.get("artifact_sha256")
        if isinstance(prior_artifacts, dict):
            artifacts = dict(current_analysis.get("artifact_sha256") or {})
            for name in ("sorter_config",):
                if name in prior_artifacts:
                    artifacts[name] = prior_artifacts[name]
            current_analysis["artifact_sha256"] = artifacts
    # The merged settings are a new immutable snapshot, not the hash of the
    # narrow post-only snapshot.  Validate its serialized form immediately.
    current_analysis = AnalysisConfig.create(
        dict(current_analysis["settings"]),
        artifact_sha256=dict(current_analysis.get("artifact_sha256") or {}),
    ).to_dict()
    AnalysisConfig.from_dict(current_analysis)
    current_stage_provenance = {
        name: read_json(store.run_dir / "snapshots" / f"{name}.json")
        for name in ("git", "environment", "inputs")
        if (store.run_dir / "snapshots" / f"{name}.json").exists()
    }
    stage_provenance = dict(prior_record.get("stage_provenance") or {})
    prior_provenance = prior_record.get("provenance")
    if isinstance(prior_provenance, dict):
        for stage_name in (prior_record.get("stages") or {}):
            stage_provenance.setdefault(str(stage_name), prior_provenance)
    for stage_name in enabled_stage_names:
        stage_provenance[stage_name] = current_stage_provenance
    record = {
        "schema_version": 1,
        "run_id": store.load_run().get("run_id"),
        "status": (
            "completed_with_log_findings"
            if warnings_found or errors_found
            else "completed_clean"
        ),
        "started_at": store.load_run().get("created_at"),
        "finished_at": utc_now(),
        "backend": store.load_run().get("resolved_backend"),
        "source_basepath": str(
            settings.preprocess_source_path or prior_record.get("source_basepath") or ""
        ),
        "session_output_dir": str(session_dir),
        "analysis": current_analysis,
        "stage_fingerprints": fingerprints,
        "execution": store.load_execution().to_dict(),
        "provenance": current_stage_provenance,
        "stage_provenance": stage_provenance,
        "stages": stage_results,
        "warnings": {"count": len(warnings_found), "items": warnings_found},
        "errors": {"count": len(errors_found), "items": errors_found},
        "sorting_output_dirs": [str(path) for path in sorting_dirs],
    }
    import yaml

    atomic_write_text(
        session_dir / "preprocess_run.yaml",
        yaml.safe_dump(record, sort_keys=False, allow_unicode=True),
    )
    _write_pipeline_log(
        store=store,
        destination=session_dir / "preprocess.log",
        sources=sources,
        warnings_found=warnings_found,
        errors_found=errors_found,
    )
    atomic_write_json(final_path, _json_safe(record))
    release_session_claim(store)
    return record
