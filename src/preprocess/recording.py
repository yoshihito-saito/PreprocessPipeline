from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import os
import uuid
import warnings

import numpy as np
from scipy.io import loadmat
from scipy import signal

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from .io import atomic_write_json


def _set_channel_property_compat(recording: Any, channel_ids: list[int], key: str, values: list[Any]) -> None:
    if hasattr(recording, "set_channel_property"):
        recording.set_channel_property(channel_ids, key, values)
        return
    if hasattr(recording, "set_property"):
        try:
            recording.set_property(key, values, ids=channel_ids)
        except TypeError:
            # Fallback for API variants that do not accept ids.
            recording.set_property(key, values)
        return
    raise AttributeError("Recording object does not support set_channel_property/set_property.")


def _chanmap_device_channel_indices(mat: dict[str, Any]) -> np.ndarray:
    if "chanMap0ind" in mat:
        return np.asarray(mat["chanMap0ind"]).flatten().astype(int)
    return np.asarray(mat["chanMap"]).flatten().astype(int) - 1


def _build_chanmap_group_properties(chanmap_mat_path: Path, channel_ids: list[int]) -> dict[str, list[int]]:
    mat = loadmat(chanmap_mat_path)
    required = {"kcoords"}
    if not required.issubset(set(mat.keys())):
        missing = ", ".join(sorted(required.difference(set(mat.keys()))))
        raise ValueError(f"chanMap is missing required fields: {missing}")
    if "chanMap" not in mat and "chanMap0ind" not in mat:
        raise ValueError("chanMap is missing channel IDs")

    device_ch_inds = _chanmap_device_channel_indices(mat)
    if device_ch_inds.size == 0:
        raise ValueError("chanMap has no channel entries.")

    probe_ids = np.asarray(mat.get("probe_ids", np.ones_like(device_ch_inds))).flatten().astype(int)
    shank_ids = np.asarray(mat["kcoords"]).flatten().astype(int)
    n = min(device_ch_inds.size, probe_ids.size, shank_ids.size)
    if n == 0:
        raise ValueError("chanMap has empty probe/shank metadata.")

    device_ch_inds = device_ch_inds[:n]
    probe_ids = probe_ids[:n]
    shank_ids = shank_ids[:n]

    probe_map: dict[int, int] = {}
    shank_map: dict[int, tuple[int, int]] = {}
    for dev_ch, probe_id, shank_id in zip(device_ch_inds, probe_ids, shank_ids, strict=True):
        probe_map[int(dev_ch)] = int(probe_id)
        shank_map[int(dev_ch)] = (int(probe_id), int(shank_id))

    channel_ids_int = [int(ch) for ch in channel_ids]
    missing_channels = [ch for ch in channel_ids_int if ch not in probe_map or ch not in shank_map]
    if missing_channels:
        preview = ", ".join(str(ch) for ch in missing_channels[:10])
        suffix = "..." if len(missing_channels) > 10 else ""
        raise ValueError(
            "chanMap does not define probe/shank assignments for all recording channels. "
            f"Missing channel ids: {preview}{suffix}"
        )

    probe_groups = [probe_map[ch] for ch in channel_ids_int]

    pair_to_group: dict[tuple[int, int], int] = {}
    next_group = 0
    shank_groups: list[int] = []
    for ch in channel_ids_int:
        key = shank_map[ch]
        if key not in pair_to_group:
            pair_to_group[key] = next_group
            next_group += 1
        shank_groups.append(pair_to_group[key])

    return {
        "artifact_group_probe": probe_groups,
        "artifact_group_shank": shank_groups,
    }


def attach_probe_from_chanmap(recording: Any, chanmap_mat_path: Path, *, original_num_channels: int | None = None) -> Any:
    from .channel_layout import load_channel_layout
    original_ids = [int(ch) for ch in recording.get_channel_ids()]
    if original_num_channels is None and hasattr(recording, "get_annotation"):
        original_num_channels = recording.get_annotation("binary_num_channels")
    bound = original_num_channels or (max(original_ids) + 1 if original_ids else 0)
    mat = load_channel_layout(chanmap_mat_path, int(bound))
    required = {"xcoords", "ycoords", "kcoords", "chanMap"}
    if not required.issubset(set(mat.keys())):
        return recording

    try:
        from probeinterface import Probe, ProbeGroup
    except Exception:
        return recording

    x = np.asarray(mat["xcoords"]).flatten()
    y = np.asarray(mat["ycoords"]).flatten()
    shank_ids = np.asarray(mat["kcoords"]).flatten()
    probe_ids = np.asarray(mat.get("probe_ids", np.ones_like(x))).flatten()
    device_ch_inds = _chanmap_device_channel_indices(mat)

    n_contacts = min(
        x.size,
        y.size,
        shank_ids.size,
        probe_ids.size,
        device_ch_inds.size,
    )
    if n_contacts <= 0:
        return recording

    x = x[:n_contacts]
    y = y[:n_contacts]
    shank_ids = shank_ids[:n_contacts]
    probe_ids = probe_ids[:n_contacts]
    device_ch_inds = device_ch_inds[:n_contacts]

    n_recording_channels = int(recording.get_num_channels())
    valid_mask = np.isin(device_ch_inds, original_ids)
    if not np.any(valid_mask):
        warnings.warn(
            "chanMap has no valid device_channel_indices for this recording; "
            "skipping probe attachment.",
            RuntimeWarning,
            stacklevel=2,
        )
        return recording

    x = x[valid_mask]
    y = y[valid_mask]
    shank_ids = shank_ids[valid_mask]
    probe_ids = probe_ids[valid_mask]
    positions = {channel: index for index, channel in enumerate(original_ids)}
    device_ch_inds = np.asarray([positions[int(ch)] for ch in device_ch_inds[valid_mask]])

    probegroup = ProbeGroup()
    unique_probes = [p for p in np.unique(probe_ids) if p > 0]
    for p_id in unique_probes:
        mask = probe_ids == p_id
        if not np.any(mask):
            continue
        probe = Probe(ndim=2, si_units="um")
        probe.set_contacts(
            positions=np.column_stack((x[mask], y[mask])),
            shapes="circle",
            shape_params={"radius": 5},
            shank_ids=shank_ids[mask],
        )
        probe.set_device_channel_indices(device_ch_inds[mask])
        probegroup.add_probe(probe)

    if len(probegroup.probes) == 0:
        return recording

    if hasattr(recording, "set_probegroup"):
        try:
            recording = recording.set_probegroup(probegroup, group_mode="by_probe")
        except Exception as exc:
            warnings.warn(
                f"Failed to attach probe from chanMap ({chanmap_mat_path}): {exc}. "
                "Continuing without probe geometry.",
                RuntimeWarning,
                stacklevel=2,
            )
            return recording
    elif hasattr(recording, "set_probe"):
        try:
            recording = recording.set_probe(probegroup.probes[0])
        except Exception as exc:
            warnings.warn(
                f"Failed to attach probe from chanMap ({chanmap_mat_path}): {exc}. "
                "Continuing without probe geometry.",
                RuntimeWarning,
                stacklevel=2,
            )
            return recording

    # Keep both probe-level and shank-level group properties for artifact routing.
    try:
        ch_ids = [int(ch) for ch in recording.get_channel_ids()]
        group_props = _build_chanmap_group_properties(chanmap_mat_path, ch_ids)
        _set_channel_property_compat(recording, ch_ids, "artifact_group_probe", group_props["artifact_group_probe"])
        _set_channel_property_compat(recording, ch_ids, "artifact_group_shank", group_props["artifact_group_shank"])
        _set_channel_property_compat(recording, ch_ids, "group", group_props["artifact_group_probe"])
    except Exception as exc:
        warnings.warn(
            f"Failed to attach artifact group properties from chanMap ({chanmap_mat_path}): {exc}. "
            "Continuing with existing group assignments.",
            RuntimeWarning,
            stacklevel=2,
        )

    return recording


def apply_artifact_group_mode(
    recording: Any,
    *,
    chanmap_mat_path: Path | None,
    mode: str,
) -> Any:
    if mode not in {"all", "probe", "shank"}:
        raise ValueError(f"Unsupported artifact group mode: {mode}")

    channel_ids = [int(ch) for ch in recording.get_channel_ids()]
    if not channel_ids:
        return recording

    if mode == "all":
        _set_channel_property_compat(recording, channel_ids, "group", [0] * len(channel_ids))
        return recording

    if chanmap_mat_path is None or not Path(chanmap_mat_path).exists():
        raise ValueError(
            f"artifact group mode '{mode}' requires a valid chanMap.mat path, but none was found."
        )

    group_props = _build_chanmap_group_properties(Path(chanmap_mat_path), channel_ids)
    _set_channel_property_compat(recording, channel_ids, "artifact_group_probe", group_props["artifact_group_probe"])
    _set_channel_property_compat(recording, channel_ids, "artifact_group_shank", group_props["artifact_group_shank"])

    selected_key = "artifact_group_probe" if mode == "probe" else "artifact_group_shank"
    _set_channel_property_compat(recording, channel_ids, "group", group_props[selected_key])
    return recording


def load_subsession_recordings(
    dat_paths: list[Path],
    sampling_frequency: float,
    num_channels: int,
    dtype: str,
    gain_to_uV: float,
    offset_to_uV: float,
    recording_paths: list[Path] | None = None,
    recording_stream_names: list[str | None] | None = None,
    ephys_channel_indices_by_subsession: list[list[int] | None] | None = None,
) -> list[Any]:
    recordings = []
    recording_paths = recording_paths or dat_paths
    recording_stream_names = recording_stream_names or [None for _ in dat_paths]
    ephys_channel_indices_by_subsession = ephys_channel_indices_by_subsession or [None for _ in dat_paths]
    for p, recording_root, stream_name, ephys_channel_indices in zip(
        dat_paths,
        recording_paths,
        recording_stream_names,
        ephys_channel_indices_by_subsession,
        strict=True,
    ):
        if recording_root.is_dir() and stream_name:
            rec = se.read_openephys(
                folder_path=str(recording_root),
                stream_name=stream_name,
            )
            rec = rec.rename_channels(list(range(rec.get_num_channels())))
            if ephys_channel_indices is not None:
                rec = select_recording_channels(rec, [int(ch) for ch in ephys_channel_indices])
                rec = rec.rename_channels(list(range(rec.get_num_channels())))
        else:
            rec = se.read_binary(
                str(p),
                sampling_frequency=sampling_frequency,
                dtype=dtype,
                num_channels=num_channels,
                gain_to_uV=gain_to_uV,
                offset_to_uV=offset_to_uV,
            )
        recordings.append(rec)
    return recordings


def load_binary_recording(
    *,
    dat_path: Path,
    sampling_frequency: float,
    num_channels: int,
    dtype: str,
    gain_to_uV: float,
    offset_to_uV: float,
) -> Any:
    return se.read_binary(
        str(dat_path),
        sampling_frequency=sampling_frequency,
        dtype=dtype,
        num_channels=num_channels,
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
    )


def concatenate_recordings_si(recordings: list[Any]) -> Any:
    return si.concatenate_recordings(recordings)


def _recording_num_frames(recording: Any) -> int | None:
    try:
        return int(recording.get_num_frames(segment_index=0))
    except TypeError:
        try:
            return int(recording.get_num_frames())
        except Exception:
            return None
    except Exception:
        return None


def _validate_existing_binary_size(
    *,
    path: Path,
    dtype: str | np.dtype,
    num_channels: int,
    expected_frames: int | None = None,
    frame_tolerance: int = 0,
    label: str,
) -> None:
    dtype_np = np.dtype(dtype)
    frame_bytes = int(dtype_np.itemsize) * int(num_channels)
    if frame_bytes <= 0:
        raise ValueError(f"Invalid {label} frame size: dtype={dtype_np}, num_channels={num_channels}")
    size = int(path.stat().st_size)
    if size % frame_bytes != 0:
        raise ValueError(
            f"Existing {label} is incompatible with current settings: "
            f"{path} size={size} is not divisible by frame_bytes={frame_bytes}."
        )
    if expected_frames is not None:
        actual_frames = size // frame_bytes
        tolerance = max(0, int(frame_tolerance))
        if abs(int(actual_frames) - int(expected_frames)) > tolerance:
            tolerance_suffix = f" with tolerance {tolerance}" if tolerance else ""
            raise ValueError(
                f"Existing {label} has stale sample count: {actual_frames} != {expected_frames}"
                f"{tolerance_suffix} "
                f"({path}). Re-run with overwrite=True or remove the stale file."
            )


def write_concatenated_dat(
    recording: Any,
    output_dat_path: Path,
    dtype: str,
    overwrite: bool,
    job_kwargs: dict[str, Any],
) -> Path:
    if output_dat_path.exists() and not overwrite:
        _validate_existing_binary_size(
            path=output_dat_path,
            dtype=dtype,
            num_channels=int(recording.get_num_channels()),
            expected_frames=_recording_num_frames(recording),
            label="concatenated dat",
        )
        return output_dat_path

    partial_path = output_dat_path.with_name(
        f"{output_dat_path.name}.partial-{uuid.uuid4().hex[:12]}"
    )
    try:
        si.write_binary_recording(
            recording,
            file_paths=str(partial_path),
            add_file_extension=False,
            dtype=dtype,
            verbose=True,
            **job_kwargs,
        )
        _validate_existing_binary_size(
            path=partial_path,
            dtype=dtype,
            num_channels=int(recording.get_num_channels()),
            expected_frames=_recording_num_frames(recording),
            label="new concatenated dat",
        )
        os.replace(partial_path, output_dat_path)
    except BaseException as exc:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"Could not remove incomplete output {partial_path}: {cleanup_exc}")
        raise
    return output_dat_path


def _write_concatenated_sidecar_dat(
    *,
    dat_paths: list[Path | None],
    output_dat_path: Path,
    sampling_frequency: float,
    num_channels: int,
    dtype: str,
    overwrite: bool,
    job_kwargs: dict[str, Any],
    sample_counts: list[int] | None = None,
    source_sample_counts: list[int | None] | None = None,
    source_sampling_frequencies: list[float | None] | None = None,
    source_num_channels: list[int | None] | None = None,
    source_channel_indices: list[list[int] | None] | None = None,
    destination_channel_indices: list[list[int] | None] | None = None,
    source_dtype: str = "int16",
) -> Path | None:
    del job_kwargs
    if not dat_paths:
        return None
    if not any(p is not None and Path(p).exists() for p in dat_paths):
        return None
    layout_path = output_dat_path.with_name(f"{output_dat_path.name}.layout.json")
    layout_payload = {
        "schema_version": 1,
        "dtype": str(np.dtype(dtype)),
        "num_channels": int(num_channels),
        "sampling_frequency": float(sampling_frequency),
        "sample_counts": None if sample_counts is None else [int(n) for n in sample_counts],
        "source_sample_counts": None if source_sample_counts is None else [None if n is None else int(n) for n in source_sample_counts],
        "source_sampling_frequencies": None if source_sampling_frequencies is None else [None if rate is None else float(rate) for rate in source_sampling_frequencies],
        "source_num_channels": None if source_num_channels is None else [None if n is None else int(n) for n in source_num_channels],
        "source_channel_indices": source_channel_indices,
        "destination_channel_indices": destination_channel_indices,
    }
    if output_dat_path.exists() and not overwrite:
        nonlegacy_destinations = destination_channel_indices is not None and any(
            mapping is not None
            and len(mapping) > 0
            and [int(ch) for ch in mapping] != list(range(len(mapping)))
            for mapping in destination_channel_indices
        )
        if nonlegacy_destinations:
            if not layout_path.exists():
                raise FileExistsError(
                    "Cannot safely reuse an existing sidecar dat with non-left-packed destination channel mappings: "
                    f"{output_dat_path}. Its layout fingerprint is absent; Set overwrite=True to rebuild it."
                )
            import json
            existing_layout = json.loads(layout_path.read_text(encoding="utf-8"))
            if existing_layout != layout_payload:
                raise FileExistsError(
                    "Existing sidecar dat has an incompatible layout fingerprint: "
                    f"{output_dat_path}. Set overwrite=True to rebuild it."
                )
        expected_frames = int(sum(int(n) for n in sample_counts)) if sample_counts is not None else None
        _validate_existing_binary_size(
            path=output_dat_path,
            dtype=dtype,
            num_channels=int(num_channels),
            expected_frames=expected_frames,
            label="sidecar dat",
        )
        # Dense identity/left-packed legacy outputs are unambiguous once the
        # current source/sample geometry has validated their exact byte size.
        # Materialize that proof for subsequent persistent inventory checks.
        if not layout_path.exists():
            atomic_write_json(layout_path, layout_payload)
        return output_dat_path

    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0 for sidecar concat: {num_channels}")

    if sample_counts is not None and len(sample_counts) != len(dat_paths):
        raise ValueError(
            "sample_counts must have the same length as sidecar dat_paths: "
            f"{len(sample_counts)} != {len(dat_paths)}"
        )
    if source_sample_counts is not None and len(source_sample_counts) != len(dat_paths):
        raise ValueError(
            "source_sample_counts must have the same length as sidecar dat_paths: "
            f"{len(source_sample_counts)} != {len(dat_paths)}"
        )
    if source_sampling_frequencies is not None and len(source_sampling_frequencies) != len(dat_paths):
        raise ValueError(
            "source_sampling_frequencies must have the same length as sidecar dat_paths: "
            f"{len(source_sampling_frequencies)} != {len(dat_paths)}"
        )
    if source_num_channels is not None and len(source_num_channels) != len(dat_paths):
        raise ValueError(
            "source_num_channels must have the same length as sidecar dat_paths: "
            f"{len(source_num_channels)} != {len(dat_paths)}"
        )
    if source_channel_indices is not None and len(source_channel_indices) != len(dat_paths):
        raise ValueError(
            "source_channel_indices must have the same length as sidecar dat_paths: "
            f"{len(source_channel_indices)} != {len(dat_paths)}"
        )
    if destination_channel_indices is not None and len(destination_channel_indices) != len(dat_paths):
        raise ValueError(
            "destination_channel_indices must have the same length as sidecar dat_paths: "
            f"{len(destination_channel_indices)} != {len(dat_paths)}"
        )

    dtype_np = np.dtype(dtype)
    frame_bytes = int(dtype_np.itemsize) * int(num_channels)
    if frame_bytes <= 0:
        raise ValueError(f"Invalid sidecar frame size: dtype={dtype}, num_channels={num_channels}")
    source_dtype_np = np.dtype(source_dtype)

    # Direct binary concatenation preserves exact uint16 sidecar words and lets us
    # fill missing epochs with zeros so sidecar timestamps stay on the merged timebase.
    output_dat_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_dat_path.with_name(f".{output_dat_path.name}.partial-{uuid.uuid4().hex}")
    try:
        with open(partial_path, "wb") as fout:
            for idx, path in enumerate(dat_paths):
                expected_output_samples = None if sample_counts is None else int(sample_counts[idx])
                expected_source_samples = (
                    expected_output_samples
                    if source_sample_counts is None
                    else (
                        None
                        if source_sample_counts[idx] is None
                        else int(source_sample_counts[idx])
                    )
                )
                source_rate = (
                    float(sampling_frequency)
                    if source_sampling_frequencies is None
                    or source_sampling_frequencies[idx] is None
                    else float(source_sampling_frequencies[idx])
                )
                if sampling_frequency <= 0 or source_rate <= 0:
                    raise ValueError(
                        f"Sidecar sampling frequencies must be positive: "
                        f"source={source_rate}, output={sampling_frequency}"
                    )
                rate_ratio = source_rate / float(sampling_frequency)
                downsample_stride = int(round(rate_ratio))
                if downsample_stride < 1 or not np.isclose(rate_ratio, downsample_stride):
                    raise ValueError(
                        "Only integer-factor sidecar downsampling is supported: "
                        f"source={source_rate} Hz, output={sampling_frequency} Hz"
                    )
                src_channels = None if source_num_channels is None else source_num_channels[idx]
                selected_channels = None if source_channel_indices is None else source_channel_indices[idx]
                destination_channels = (
                None if destination_channel_indices is None else destination_channel_indices[idx]
                )
                if path is None or not Path(path).exists():
                    if expected_output_samples is None:
                        continue
                    _write_zero_sidecar_frames(
                    fout,
                    n_samples=expected_output_samples,
                    num_channels=int(num_channels),
                    dtype=dtype_np,
                    )
                    continue

                p = Path(path)
                if selected_channels is not None or destination_channels is not None or (
                src_channels is not None and int(src_channels) != int(num_channels)
                ) or downsample_stride != 1 or expected_source_samples != expected_output_samples:
                    if src_channels is None:
                        raise ValueError(f"source_num_channels is required for selected sidecar extraction: {p}")
                    channels_to_copy = (
                    [int(ch) for ch in selected_channels]
                    if selected_channels is not None
                    else list(range(int(src_channels)))
                    )
                    _write_selected_sidecar_frames(
                    fout,
                    path=p,
                    source_num_channels=int(src_channels),
                    selected_channel_indices=channels_to_copy,
                    destination_channel_indices=destination_channels,
                    output_num_channels=int(num_channels),
                    source_dtype=source_dtype_np,
                    output_dtype=dtype_np,
                    expected_source_samples=expected_source_samples,
                    expected_output_samples=expected_output_samples,
                    downsample_stride=downsample_stride,
                    )
                    continue

                size = p.stat().st_size
                if size % frame_bytes != 0:
                    raise ValueError(
                    f"{p} size is not divisible by sidecar frame size: "
                    f"size={size}, frame_bytes={frame_bytes}"
                    )
                actual_samples = size // frame_bytes
                if expected_source_samples is not None and int(actual_samples) != int(expected_source_samples):
                    raise ValueError(
                        f"{p} sample count does not match source sidecar metadata: "
                        f"{actual_samples} != {expected_source_samples}"
                    )
                with open(p, "rb") as fin:
                    while True:
                        chunk = fin.read(1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)
        expected_frames = int(sum(int(n) for n in sample_counts)) if sample_counts is not None else None
        _validate_existing_binary_size(path=partial_path, dtype=dtype_np, num_channels=num_channels,
                                       expected_frames=expected_frames, label="new sidecar dat")
        # A sidecar and its layout fingerprint form one logical output.  Keep
        # recoverable same-directory backups until both publications succeed,
        # so an injected layout failure cannot strand a new binary with stale
        # provenance or destroy the previous compatible pair.
        backup_binary = output_dat_path.with_name(f".{output_dat_path.name}.backup-{uuid.uuid4().hex}")
        backup_layout = layout_path.with_name(f".{layout_path.name}.backup-{uuid.uuid4().hex}")
        had_binary = output_dat_path.exists()
        had_layout = layout_path.exists()
        try:
            if had_binary:
                os.replace(output_dat_path, backup_binary)
            if had_layout:
                os.replace(layout_path, backup_layout)
            os.replace(partial_path, output_dat_path)
            atomic_write_json(layout_path, layout_payload)
        except Exception:
            if output_dat_path.exists():
                output_dat_path.unlink()
            if layout_path.exists():
                layout_path.unlink()
            if had_binary and backup_binary.exists():
                os.replace(backup_binary, output_dat_path)
            if had_layout and backup_layout.exists():
                os.replace(backup_layout, layout_path)
            raise
        finally:
            if backup_binary.exists():
                backup_binary.unlink()
            if backup_layout.exists():
                backup_layout.unlink()
    finally:
        if partial_path.exists():
            partial_path.unlink()
    return output_dat_path


def _write_selected_sidecar_frames(
    fout: Any,
    *,
    path: Path,
    source_num_channels: int,
    selected_channel_indices: list[int],
    destination_channel_indices: list[int] | None,
    output_num_channels: int,
    source_dtype: np.dtype,
    output_dtype: np.dtype,
    expected_source_samples: int | None,
    expected_output_samples: int | None,
    downsample_stride: int,
) -> None:
    if source_num_channels <= 0:
        raise ValueError(f"source_num_channels must be > 0 for sidecar extraction: {source_num_channels}")
    if output_num_channels <= 0:
        raise ValueError(f"output_num_channels must be > 0 for sidecar extraction: {output_num_channels}")
    if downsample_stride <= 0:
        raise ValueError(f"downsample_stride must be positive: {downsample_stride}")
    if len(selected_channel_indices) > output_num_channels:
        raise ValueError(
            "Selected sidecar channel count exceeds output channel count: "
            f"{len(selected_channel_indices)} > {output_num_channels}"
        )
    invalid = [ch for ch in selected_channel_indices if ch < 0 or ch >= source_num_channels]
    if invalid:
        raise ValueError(
            f"Selected sidecar channels are outside source range [0, {source_num_channels - 1}]: "
            f"{sorted(set(invalid))}"
        )
    if destination_channel_indices is not None:
        if len(destination_channel_indices) != len(selected_channel_indices):
            raise ValueError("Selected sidecar and destination channel mappings must have equal lengths")
        invalid_destinations = [ch for ch in destination_channel_indices if ch < 0 or ch >= output_num_channels]
        if invalid_destinations or len(set(destination_channel_indices)) != len(destination_channel_indices):
            raise ValueError("Destination sidecar channels must be unique output indices in range")

    source_frame_bytes = int(source_dtype.itemsize) * int(source_num_channels)
    size = path.stat().st_size
    if source_frame_bytes <= 0 or size % source_frame_bytes != 0:
        raise ValueError(
            f"{path} size is not divisible by source sidecar frame size: "
            f"size={size}, frame_bytes={source_frame_bytes}"
        )
    actual_samples = size // source_frame_bytes
    if expected_source_samples is not None and int(actual_samples) != int(expected_source_samples):
        raise ValueError(
            f"{path} sample count does not match source sidecar metadata: "
            f"{actual_samples} != {expected_source_samples}"
        )

    frames_per_chunk = max(1, (1024 * 1024) // source_frame_bytes)
    source_offset = 0
    written_samples = 0
    with open(path, "rb") as fin:
        while True:
            chunk = fin.read(frames_per_chunk * source_frame_bytes)
            if not chunk:
                break
            raw = np.frombuffer(chunk, dtype=source_dtype)
            if raw.size % source_num_channels != 0:
                raise ValueError(f"Partial sidecar frame encountered while reading {path}")
            source = raw.reshape(-1, source_num_channels)
            first_selected = (-source_offset) % downsample_stride
            source = source[first_selected::downsample_stride]
            source_offset += raw.size // source_num_channels
            if expected_output_samples is not None:
                remaining = int(expected_output_samples) - written_samples
                if remaining <= 0:
                    break
                source = source[:remaining]
            selected = source[:, selected_channel_indices]
            if selected.shape[1] == output_num_channels and destination_channel_indices is None:
                output = selected.astype(output_dtype, copy=False)
            else:
                output = np.zeros((selected.shape[0], output_num_channels), dtype=output_dtype)
                if selected.shape[1] > 0:
                    destinations = destination_channel_indices or list(range(selected.shape[1]))
                    output[:, destinations] = selected.astype(output_dtype, copy=False)
            fout.write(output.tobytes(order="C"))
            written_samples += int(output.shape[0])
    if expected_output_samples is not None and written_samples != int(expected_output_samples):
        raise ValueError(
            f"{path} cannot provide the requested resampled sidecar length: "
            f"wrote={written_samples}, expected={expected_output_samples}, "
            f"source_samples={actual_samples}, stride={downsample_stride}"
        )


def _write_zero_sidecar_frames(
    fout: Any,
    *,
    n_samples: int,
    num_channels: int,
    dtype: np.dtype,
) -> None:
    remaining = max(0, int(n_samples))
    if remaining == 0:
        return
    chunk_frames = 1_000_000
    zeros = np.zeros((min(chunk_frames, remaining), int(num_channels)), dtype=dtype)
    while remaining > 0:
        n = min(chunk_frames, remaining)
        if n != zeros.shape[0]:
            zeros = np.zeros((n, int(num_channels)), dtype=dtype)
        fout.write(zeros.tobytes(order="C"))
        remaining -= n


def concatenate_binary_files(
    *,
    dat_paths: list[Path | None],
    output_dat_path: Path,
    overwrite: bool,
    dtype: str,
    num_channels: int,
    sample_counts: list[int] | None = None,
) -> Path | None:
    return _write_concatenated_sidecar_dat(
        dat_paths=dat_paths,
        output_dat_path=output_dat_path,
        sampling_frequency=1.0,
        num_channels=num_channels,
        dtype=dtype,
        overwrite=overwrite,
        job_kwargs={},
        sample_counts=sample_counts,
    )


def write_concatenated_dat_analogin(
    *,
    dat_paths: list[Path | None],
    output_dat_path: Path,
    sampling_frequency: float,
    num_channels: int,
    overwrite: bool,
    job_kwargs: dict[str, Any],
    sample_counts: list[int] | None = None,
    source_sample_counts: list[int | None] | None = None,
    source_sampling_frequencies: list[float | None] | None = None,
    source_num_channels: list[int | None] | None = None,
    source_channel_indices: list[list[int] | None] | None = None,
    destination_channel_indices: list[list[int] | None] | None = None,
) -> Path | None:
    return _write_concatenated_sidecar_dat(
        dat_paths=dat_paths,
        output_dat_path=output_dat_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype="uint16",
        overwrite=overwrite,
        job_kwargs=job_kwargs,
        sample_counts=sample_counts,
        source_sample_counts=source_sample_counts,
        source_sampling_frequencies=source_sampling_frequencies,
        source_num_channels=source_num_channels,
        source_channel_indices=source_channel_indices,
        destination_channel_indices=destination_channel_indices,
        source_dtype="int16",
    )


def write_concatenated_dat_digitalin(
    *,
    dat_paths: list[Path | None],
    output_dat_path: Path,
    sampling_frequency: float,
    num_channels: int,
    overwrite: bool,
    job_kwargs: dict[str, Any],
    sample_counts: list[int] | None = None,
) -> Path | None:
    return _write_concatenated_sidecar_dat(
        dat_paths=dat_paths,
        output_dat_path=output_dat_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype="uint16",
        overwrite=overwrite,
        job_kwargs=job_kwargs,
        sample_counts=sample_counts,
    )


def _load_bad_channels_from_chanmap(chanmap_mat_path: Path) -> list[int]:
    mat = loadmat(chanmap_mat_path)
    if "connected" not in mat:
        return []

    connected = np.asarray(mat["connected"]).reshape(-1).astype(float) > 0
    device_ch_inds = _chanmap_device_channel_indices(mat)
    n = min(connected.size, device_ch_inds.size)
    if n == 0:
        return []
    return device_ch_inds[:n][~connected[:n]].astype(int).tolist()


def attach_probe_and_remove_bad_channels(
    recording: Any,
    chanmap_mat_path: Path | None,
    reject_channels_0based: list[int],
    *,
    original_num_channels: int | None = None,
) -> tuple[Any, list[int], list[int]]:
    bad = set(reject_channels_0based)

    recording_with_probe = recording
    if chanmap_mat_path is not None and Path(chanmap_mat_path).exists():
        chanmap_path = Path(chanmap_mat_path)
        bad.update(_load_bad_channels_from_chanmap(chanmap_path))
        recording_with_probe = attach_probe_from_chanmap(
            recording_with_probe, chanmap_path, original_num_channels=original_num_channels)

    if hasattr(recording_with_probe, "get_num_channels"):
        # Probe attachment may select a partial map. IDs remain original binary
        # columns, so the reduced channel count is not a valid ID bound.
        if original_num_channels is None and hasattr(recording, "get_annotation"):
            original_num_channels = recording.get_annotation("binary_num_channels")
        valid_ids = (set(range(original_num_channels)) if original_num_channels is not None
                     else set(int(ch) for ch in recording.get_channel_ids()))
        invalid_bad = sorted(bad - valid_ids)
        if invalid_bad:
            raise ValueError(f"Bad/reject channels absent from the input recording: {invalid_bad}")

    bad_0 = sorted(bad)
    bad_1 = [b + 1 for b in bad_0]

    # Keep all channels in the recording and use bad-channel metadata downstream
    # (neurocode-style behavior with rejectchannels passed to sorters/statescore).
    return recording_with_probe, bad_0, bad_1


def select_recording_channels(recording: Any, channel_ids: list[int]) -> Any:
    if hasattr(recording, "channel_slice"):
        return recording.channel_slice(channel_ids=channel_ids)
    if hasattr(recording, "select_channels"):
        return recording.select_channels(channel_ids=channel_ids)
    if hasattr(recording, "remove_channels") and hasattr(recording, "get_channel_ids"):
        existing = [int(ch) for ch in recording.get_channel_ids()]
        keep = {int(ch) for ch in channel_ids}
        remove = [ch for ch in existing if ch not in keep]
        return recording.remove_channels(channel_ids=remove)
    raise AttributeError(
        "Recording object does not support channel slicing APIs "
        "(channel_slice/select_channels/remove_channels)."
    )


def apply_preprocessing(
    recording_raw: Any,
    bandpass_min_hz: float,
    bandpass_max_hz: float,
    reference: str,
    local_radius_um: tuple[float, float],
) -> Any:
    rec_f = spre.bandpass_filter(
        recording_raw,
        freq_min=bandpass_min_hz,
        freq_max=bandpass_max_hz,
    )

    reference_normalized = str(reference).strip().lower()
    if reference_normalized in {"none", "no", "off"}:
        return rec_f

    if reference_normalized == "local":
        has_locations = True
        try:
            locations = rec_f.get_channel_locations()
        except Exception:
            has_locations = False

        if has_locations:
            isolated_channels = _local_reference_channels_without_neighbors(
                channel_ids=[int(ch) for ch in rec_f.get_channel_ids()],
                locations=locations,
                local_radius_um=local_radius_um,
            )
            if isolated_channels:
                shown = ", ".join(str(ch) for ch in isolated_channels[:20])
                suffix = ", ..." if len(isolated_channels) > 20 else ""
                raise ValueError(
                    "Local CMR cannot be applied because some channels have no "
                    "reference channels inside the local annulus "
                    f"{tuple(local_radius_um)} um. Channels: {shown}{suffix}. "
                    "Increase CMR radius max, or set common median reference to "
                    "global/none."
                )
            rec_ref = spre.common_reference(
                rec_f,
                reference="local",
                local_radius=list(local_radius_um),
                operator="median",
            )
        else:
            print("Channel locations are unavailable. Falling back to global median reference.")
            rec_ref = spre.common_reference(rec_f, reference="global", operator="median")
    elif reference_normalized == "global":
        rec_ref = spre.common_reference(rec_f, reference="global", operator="median")
    else:
        raise ValueError("reference must be one of: none, local, global")

    return rec_ref


def _local_reference_channels_without_neighbors(
    *,
    channel_ids: list[int],
    locations: Any,
    local_radius_um: tuple[float, float],
) -> list[int]:
    loc = np.asarray(locations, dtype=np.float64)
    if loc.ndim != 2 or loc.shape[0] == 0:
        return []
    n = min(len(channel_ids), loc.shape[0])
    if n <= 1:
        return [int(ch) for ch in channel_ids[:n]]

    loc = loc[:n]
    ids = [int(ch) for ch in channel_ids[:n]]
    r_min, r_max = sorted(float(v) for v in local_radius_um)
    diffs = loc[:, None, :] - loc[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    neighbor_mask = (dist >= r_min) & (dist <= r_max)
    np.fill_diagonal(neighbor_mask, False)
    return [ch for ch, has_neighbor in zip(ids, np.any(neighbor_mask, axis=1)) if not has_neighbor]


def preprocess_selected_channels_preserve_shape(
    *,
    recording_raw: Any,
    selected_channel_ids: list[int],
    bandpass_min_hz: float,
    bandpass_max_hz: float,
    reference: str,
    local_radius_um: tuple[float, float],
) -> Any:
    all_channel_ids = list(recording_raw.get_channel_ids())
    if not all_channel_ids:
        return recording_raw

    selected_set = {int(ch) for ch in selected_channel_ids}
    selected_ids_in_order = [ch for ch in all_channel_ids if int(ch) in selected_set]
    if not selected_ids_in_order:
        return recording_raw

    rec_selected = select_recording_channels(recording_raw, selected_ids_in_order)
    rec_selected_pre = apply_preprocessing(
        recording_raw=rec_selected,
        bandpass_min_hz=bandpass_min_hz,
        bandpass_max_hz=bandpass_max_hz,
        reference=reference,
        local_radius_um=local_radius_um,
    )
    return _merge_selected_with_bypass_preserve_order(
        recording_raw=recording_raw,
        recording_selected_processed=rec_selected_pre,
        selected_set=selected_set,
        all_channel_ids=all_channel_ids,
    )


def _merge_selected_with_bypass_preserve_order(
    *,
    recording_raw: Any,
    recording_selected_processed: Any,
    selected_set: set[int],
    all_channel_ids: list[int],
) -> Any:
    bypass_ids = [ch for ch in all_channel_ids if int(ch) not in selected_set]
    if not bypass_ids:
        return recording_selected_processed

    # Build channel runs in original order and aggregate run-by-run.
    # This avoids re-slicing ChannelsAggregationRecording with interleaved
    # integer channel ids, which can reorder channels in some SI code paths.
    run_recordings: list[Any] = []
    run_ids: list[int] = []
    run_is_selected: bool | None = None

    def _flush_run() -> None:
        nonlocal run_ids, run_is_selected
        if not run_ids:
            return
        src = recording_selected_processed if bool(run_is_selected) else recording_raw
        run_recordings.append(select_recording_channels(src, run_ids))
        run_ids = []

    for ch in all_channel_ids:
        ch_selected = int(ch) in selected_set
        if run_is_selected is None:
            run_is_selected = ch_selected
        elif ch_selected != run_is_selected:
            _flush_run()
            run_is_selected = ch_selected
        run_ids.append(ch)

    _flush_run()

    if len(run_recordings) == 1:
        return run_recordings[0]
    return si.aggregate_channels(run_recordings)


def apply_transform_to_selected_channels_preserve_shape(
    *,
    recording_raw: Any,
    selected_channel_ids: list[int],
    transform_fn: Callable[[Any], Any],
) -> Any:
    all_channel_ids = list(recording_raw.get_channel_ids())
    if not all_channel_ids:
        return recording_raw

    selected_set = {int(ch) for ch in selected_channel_ids}
    selected_ids_in_order = [ch for ch in all_channel_ids if int(ch) in selected_set]
    if not selected_ids_in_order:
        return recording_raw

    rec_selected = select_recording_channels(recording_raw, selected_ids_in_order)
    rec_selected_processed = transform_fn(rec_selected)
    return _merge_selected_with_bypass_preserve_order(
        recording_raw=recording_raw,
        recording_selected_processed=rec_selected_processed,
        selected_set=selected_set,
        all_channel_ids=all_channel_ids,
    )


def zero_selected_channels_preserve_shape(
    *,
    recording_raw: Any,
    selected_channel_ids: list[int],
) -> Any:
    target_dtype = recording_raw.get_dtype() if hasattr(recording_raw, "get_dtype") else None

    def _zero_and_cast(rec_sel: Any) -> Any:
        rec_zero = spre.scale(rec_sel, gain=0.0, offset=0.0)
        if target_dtype is not None:
            rec_zero = spre.astype(rec_zero, dtype=target_dtype)
        return rec_zero

    return apply_transform_to_selected_channels_preserve_shape(
        recording_raw=recording_raw,
        selected_channel_ids=selected_channel_ids,
        transform_fn=_zero_and_cast,
    )


def write_lfp(
    recording_raw: Any,
    lfp_path: Path,
    lfp_fs: float,
    dtype: str,
    overwrite: bool,
    job_kwargs: dict[str, Any],
) -> Path:
    if isinstance(lfp_fs, float):
        if not float(lfp_fs).is_integer():
            raise ValueError(f"lfp_fs must be an integer Hz for spikeinterface.resample: got {lfp_fs}")
        lfp_rate = int(lfp_fs)
    else:
        lfp_rate = int(lfp_fs)

    input_fs = float(recording_raw.get_sampling_frequency())
    if lfp_path.exists() and not overwrite:
        n_channels = int(recording_raw.get_num_channels()) if hasattr(recording_raw, "get_num_channels") else 1
        raw_frames = _recording_num_frames(recording_raw)
        expected_lfp_frames = (
            int(round(float(raw_frames) * float(lfp_rate) / input_fs))
            if raw_frames is not None and input_fs > 0
            else None
        )
        _validate_existing_binary_size(
            path=lfp_path,
            dtype=dtype,
            num_channels=n_channels,
            expected_frames=expected_lfp_frames,
            frame_tolerance=1,
            label="lfp",
        )
        return lfp_path

    # MATLAB neurocode parity: apply explicit 450 Hz low-pass before downsampling.
    lowpass_hz = 450.0
    filt_order = 5
    nyquist_out = float(lfp_rate) / 2.0
    if lowpass_hz >= nyquist_out:
        adjusted = max(nyquist_out - 1.0, 1.0)
        warnings.warn(
            f"Adjusting LFP low-pass from {lowpass_hz:.1f} Hz to {adjusted:.1f} Hz "
            f"because it exceeds output Nyquist ({nyquist_out:.1f} Hz).",
            RuntimeWarning,
            stacklevel=2,
        )
        lowpass_hz = adjusted

    nyquist_in = input_fs / 2.0
    if lowpass_hz >= nyquist_in:
        adjusted = max(nyquist_in - 1.0, 1.0)
        warnings.warn(
            f"Adjusting LFP low-pass from {lowpass_hz:.1f} Hz to {adjusted:.1f} Hz "
            f"because it exceeds input Nyquist ({nyquist_in:.1f} Hz).",
            RuntimeWarning,
            stacklevel=2,
        )
        lowpass_hz = adjusted

    lfp_sos = signal.iirfilter(
        filt_order,
        lowpass_hz,
        btype="lowpass",
        ftype="butter",
        fs=input_fs,
        output="sos",
    )
    rec_lfp_prefiltered = spre.filter(
        recording_raw,
        coeff=lfp_sos,
        filter_mode="sos",
        direction="forward-backward",
    )
    rec_lfp = spre.resample(rec_lfp_prefiltered, resample_rate=lfp_rate)
    partial_path = lfp_path.with_name(f"{lfp_path.name}.partial-{uuid.uuid4().hex[:12]}")
    try:
        si.write_binary_recording(
            rec_lfp,
            file_paths=str(partial_path),
            add_file_extension=False,
            dtype=dtype,
            verbose=True,
            **job_kwargs,
        )
        n_channels = (
            int(recording_raw.get_num_channels())
            if hasattr(recording_raw, "get_num_channels")
            else 1
        )
        raw_frames = _recording_num_frames(recording_raw)
        expected_lfp_frames = (
            int(round(float(raw_frames) * float(lfp_rate) / input_fs))
            if raw_frames is not None and input_fs > 0
            else None
        )
        _validate_existing_binary_size(
            path=partial_path,
            dtype=dtype,
            num_channels=n_channels,
            expected_frames=expected_lfp_frames,
            frame_tolerance=1,
            label="new lfp",
        )
        os.replace(partial_path, lfp_path)
    except BaseException as exc:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"Could not remove incomplete output {partial_path}: {cleanup_exc}")
        raise
    return lfp_path
