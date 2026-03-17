from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
from scipy.io import loadmat
from scipy import signal

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre


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


def _build_chanmap_group_properties(chanmap_mat_path: Path, channel_ids: list[int]) -> dict[str, list[int]]:
    mat = loadmat(chanmap_mat_path)
    required = {"kcoords", "chanMap"}
    if not required.issubset(set(mat.keys())):
        missing = ", ".join(sorted(required.difference(set(mat.keys()))))
        raise ValueError(f"chanMap is missing required fields: {missing}")

    device_ch_inds = np.asarray(mat["chanMap"]).flatten().astype(int) - 1
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


def attach_probe_from_chanmap(recording: Any, chanmap_mat_path: Path) -> Any:
    mat = loadmat(chanmap_mat_path)
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
    device_ch_inds = np.asarray(mat["chanMap"]).flatten().astype(int) - 1

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
    valid_mask = (device_ch_inds >= 0) & (device_ch_inds < n_recording_channels)
    if not np.any(valid_mask):
        warnings.warn(
            "chanMap has no valid device_channel_indices for this recording; "
            "skipping probe attachment.",
            RuntimeWarning,
            stacklevel=2,
        )
        return recording
    if int(np.size(valid_mask)) != int(np.count_nonzero(valid_mask)):
        dropped = int(np.size(valid_mask) - np.count_nonzero(valid_mask))
        warnings.warn(
            f"Dropping {dropped} chanMap contacts outside recording channel range "
            f"[0, {n_recording_channels - 1}] before probe attachment.",
            RuntimeWarning,
            stacklevel=2,
        )

    x = x[valid_mask]
    y = y[valid_mask]
    shank_ids = shank_ids[valid_mask]
    probe_ids = probe_ids[valid_mask]
    device_ch_inds = device_ch_inds[valid_mask]

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
) -> list[Any]:
    recordings = []
    recording_paths = recording_paths or dat_paths
    recording_stream_names = recording_stream_names or [None for _ in dat_paths]
    for p, recording_root, stream_name in zip(dat_paths, recording_paths, recording_stream_names, strict=True):
        if recording_root.is_dir() and stream_name:
            rec = se.read_openephys(
                folder_path=str(recording_root),
                stream_name=stream_name,
            )
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


def write_concatenated_dat(
    recording: Any,
    output_dat_path: Path,
    overwrite: bool,
    job_kwargs: dict[str, Any],
) -> Path:
    if output_dat_path.exists() and not overwrite:
        return output_dat_path

    si.write_binary_recording(
        recording,
        file_paths=str(output_dat_path),
        add_file_extension=False,
        verbose=True,
        **job_kwargs,
    )
    return output_dat_path


def _load_sidecar_recordings_si(
    *,
    dat_paths: list[Path],
    sampling_frequency: float,
    num_channels: int,
    dtype: str,
) -> list[Any]:
    recordings: list[Any] = []
    for p in dat_paths:
        rec = se.read_binary(
            str(p),
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            num_channels=num_channels,
            gain_to_uV=1.0,
            offset_to_uV=0.0,
        )
        recordings.append(rec)
    return recordings


def _write_concatenated_sidecar_dat(
    *,
    dat_paths: list[Path],
    output_dat_path: Path,
    sampling_frequency: float,
    num_channels: int,
    dtype: str,
    overwrite: bool,
    job_kwargs: dict[str, Any],
) -> Path | None:
    if not dat_paths:
        return None
    if output_dat_path.exists() and not overwrite:
        return output_dat_path

    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0 for sidecar concat: {num_channels}")

    recs = _load_sidecar_recordings_si(
        dat_paths=dat_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
    )
    rec_concat = si.concatenate_recordings(recs)
    si.write_binary_recording(
        rec_concat,
        file_paths=str(output_dat_path),
        add_file_extension=False,
        dtype=dtype,
        verbose=True,
        **job_kwargs,
    )
    return output_dat_path


def write_concatenated_dat_analogin(
    *,
    dat_paths: list[Path],
    output_dat_path: Path,
    sampling_frequency: float,
    num_channels: int,
    overwrite: bool,
    job_kwargs: dict[str, Any],
) -> Path | None:
    return _write_concatenated_sidecar_dat(
        dat_paths=dat_paths,
        output_dat_path=output_dat_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype="uint16",
        overwrite=overwrite,
        job_kwargs=job_kwargs,
    )


def write_concatenated_dat_digitalin(
    *,
    dat_paths: list[Path],
    output_dat_path: Path,
    sampling_frequency: float,
    num_channels: int,
    overwrite: bool,
    job_kwargs: dict[str, Any],
) -> Path | None:
    return _write_concatenated_sidecar_dat(
        dat_paths=dat_paths,
        output_dat_path=output_dat_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype="uint16",
        overwrite=overwrite,
        job_kwargs=job_kwargs,
    )


def _load_bad_channels_from_chanmap(chanmap_mat_path: Path) -> list[int]:
    mat = loadmat(chanmap_mat_path)
    if "connected" not in mat:
        return []

    connected = np.asarray(mat["connected"]).squeeze()
    if connected.dtype != bool:
        connected = connected.astype(int) != 0
    return np.flatnonzero(~connected).astype(int).tolist()


def attach_probe_and_remove_bad_channels(
    recording: Any,
    chanmap_mat_path: Path | None,
    reject_channels_0based: list[int],
) -> tuple[Any, list[int], list[int]]:
    bad = set(reject_channels_0based)

    recording_with_probe = recording
    if chanmap_mat_path is not None and Path(chanmap_mat_path).exists():
        chanmap_path = Path(chanmap_mat_path)
        bad.update(_load_bad_channels_from_chanmap(chanmap_path))
        recording_with_probe = attach_probe_from_chanmap(recording_with_probe, chanmap_path)

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

    if reference == "local":
        has_locations = True
        try:
            _ = rec_f.get_channel_locations()
        except Exception:
            has_locations = False

        if has_locations:
            rec_ref = spre.common_reference(
                rec_f,
                reference="local",
                local_radius=list(local_radius_um),
                operator="median",
            )
        else:
            print("Channel locations are unavailable. Falling back to global median reference.")
            rec_ref = spre.common_reference(rec_f, reference="global", operator="median")
    else:
        rec_ref = spre.common_reference(rec_f, reference="global", operator="median")

    return rec_ref


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
    if lfp_path.exists() and not overwrite:
        return lfp_path

    if isinstance(lfp_fs, float):
        if not float(lfp_fs).is_integer():
            raise ValueError(f"lfp_fs must be an integer Hz for spikeinterface.resample: got {lfp_fs}")
        lfp_rate = int(lfp_fs)
    else:
        lfp_rate = int(lfp_fs)

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

    input_fs = float(recording_raw.get_sampling_frequency())
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
    si.write_binary_recording(
        rec_lfp,
        file_paths=str(lfp_path),
        add_file_extension=False,
        dtype=dtype,
        verbose=True,
        **job_kwargs,
    )
    return lfp_path
