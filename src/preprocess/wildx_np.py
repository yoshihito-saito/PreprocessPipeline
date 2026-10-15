"""WILDX NP applied receipts, original-column geometry, and bounded excerpts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


RECEIPT_FORMAT = "WILDX NP applied probe receipt"
NP1_PART = "PRB_1_4_0480_1"
CHANNELS_PER_PROBE = 384


@dataclass(frozen=True)
class WildxNpRecording:
    directory: Path
    ap_path: Path
    lfp_path: Path | None
    sampling_frequency: float
    lfp_sampling_frequency: float
    num_channels: int
    sample_count: int
    lfp_sample_count: int
    probes: tuple[dict, ...]
    metadata: dict[str, str]


def np1_site_geometry() -> dict[str, np.ndarray]:
    """NP1 physical site IDs and micrometre coordinates, independent of banks.

    Coordinates follow Open Ephys Geometry::NP1, commit
    8c2d85a2ace63fd873b1f74fb7516a660b30562d, Geometry.cpp lines 170-213.
    The three reference sites are retained in the catalog but are not contacts
    for ordinary neural sorting. The coordinate origin matches that source.
    """
    sites = np.arange(960, dtype=np.int64)
    return {
        "site_ids": sites,
        "xcoords": np.asarray([27., 59., 11., 43.])[sites % 4],
        "ycoords": (sites // 2).astype(float) * 20.,
        "shank_ids": np.zeros(sites.size, dtype=np.int64),
        "is_reference": np.isin(sites, [191, 575, 959]),
    }


def _read_key_values(path: Path) -> dict[str, str]:
    return dict(line.strip().split("=", 1) for line in path.read_text(
        encoding="utf-8-sig").splitlines() if "=" in line and not line.lstrip().startswith("#"))


def _positive_number(metadata: dict[str, str], key: str, *, integer: bool = False):
    value = float(metadata[key])
    if not np.isfinite(value) or value <= 0 or (integer and value != int(value)):
        raise ValueError(f"WILDX NP {key} must be a positive {'integer' if integer else 'number'}")
    return int(value) if integer else value


def find_wildx_np_recording_directories(basepath: Path) -> list[Path]:
    """Find NP markers in a segment, capture, or selected day folder."""
    basepath = Path(basepath)
    if not basepath.is_dir():
        return []
    children = sorted(child for child in basepath.iterdir() if child.is_dir())
    candidates = [basepath, *children]
    for capture in children:
        if capture.name.startswith("NP_SESSION_"):
            candidates.extend(segment for segment in sorted(capture.iterdir())
                              if segment.is_dir() and segment.name.startswith("segment_"))
    return [path for path in candidates if (path / "neuropixels_ap.dat").exists()
            or any(path.glob("neuropixels_probe*_metadata.json"))]


def has_wildx_np_marker(basepath: Path) -> bool:
    """Find candidates cheaply; the reader still validates format and version."""
    return bool(find_wildx_np_recording_directories(basepath))


def read_wildx_np_recording(directory: Path) -> WildxNpRecording | None:
    """Recognize by the versioned receipt; validate the saved stream contract."""
    directory = Path(directory)
    receipts = sorted(directory.glob("neuropixels_probe*_metadata.json"))
    ap_path = directory / "neuropixels_ap.dat"
    if not receipts and not ap_path.exists():
        return None
    if not receipts:
        raise ValueError(f"WILDX NP AP input has no applied-probe receipt: {directory}")
    probes = []
    for path in receipts:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("format") != RECEIPT_FORMAT or payload.get("format_version") != 1:
            raise ValueError(f"Unsupported WILDX NP receipt format/version: {path}")
        identity = payload["identity"]
        if identity.get("model") != "Neuropixels 1.0" or identity.get("part") != NP1_PART:
            raise ValueError(f"Unsupported WILDX NP probe geometry: {identity} ({path})")
        configuration = payload["configuration"]
        if configuration.get("state") != "applied":
            raise ValueError(f"WILDX NP receipt does not contain applied settings: {path}")
        channels = configuration["channels"]
        ids = [row["output_channel"] for row in channels]
        if (any(type(ch) is not int for ch in ids)
                or sorted(ids) != list(range(CHANNELS_PER_PROBE))):
            raise ValueError(f"WILDX NP receipt must define each hardware output channel once: {path}")
        for row in channels:
            site = row["site"]
            if site is not None and (type(site) is not int or not 0 <= site < 960):
                raise ValueError(f"Invalid NP1 site in {path}: {site}")
            if type(row["connected"]) is not bool or type(row["standby"]) is not bool:
                raise ValueError(f"WILDX NP connected/standby must be booleans: {path}")
            if row["connected"] and not row["standby"] and site is None:
                raise ValueError(f"Connected WILDX NP channel has no site: {path}")
            if site is not None and site % CHANNELS_PER_PROBE != row["output_channel"]:
                raise ValueError(f"NP1 site and hardware output channel disagree: {path}")
        port = payload["physical_port"]
        if type(port) is not int or port <= 0:
            raise ValueError(f"Invalid WILDX NP physical port: {path}")
        probes.append(payload)
    probes.sort(key=lambda probe: probe["physical_port"])
    ports = [probe["physical_port"] for probe in probes]
    if len(set(ports)) != len(ports):
        raise ValueError(f"Duplicate WILDX NP physical ports: {directory}")
    metadata_path = directory / "neuropixels_probes.txt"
    metadata = _read_key_values(metadata_path)
    if metadata.get("complete") != "True":
        raise ValueError(f"Incomplete WILDX NP segments require gap handling: {directory}")
    if (metadata.get("dat_layout") != "merged"
            or metadata.get("merged_channel_order") != "ascending physical probe; 384 contiguous channels per probe"
            or metadata.get("physical_ports_known") != "True"):
        raise ValueError(f"Unsupported WILDX NP saved channel layout: {metadata_path}")
    if metadata.get("codes") != "unsigned 10-bit ADC codes stored in little-endian int16; no calibration applied":
        raise ValueError(f"Unsupported WILDX NP voltage storage: {metadata_path}")
    n_channels = _positive_number(metadata, "dat_channel_count", integer=True)
    n_probes = _positive_number(metadata, "probe_count", integer=True)
    if n_probes != len(probes) or n_channels != len(probes) * CHANNELS_PER_PROBE:
        raise ValueError(f"WILDX NP receipt count and saved binary width disagree: {directory}")
    mask = sum(1 << (port - 1) for port in ports)
    if int(metadata["probe_mask"]) != mask:
        raise ValueError(f"WILDX NP probe mask and physical ports disagree: {directory}")
    fs = _positive_number(metadata, "output_ap_sample_rate_hz")
    lfp_fs = _positive_number(metadata, "output_lfp_sample_rate_hz")
    samples = _positive_number(metadata, "merged_ap_output_rows", integer=True)
    lfp_samples = _positive_number(metadata, "merged_lfp_output_rows", integer=True)
    lfp_path = directory / "neuropixels_lfp.dat"
    for path, count in ((ap_path, samples), (lfp_path, lfp_samples)):
        if path.exists():
            expected = count * n_channels * 2
            if path.stat().st_size != expected:
                raise ValueError(f"WILDX NP file size disagrees with metadata: {path}, expected {expected} bytes")
        elif path == ap_path:
            raise FileNotFoundError(path)
    return WildxNpRecording(directory, ap_path, lfp_path if lfp_path.exists() else None,
                            fs, lfp_fs, n_channels, samples, lfp_samples, tuple(probes), metadata)


def channel_map_data(recording: WildxNpRecording, reject_channels: list[int] = ()) -> dict:
    """Join (physical port, hardware output channel) to sites and saved columns."""
    geometry = np1_site_geometry()
    n = recording.num_channels
    ids = np.arange(n, dtype=np.int64)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    sites = np.full(n, -1, dtype=np.int64)
    connected = np.zeros(n, dtype=bool)
    probe_ids = np.empty(n, dtype=np.int64)
    hardware_ids = np.empty(n, dtype=np.int64)
    reasons = np.empty(n, dtype=object)
    rejected = set(int(ch) for ch in reject_channels)
    if any(ch < 0 or ch >= n for ch in rejected):
        raise ValueError("Rejected WILDX NP channel is outside the saved binary width")
    for probe_index, probe in enumerate(recording.probes):
        for row in probe["configuration"]["channels"]:
            hardware = row["output_channel"]
            column = probe_index * CHANNELS_PER_PROBE + hardware
            site = row["site"]
            probe_ids[column] = probe["physical_port"]
            hardware_ids[column] = hardware
            if site is not None:
                sites[column] = site
                x[column] = geometry["xcoords"][site]
                y[column] = geometry["ycoords"][site]
            reason = []
            if not row["connected"]:
                reason.append("disconnected")
            if row["standby"]:
                reason.append("standby")
            if site is None:
                reason.append("missing_site")
            elif geometry["is_reference"][site]:
                reason.append("reference_site")
            if column in rejected:
                reason.append("manual_reject")
            reasons[column] = ";".join(reason)
            connected[column] = not reason
    return {"chanMap": (ids + 1).reshape(-1, 1), "chanMap0ind": ids.reshape(-1, 1),
            "connected": connected.reshape(-1, 1), "xcoords": x.reshape(-1, 1),
            "ycoords": y.reshape(-1, 1), "kcoords": np.ones((n, 1), dtype=np.int64),
            "probe_ids": probe_ids.reshape(-1, 1), "site_ids": sites.reshape(-1, 1),
            "hardware_channel_ids": hardware_ids.reshape(-1, 1),
            "bad_channel_reason": reasons.reshape(-1, 1),
            "fs": float(recording.sampling_frequency), "recording_format": "wildx_np"}


def nominal_gain_to_uV(recording: WildxNpRecording, stream: str) -> float:
    """Uncalibrated NP1 ADC transfer, using the receipt's applied gain indices.

    Open Ephys Neuropixels1.cpp at the geometry commit above defines the gain
    table (lines 84-100) and 1.2 V / 1024 / gain conversion (lines 471-484).
    This supplies a nominal count scale, not per-channel calibration or a DC
    zero point. A single scalar output scale requires uniform applied gains.
    """
    gains = (50., 125., 250., 500., 1000., 1500., 2000., 3000.)
    key = "ap_gain_index" if stream == "ap" else "lfp_gain_index"
    indices = {row[key] for probe in recording.probes
               for row in probe["configuration"]["channels"]}
    if (len(indices) != 1 or any(type(index) is not int or not 0 <= index < len(gains)
                                 for index in indices)):
        raise ValueError(f"WILDX NP {stream} requires uniform valid gain indices for scalar output scaling")
    return 1.2e6 / 1024. / gains[next(iter(indices))]


def session_recordings(paths: list[Path]) -> list[WildxNpRecording]:
    """Resolve the initial single-segment NP route without inventing gap timing."""
    recordings = [read_wildx_np_recording(Path(path).parent)
                  if Path(path).name == "neuropixels_ap.dat" else None for path in paths]
    if not any(recording is not None for recording in recordings):
        return []
    if any(recording is None for recording in recordings):
        raise ValueError("Mixing WILDX NP and other acquisition formats in one output is unsupported")
    if len(recordings) != 1:
        raise ValueError("WILDX NP currently accepts one complete segment per session; multiple segments require gap/time mapping")
    return recordings


def prepare_session_metadata(
    recordings: list[WildxNpRecording], *, output_dir: Path, basename: str,
    reject_channels: list[int], overwrite: bool,
) -> tuple[Path, Path]:
    """Write local XML and chanMap from applied settings without source edits."""
    import xml.etree.ElementTree as ET
    from scipy.io import loadmat
    from .io import atomic_savemat, atomic_write_json, atomic_write_path, save_cell_explorer_chan_coords

    first = recordings[0]
    mapping = channel_map_data(first, reject_channels)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chanmap_path = output_dir / "chanMap.mat"
    if chanmap_path.exists() and not overwrite:
        saved = loadmat(chanmap_path, simplify_cells=True)
        keys = ("chanMap0ind", "connected", "xcoords", "ycoords", "kcoords", "probe_ids", "site_ids", "hardware_channel_ids")
        if any(key not in saved or not np.array_equal(
                np.asarray(saved[key]).reshape(-1), np.asarray(mapping[key]).reshape(-1), equal_nan=True)
               for key in keys):
            raise ValueError(f"Existing WILDX NP chanMap disagrees with applied settings: {chanmap_path}")
    else:
        atomic_savemat(chanmap_path, mapping, required_key="chanMap")
    save_cell_explorer_chan_coords(
        chanmap_data=mapping, output_dir=output_dir, basename=basename,
        source="WILDX NP1 applied physical sites", overwrite=overwrite)
    root = ET.Element("parameters")
    general = ET.SubElement(root, "generalInfo")
    ET.SubElement(general, "description").text = "WILDX NP1 receipt-derived original-column channel groups"
    acquisition = ET.SubElement(root, "acquisitionSystem")
    for key, value in (("nChannels", first.num_channels), ("samplingRate", first.sampling_frequency),
                       ("nBits", 16)):
        ET.SubElement(acquisition, key).text = str(value)
    field_potentials = ET.SubElement(root, "fieldPotentials")
    ET.SubElement(field_potentials, "lfpSamplingRate").text = str(first.lfp_sampling_frequency)
    ports = np.asarray(mapping["probe_ids"]).reshape(-1)
    connected = np.asarray(mapping["connected"]).reshape(-1)
    for section in ("anatomicalDescription", "spikeDetection"):
        groups = ET.SubElement(ET.SubElement(root, section), "channelGroups")
        for port in sorted(set(ports.tolist())):
            channels = np.flatnonzero(ports == port)
            if section == "spikeDetection":
                channels = channels[connected[channels]]
            if channels.size == 0:
                continue
            group = ET.SubElement(groups, "group")
            channel_nodes = ET.SubElement(group, "channels")
            for column in channels:
                attributes = {"skip": "0" if connected[column] else "1"}
                ET.SubElement(channel_nodes, "channel", attributes).text = str(int(column))
    ET.indent(root)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    xml_path = output_dir / f"{basename}.xml"
    if xml_path.exists() and not overwrite:
        if xml_path.read_bytes() != xml_bytes:
            raise ValueError(f"Existing WILDX NP XML disagrees with applied settings: {xml_path}")
    else:
        atomic_write_path(xml_path, lambda path: path.write_bytes(xml_bytes), validator=ET.parse)
    sources = []
    for recording in recordings:
        sources.append({"directory": str(recording.directory), "ap_file": str(recording.ap_path),
                        "lfp_file": str(recording.lfp_path) if recording.lfp_path else None,
                        "ap_samples": recording.sample_count, "lfp_samples": recording.lfp_sample_count,
                        "stream_metadata": recording.metadata,
                        "probes": [{"physical_port": p["physical_port"], "identity": p["identity"],
                                    "configuration_id_sha256": p["configuration"]["configuration_id_sha256"]}
                                   for p in recording.probes]})
    atomic_write_json(output_dir / "wildx_np_metadata.json", {
        "format": "WILDX NP preprocessing input", "format_version": 1,
        "source_type": "wildx_np", "dtype": "<i2", "num_channels": first.num_channels,
        "sampling_frequency": first.sampling_frequency, "lfp_sampling_frequency": first.lfp_sampling_frequency,
        "ap_gain_to_uV": nominal_gain_to_uV(first, "ap"),
        "lfp_gain_to_uV": nominal_gain_to_uV(first, "lfp"),
        "calibration_applied": False, "adc_zero_code": None,
        "lfp_time_origin": "first saved LF row", "ap_lfp_alignment_verified": False,
        "inactive_columns_0based": np.flatnonzero(~connected).tolist(), "sources": sources})
    return xml_path, chanmap_path


def write_recorded_lfp(
    recordings: list[WildxNpRecording], *, output_path: Path, lfp_fs: float,
    overwrite: bool, job_kwargs: dict,
) -> Path:
    """Preserve saved LF samples and their own time origin; do not filter AP."""
    from .io import atomic_write_json
    from .recording import load_subsession_recordings, concatenate_recordings_si, write_concatenated_dat
    first = recordings[0]
    if any(recording.lfp_path is None for recording in recordings):
        raise FileNotFoundError("Recorded WILDX NP LF input is missing")
    if any(recording.lfp_sampling_frequency != float(lfp_fs) for recording in recordings):
        raise ValueError("WILDX NP recorded LF output must retain its saved sampling rate")
    lfp_gain = nominal_gain_to_uV(first, "lfp")
    inputs = load_subsession_recordings(
        [recording.lfp_path for recording in recordings], sampling_frequency=float(lfp_fs),
        num_channels=first.num_channels, dtype="<i2", gain_to_uV=lfp_gain, offset_to_uV=0.)
    write_concatenated_dat(recording=concatenate_recordings_si(inputs), output_dat_path=output_path,
                           dtype="int16", overwrite=overwrite, job_kwargs=job_kwargs)
    atomic_write_json(output_path.with_suffix(".lfp_metadata.json"), {
        "source": "recorded WILDX NP LF", "sampling_frequency": float(lfp_fs),
        "gain_to_uV": lfp_gain, "calibration_applied": False, "adc_zero_code": None,
        "time_origin": "first saved LF row", "ap_alignment_verified": False,
        "source_files": [str(recording.lfp_path) for recording in recordings],
        "sample_counts": [recording.lfp_sample_count for recording in recordings]})
    return output_path


def copy_first_seconds(source: Path, destination: Path, seconds: float) -> Path:
    """Copy whole frames from each saved neural stream; never modify the source."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("The excerpt must be outside its source recording")
    recording = read_wildx_np_recording(source)
    if recording is None:
        raise ValueError(f"No WILDX NP recording found: {source}")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Excerpt duration must be positive")
    ap_samples = int(round(seconds * recording.sampling_frequency))
    lfp_samples = int(round(seconds * recording.lfp_sampling_frequency))
    streams = [(recording.ap_path, ap_samples, recording.sample_count)]
    if recording.lfp_path is not None:
        streams.append((recording.lfp_path, lfp_samples, recording.lfp_sample_count))
    for _, count, available in streams:
        if count <= 0 or count > available:
            raise ValueError("Requested excerpt exceeds the saved recording")
    required = sum(count * recording.num_channels * 2 for _, count, _ in streams)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError(f"Insufficient local space for {required} excerpt bytes")
    destination.mkdir(exist_ok=False)
    copied = []
    for path, count, source_count in streams:
        byte_count = count * recording.num_channels * 2
        digest = hashlib.sha256()
        target = destination / path.name
        partial = target.with_suffix(target.suffix + ".partial")
        with path.open("rb") as reader, partial.open("xb") as writer:
            remaining = byte_count
            while remaining:
                block = reader.read(min(16 * 1024 * 1024, remaining))
                if not block:
                    raise OSError(f"Unexpected end of recording while copying: {path}")
                writer.write(block)
                digest.update(block)
                remaining -= len(block)
        partial.rename(target)
        copied.append({"file": path.name, "source": str(path), "source_samples": source_count,
                       "start_sample": 0, "sample_count": count, "bytes": byte_count,
                       "sha256": digest.hexdigest()})
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".json", ".npb1"}:
            shutil.copy2(path, destination / path.name)
    shutil.copy2(source / "neuropixels_probes.txt", destination / "neuropixels_probes.source.txt")
    capture_status = source / "neuropixels_capture_status.txt"
    if capture_status.exists():
        shutil.copy2(capture_status, destination / "neuropixels_capture_status.source.txt")
    metadata = dict(recording.metadata, merged_ap_output_rows=str(ap_samples),
                    merged_lfp_output_rows=str(lfp_samples), excerpt="True",
                    excerpt_duration_seconds=str(seconds), source_metadata="neuropixels_probes.source.txt")
    (destination / "neuropixels_probes.txt").write_text(
        "# Local prefix excerpt; source per-probe acquisition statistics are retained below.\n"
        + "\n".join(f"{key}={value}" for key, value in metadata.items()) + "\n", encoding="utf-8")
    (destination / "neuropixels_excerpt.json").write_text(json.dumps({
        "format": "WILDX NP local prefix excerpt", "format_version": 1,
        "source_directory": str(source), "requested_seconds": seconds,
        "streams": copied, "sensor_streams_copied": False,
        "source_metadata": "neuropixels_probes.source.txt"}, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a bounded WILDX NP recording prefix locally")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600.)
    args = parser.parse_args()
    print(copy_first_seconds(args.source, args.destination, args.seconds), flush=True)


if __name__ == "__main__":
    main()
