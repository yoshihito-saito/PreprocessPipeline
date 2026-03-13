from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import os
import shutil
from pathlib import Path
from shutil import copy2
import json
import re
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat, savemat

from .intan_rhd import IntanRhdHeader
from .metafile import (
    AcquisitionCatalog,
    PreprocessConfig,
    PreprocessResult,
    SessionXmlMeta,
    XmlMeta,
)


_OPENEPHYS_DATETIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
_OPENEPHYS_RECORD_NODE_NAME = "Record Node 101"


def extract_datetime(path: str) -> datetime:
    m = re.search(r"(\d{6}_\d{6})", path)
    if m:
        return datetime.strptime(m.group(1), "%y%m%d_%H%M%S")
    return datetime.min


def select_folder(initial_drive: str = "T:\\") -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    root.update()

    selected = filedialog.askdirectory(
        title="Select data folder",
        initialdir=initial_drive if os.path.exists(initial_drive) else os.getcwd(),
    )
    root.destroy()
    return selected or None


def select_basepath(
    *,
    use_gui: bool = True,
    manual_basepath: str | Path | None = None,
    initial_drive: str = "T:\\",
) -> Path:
    if use_gui:
        try:
            selected = select_folder(initial_drive=initial_drive)
        except Exception as exc:
            selected = None
            print(f"GUI selection failed: {exc}")
        if selected:
            return Path(selected)
        if manual_basepath is None:
            raise RuntimeError(
                "GUI selection returned no folder. Set use_gui=False and provide manual_basepath."
            )

    if manual_basepath is None:
        raise ValueError("manual_basepath is required when use_gui=False")

    p = Path(manual_basepath)
    if not p.exists() or not p.is_dir():
        raise NotADirectoryError(f"Invalid manual_basepath: {p}")
    return p.resolve()


def select_paths_with_gui(
    *,
    initial_drive: str = "T:\\",
    local_root: Path | None = None,
    use_gui: bool = True,
    manual_basepath: str | Path | None = None,
) -> tuple[Path, str, Path, Path]:
    basepath = select_basepath(
        use_gui=use_gui,
        manual_basepath=manual_basepath,
        initial_drive=initial_drive,
    )
    basename = basepath.name

    if local_root is None:
        local_root = Path.cwd() / "sorting_temp"
    local_output_dir = (local_root / basename).resolve()
    local_output_dir.mkdir(parents=True, exist_ok=True)

    xml_path = ensure_xml(basepath=basepath, local_output_dir=local_output_dir, basename=basename)
    return basepath, basename, local_output_dir, xml_path


def get_sampling_rate(xml_path: Path | str) -> float | None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        sr_tag = root.find(".//sampleRate")
        if sr_tag is None:
            acq = root.find(".//acquisitionSystem")
            if acq is not None:
                sr_tag = acq.find("samplingRate")
        return float(sr_tag.text) if sr_tag is not None and sr_tag.text else None
    except (AttributeError, ValueError):
        return None


def _clean_text(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text if text else None


def _group_channel_nodes(group: ET.Element) -> list[ET.Element]:
    channels_container = group.find("channels")
    if channels_container is not None:
        tags = channels_container.findall("n")
        if not tags:
            tags = channels_container.findall("channel")
        return tags
    tags = group.findall("n")
    if not tags:
        tags = group.findall("channel")
    return tags


def _parse_group_channels(group: ET.Element) -> list[int]:
    channels: list[int] = []
    for ch in _group_channel_nodes(group):
        try:
            if ch.text and ch.text.strip():
                channels.append(int(ch.text.strip()))
        except (ValueError, TypeError):
            continue
    return channels


def _parse_xml_channel_groups(root: ET.Element) -> tuple[list[list[int]], list[list[int]], list[int]]:
    anat_grps: list[list[int]] = []
    spike_grps: list[list[int]] = []
    skipped_channels: list[int] = []

    anat_desc = root.find("anatomicalDescription")
    if anat_desc is not None:
        ch_grps = anat_desc.find("channelGroups")
        if ch_grps is not None:
            for group in ch_grps.findall("group"):
                channels = _parse_group_channels(group)
                if channels:
                    anat_grps.append(channels)
                for ch in _group_channel_nodes(group):
                    try:
                        if ch.get("skip") == "1" and ch.text and ch.text.strip():
                            skipped_channels.append(int(ch.text.strip()))
                    except (ValueError, TypeError):
                        continue

    spk_desc = root.find("spikeDetection")
    if spk_desc is not None:
        ch_grps = spk_desc.find("channelGroups")
        if ch_grps is not None:
            for group in ch_grps.findall("group"):
                channels = _parse_group_channels(group)
                if channels:
                    spike_grps.append(channels)

    return anat_grps, spike_grps, sorted(set(skipped_channels))


def load_session_xml_metadata(xml_path: Path) -> SessionXmlMeta:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    anat_grps, spike_grps, skipped = _parse_xml_channel_groups(root)
    return SessionXmlMeta(
        date=_clean_text(root.find("generalInfo/date")),
        experimenters=_clean_text(root.find("generalInfo/experimenters")),
        notes=_clean_text(root.find("generalInfo/notes")) or "",
        description=_clean_text(root.find("generalInfo/description")) or "",
        anatomical_groups_0based=anat_grps,
        spike_groups_0based=spike_grps,
        skipped_channels_0based=skipped,
    )


def _load_xml_groups_for_chanmap(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    anat_grps, spike_grps, skipped_channels = _parse_xml_channel_groups(root)
    spk_channels = [ch for group in spike_grps for ch in group]
    has_spk_groups = len(spike_grps) > 0
    return anat_grps, spk_channels, has_spk_groups, skipped_channels, root


def create_channel_map(
    basepath: Path | str,
    outputDir: Path | str,
    basename: str | None = None,
    electrode_type: str | None = None,
    reject_channels: list[int] | None = None,
    probe_assignments: list[dict] | None = None,
) -> Path | None:
    reject_channels = reject_channels or []

    base_path = Path(basepath)
    if basename is None:
        basename = base_path.name
        if basename.endswith(".xml"):
            basename = os.path.splitext(basename)[0]

    xml_path = base_path / f"{basename}.xml"
    if not xml_path.exists():
        print(f"Error: XML file {xml_path} not found.")
        return None

    anat_grps, spk_channels, has_spk_groups, skipped_channels, root = _load_xml_groups_for_chanmap(xml_path)

    ngroups = len(anat_grps)
    if ngroups == 0:
        print("Warning: No anatomical groups found in XML.")
        return None

    if electrode_type is None:
        electrode_type = "staggered"
        desc_node = root.find("generalInfo/description")
        if desc_node is not None and desc_node.text:
            val = desc_node.text.strip().lower()
            if "neuropixel" in val:
                electrode_type = "NeuroPixel"
            elif "staggered" in val:
                electrode_type = "staggered"
            elif "neurogrid" in val or "grid" in val:
                electrode_type = "neurogrid"
            elif "poly3" in val:
                electrode_type = "poly3"
            elif "poly5" in val:
                electrode_type = "poly5"

    if not probe_assignments:
        probe_assignments = [
            {"type": electrode_type, "groups": list(range(ngroups)), "x_offset": 0}
        ]

    channel_coords = []
    for probe_idx, probe in enumerate(probe_assignments):
        p_type = probe.get("type", electrode_type)
        p_groups = probe.get("groups", [])
        p_x_offset = probe.get("x_offset", 0)

        for local_idx, g_idx in enumerate(p_groups):
            if g_idx >= ngroups:
                continue

            tchannels = anat_grps[g_idx]
            n_ch = len(tchannels)

            x = np.zeros(n_ch)
            y = np.zeros(n_ch)

            shank_id = local_idx + 1

            if p_type == "double_sided":
                pair_idx = local_idx // 2
                is_front = local_idx % 2 == 1
                y = np.arange(1, n_ch + 1) * -20.0
                x[:] = 20.0
                x[::2] = -20.0
                pair_origin = (pair_idx + 1) * 400.0
                intra_pair_offset = 80.0 if is_front else 0.0
                x = x + pair_origin + intra_pair_offset
            elif p_type == "NeuroPixel":
                x_pat = [20, 60, 0, 40]
                x = np.tile(x_pat, (n_ch // 4) + 1)[:n_ch]
                y_base = (np.arange(n_ch) // 2) + 1
                y = y_base * -20.0
                x = x + shank_id * 200
            elif p_type == "staggered":
                x[:] = 20.0
                y = np.arange(1, n_ch + 1) * -20.0
                x[::2] = -20.0
                x = x + shank_id * 200
            elif p_type == "poly3":
                ext = n_ch % 3
                poly = (np.arange(1, n_ch - ext + 1)) % 3
                x[:] = 0
                idx_p1 = np.where(poly == 1)[0] + ext
                idx_p2 = np.where(poly == 2)[0] + ext
                idx_p0 = np.where(poly == 0)[0] + ext
                x[idx_p1] = -18
                x[idx_p2] = 0
                x[idx_p0] = 18
                x[:ext] = 0
                mask_18 = x == 18
                y[mask_18] = np.arange(1, np.sum(mask_18) + 1) * -20
                mask_0 = x == 0
                y[mask_0] = np.arange(1, np.sum(mask_0) + 1) * -20 - 10 + ext * 20
                mask_m18 = x == -18
                y[mask_m18] = np.arange(1, np.sum(mask_m18) + 1) * -20
                x = x + shank_id * 200
            elif p_type == "poly5":
                ext = n_ch % 5
                poly = (np.arange(1, n_ch - ext + 1)) % 5
                x[:] = np.nan
                x[np.where(poly == 1)[0] + ext] = -36
                x[np.where(poly == 2)[0] + ext] = -18
                x[np.where(poly == 3)[0] + ext] = 0
                x[np.where(poly == 4)[0] + ext] = 18
                x[np.where(poly == 0)[0] + ext] = 36
                if ext > 0:
                    x[:ext] = 18 * ((-1.0) ** np.arange(1, ext + 1))
                for val, y_off in [(36, 0), (18, -14), (0, 0), (-18, -14), (-36, 0)]:
                    mask = x == val
                    if np.any(mask):
                        y[mask] = np.arange(1, np.sum(mask) + 1) * -28 + y_off
                x = x + shank_id * 200
            elif p_type == "neurogrid":
                for i in range(n_ch):
                    x[i] = n_ch - (i + 1)
                    y[i] = -(i + 1) * 30
                x = x + shank_id * 30

            x = x + p_x_offset
            k_val = (g_idx // 4) + 1 if p_type == "neurogrid" else g_idx + 1

            for i in range(n_ch):
                channel_coords.append(
                    {
                        "id": tchannels[i],
                        "x": x[i],
                        "y": y[i],
                        "k": k_val,
                        "p": probe_idx + 1,
                    }
                )

    sorted_coords = sorted(channel_coords, key=lambda d: d["id"])
    if not sorted_coords:
        return None

    n_channels = len(sorted_coords)
    xcoords = np.array([d["x"] for d in sorted_coords])
    ycoords = np.array([d["y"] for d in sorted_coords])
    kcoords = np.array([d["k"] for d in sorted_coords])
    pcoords = np.array([d["p"] for d in sorted_coords])
    real_channels = np.array([d["id"] for d in sorted_coords])

    connected = np.ones(n_channels, dtype=bool)
    for rc in reject_channels:
        matches = np.where(real_channels == rc)[0]
        if len(matches) > 0:
            connected[matches] = False

    for sc in skipped_channels:
        matches = np.where(real_channels == sc)[0]
        if len(matches) > 0:
            connected[matches] = False

    if has_spk_groups:
        spk_set = set(spk_channels)
        for i, ch_id in enumerate(real_channels):
            if ch_id not in spk_set:
                connected[i] = False

    chanMap = np.arange(1, n_channels + 1).reshape(1, -1)
    chanMap0ind = real_channels.reshape(1, -1)
    save_dict = {
        "chanMap": chanMap.astype(float),
        "chanMap0ind": chanMap0ind.astype(float),
        "connected": connected.reshape(-1, 1).astype(float),
        "xcoords": xcoords.reshape(-1, 1).astype(float),
        "ycoords": ycoords.reshape(-1, 1).astype(float),
        "kcoords": kcoords.reshape(-1, 1).astype(float),
        "probe_ids": pcoords.reshape(-1, 1).astype(float),
    }

    out_file = Path(outputDir) / "chanMap.mat"
    savemat(out_file, save_dict)
    print(f"Successfully saved chanMap.mat to {out_file}")
    return out_file


def prepare_chanmap(
    *,
    basepath: Path,
    basename: str,
    local_output_dir: Path,
    probe_assignments: list[dict],
    reject_channels: list[int] | None = None,
) -> tuple[Path, list[int]]:
    chanmap_path = create_channel_map(
        basepath=local_output_dir,
        basename=basename,
        outputDir=local_output_dir,
        probe_assignments=probe_assignments,
        reject_channels=reject_channels or [],
    )
    if chanmap_path is None:
        raise RuntimeError("Failed to create chanMap.mat")

    chan = loadmat(chanmap_path)
    connected = np.asarray(chan["connected"]).flatten().astype(int)
    bad_ch_ids = np.where(connected == 0)[0].tolist()
    return Path(chanmap_path), bad_ch_ids


def show_chanmap(
    chanmap_mat_path: Path | str,
    *,
    figsize: tuple[float, float] = (12, 8),
    with_contact_id: bool = False,
    with_device_index: bool = True,
    title: str = "Probe Geometry with Device Indices",
) -> list[int]:
    try:
        import matplotlib.pyplot as plt
        from probeinterface import Probe, ProbeGroup
        from probeinterface.plotting import plot_probegroup
    except Exception as exc:
        raise ImportError(
            "show_chanmap requires matplotlib and probeinterface to be installed"
        ) from exc

    chanmap = loadmat(chanmap_mat_path)
    x = np.asarray(chanmap["xcoords"]).flatten()
    y = np.asarray(chanmap["ycoords"]).flatten()
    shank_ids = np.asarray(chanmap["kcoords"]).flatten()
    probe_ids = np.asarray(chanmap.get("probe_ids", np.ones_like(x))).flatten()
    device_ch_inds = np.asarray(chanmap["chanMap"]).flatten().astype(int) - 1
    connected = np.asarray(chanmap["connected"]).flatten().astype(int)
    bad_ch_ids = np.where(connected == 0)[0].tolist()

    probegroup = ProbeGroup()
    unique_probes = [p for p in np.unique(probe_ids) if p > 0]
    for p_id in unique_probes:
        mask = probe_ids == p_id
        probe = Probe(ndim=2, si_units="um")
        probe.set_contacts(
            positions=np.column_stack((x[mask], y[mask])),
            shapes="circle",
            shape_params={"radius": 5},
            shank_ids=shank_ids[mask],
        )
        probe.set_device_channel_indices(device_ch_inds[mask])
        probegroup.add_probe(probe)

    fig, ax = plt.subplots(figsize=figsize)
    plot_probegroup(
        probegroup,
        with_contact_id=with_contact_id,
        with_device_index=with_device_index,
        ax=ax,
    )
    plt.title(title)
    plt.show()
    return bad_ch_ids


def resolve_basepath_and_basename(basepath: Path) -> tuple[Path, str]:
    basepath = Path(basepath).resolve()
    if not basepath.exists() or not basepath.is_dir():
        raise NotADirectoryError(f"Invalid basepath: {basepath}")
    return basepath, basepath.name


def resolve_local_output_dir(basepath: Path, basename: str, config: PreprocessConfig) -> Path:
    if config.localpath is not None:
        root = Path(config.localpath)
    elif config.output_dir is not None:
        root = Path(config.output_dir)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        root = repo_root / "sorting_temp"
    out = root / basename
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def _first_path(paths) -> Path | None:
    for p in paths:
        return p
    return None


def _direct_child_file_candidates(basepath: Path, *filenames: str) -> list[Path]:
    candidates: list[Path] = []
    for child in sorted(basepath.iterdir()):
        if not child.is_dir():
            continue
        for name in filenames:
            candidate = child / name
            if candidate.exists():
                candidates.append(candidate)
    return candidates


def _extract_openephys_datetime(name: str) -> datetime | None:
    match = _OPENEPHYS_DATETIME_PATTERN.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _is_openephys_datetime_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and _extract_openephys_datetime(path.name) is not None
        and (path / _OPENEPHYS_RECORD_NODE_NAME).is_dir()
    )


def _discover_openephys_recordings(basepath: Path, ignore_folders: list[str]) -> list[Path]:
    recording_roots: list[Path] = []
    for child in sorted(basepath.iterdir()):
        if not child.is_dir():
            continue
        pstr = str(child).lower()
        if any(tok.lower() in pstr for tok in ignore_folders):
            continue
        if not _is_openephys_datetime_dir(child):
            continue
        recording_root = child / _OPENEPHYS_RECORD_NODE_NAME / "experiment1" / "recording1"
        if recording_root.is_dir() and (recording_root / "structure.oebin").exists():
            recording_roots.append(recording_root)
    return recording_roots


def _find_openephys_datetime_ancestor(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if _extract_openephys_datetime(candidate.name) is not None:
            return candidate
    return None


def _openephys_subsession_name(recording_root: Path) -> str:
    dt_dir = _find_openephys_datetime_ancestor(recording_root)
    if dt_dir is not None:
        return dt_dir.name
    return recording_root.name


def _load_openephys_structure(recording_root: Path) -> dict:
    structure_path = recording_root / "structure.oebin"
    with open(structure_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_openephys_stream_info(recording_root: Path) -> tuple[Path, str, Path, int, float]:
    structure = _load_openephys_structure(recording_root)
    continuous_entries = structure.get("continuous", [])
    if not continuous_entries:
        raise FileNotFoundError(f"No continuous streams found in {recording_root / 'structure.oebin'}")

    entry = continuous_entries[0]
    folder_name = str(entry["folder_name"]).rstrip("/\\")
    continuous_dat = recording_root / "continuous" / folder_name / "continuous.dat"
    if not continuous_dat.exists():
        raise FileNotFoundError(f"Missing Open Ephys continuous.dat: {continuous_dat}")

    recorded_processor = str(entry.get("recorded_processor", "Record Node")).strip()
    recorded_processor_id = entry.get("recorded_processor_id")
    if recorded_processor_id is not None:
        stream_name = f"{recorded_processor} {recorded_processor_id}#{folder_name}"
    else:
        stream_name = folder_name

    ttl_path = recording_root / "events" / folder_name / "TTL"
    num_channels = int(entry["num_channels"])
    sampling_frequency = float(entry["sample_rate"])
    return continuous_dat, stream_name, ttl_path, num_channels, sampling_frequency


def ensure_xml(basepath: Path, local_output_dir: Path, basename: str) -> Path:
    target = local_output_dir / f"{basename}.xml"
    base_xml = basepath / f"{basename}.xml"
    if base_xml.exists():
        copy2(base_xml, target)
        return target

    parent_xml = sorted(basepath.parent.glob("*.xml"))
    if len(parent_xml) == 1:
        copy2(parent_xml[0], target)
        return target
    if len(parent_xml) > 1:
        names = ", ".join(p.name for p in parent_xml)
        raise FileNotFoundError(
            f"Multiple xml files found in parent directory ({basepath.parent}): {names}. "
            f"Place {basename}.xml in {basepath} or keep exactly one parent xml."
        )

    if target.exists():
        return target

    raise FileNotFoundError(
        f"No xml file found. Expected {base_xml} or exactly one xml in parent directory ({basepath.parent})."
    )


def ensure_rhd(basepath: Path, local_output_dir: Path, basename: str) -> Path | None:
    target = local_output_dir / f"{basename}.rhd"
    preferred = [basepath / f"{basename}.rhd", basepath / "info.rhd"]

    for src in preferred:
        if src is not None and src.exists():
            copy2(src, target)
            return target

    child_matches = _direct_child_file_candidates(basepath, "info.rhd", f"{basename}.rhd")
    if len(child_matches) == 1:
        copy2(child_matches[0], target)
        return target
    if len(child_matches) > 1:
        names = ", ".join(str(p.relative_to(basepath)) for p in child_matches)
        raise FileNotFoundError(
            f"Multiple rhd files found in direct child folders under {basepath}: {names}. "
            f"Keep exactly one match or place {basename}.rhd/info.rhd directly in {basepath}."
        )

    local_rhd = sorted(basepath.glob("*.rhd"))
    if len(local_rhd) == 1:
        copy2(local_rhd[0], target)
        return target
    if len(local_rhd) > 1:
        names = ", ".join(p.name for p in local_rhd)
        raise FileNotFoundError(
            f"Multiple rhd files found in {basepath}: {names}. "
            f"Keep exactly one match or place {basename}.rhd/info.rhd directly in {basepath}."
        )

    if target.exists():
        return target

    return None


def load_xml_metadata(xml_path: Path) -> XmlMeta:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    session_xml_meta = load_session_xml_metadata(xml_path)

    sr_tag = root.find(".//sampleRate")
    if sr_tag is None:
        sr_tag = root.find(".//samplingRate")
    if sr_tag is None or sr_tag.text is None:
        raise ValueError(f"Could not parse sampling rate from {xml_path}")
    sr = float(sr_tag.text)

    sr_lfp: float | None = None
    lfp_tag = root.find(".//fieldPotentials/lfpSamplingRate")
    if lfp_tag is None:
        lfp_tag = root.find(".//lfpSampleRate")
    if lfp_tag is not None and lfp_tag.text:
        try:
            sr_lfp = float(lfp_tag.text)
        except ValueError:
            sr_lfp = None

    n_channels = None
    n_tag = root.find(".//acquisitionSystem/nChannels")
    if n_tag is not None and n_tag.text:
        n_channels = int(n_tag.text)
    if n_channels is None:
        acq = root.find(".//acquisitionSystem")
        if acq is not None:
            cands = [acq.find("numChannels"), acq.find("nChannels")]
            for cand in cands:
                if cand is not None and cand.text:
                    n_channels = int(cand.text)
                    break
    if n_channels is None:
        raise ValueError(f"Could not parse channel count from {xml_path}")

    return XmlMeta(
        sr=sr,
        sr_lfp=sr_lfp,
        n_channels=n_channels,
        skipped_channels_0based=session_xml_meta.skipped_channels_0based,
    )


def discover_subsessions(
    basepath: Path,
    sort_files: bool,
    alt_sort: list[int] | None,
    ignore_folders: list[str] | None,
) -> list[Path]:
    if sort_files and alt_sort:
        raise ValueError("sort_files=True cannot be used with alt_sort")

    ignore_folders = ignore_folders or []

    paths = _discover_openephys_recordings(basepath, ignore_folders)
    if not paths:
        paths = []
        for child in sorted(basepath.iterdir()):
            if not child.is_dir():
                continue
            pstr = str(child).lower()
            if any(tok.lower() in pstr for tok in ignore_folders):
                continue
            amp = child / "amplifier.dat"
            cont = child / "continuous.dat"
            if amp.exists():
                paths.append(amp)
            elif cont.exists():
                paths.append(cont)

    if not paths:
        return []

    if sort_files:
        paths = sorted(paths, key=_subsession_sort_key)
    elif alt_sort:
        idx = _normalize_alt_sort_indices(alt_sort, len(paths))
        paths = [paths[i] for i in idx]

    return paths


def _normalize_alt_sort_indices(alt_sort: list[int], n: int) -> list[int]:
    if not alt_sort:
        return list(range(n))
    if min(alt_sort) >= 1 and max(alt_sort) <= n:
        idx = [i - 1 for i in alt_sort]
    else:
        idx = alt_sort
    if sorted(idx) != list(range(n)):
        raise ValueError(f"Invalid alt_sort for {n} subsessions: {alt_sort}")
    return idx


def _subsession_sort_key(path: Path) -> tuple[int, str]:
    name = path.parent.name
    oe_dt_dir = _find_openephys_datetime_ancestor(path)
    if oe_dt_dir is not None:
        dt = _extract_openephys_datetime(oe_dt_dir.name)
        if dt is not None:
            return int(dt.strftime("%Y%m%d%H%M%S")), str(path)

    intan_match = re.search(r"(\d{6}_\d{6})", name)
    if intan_match:
        dt = datetime.strptime(intan_match.group(1), "%y%m%d_%H%M%S")
        return int(dt.strftime("%Y%m%d%H%M%S")), str(path)

    oe_dt = _extract_openephys_datetime(name)
    if oe_dt is not None:
        return int(oe_dt.strftime("%Y%m%d%H%M%S")), str(path)

    return 0, str(path)


def _infer_channels_from_file(path: Path, sample_count: int, bytes_per_sample: int = 2) -> int:
    if sample_count <= 0 or not path.exists():
        return 0
    return int(path.stat().st_size // (sample_count * bytes_per_sample))


def build_acquisition_catalog(
    amplifier_paths: list[Path],
    n_amplifier_channels: int,
    dtype: str,
    intan_header: IntanRhdHeader | None = None,
) -> AcquisitionCatalog:
    if amplifier_paths and amplifier_paths[0].is_dir() and (amplifier_paths[0] / "structure.oebin").exists():
        itemsize = np.dtype(dtype).itemsize
        recording_paths = [Path(p) for p in amplifier_paths]
        continuous_paths: list[Path] = []
        recording_stream_names: list[str | None] = []
        ttl_event_paths: list[Path] = []
        sample_counts: list[int] = []
        sampling_frequency: float | None = None
        amplifier_channels: int | None = None

        for recording_root in recording_paths:
            continuous_path, stream_name, ttl_path, n_channels, sr = _resolve_openephys_stream_info(recording_root)
            if sampling_frequency is None:
                sampling_frequency = sr
            elif not np.isclose(sampling_frequency, sr):
                raise ValueError(
                    "Open Ephys recordings with mismatched sampling frequencies are unsupported: "
                    f"{sampling_frequency} vs {sr} ({recording_root})"
                )
            if amplifier_channels is None:
                amplifier_channels = n_channels
            elif amplifier_channels != n_channels:
                raise ValueError(
                    "Open Ephys recordings with mismatched channel counts are unsupported: "
                    f"{amplifier_channels} vs {n_channels} ({recording_root})"
                )

            continuous_paths.append(continuous_path)
            recording_stream_names.append(stream_name)
            ttl_event_paths.append(ttl_path)
            sample_counts.append(int(continuous_path.stat().st_size // (n_channels * itemsize)))

        has_ttl = any(path.exists() for path in ttl_event_paths)
        dig_ch = 16 if has_ttl else 0
        dig_word_ch = 1 if has_ttl else 0
        dig_native_orders = list(range(16)) if has_ttl else []

        return AcquisitionCatalog(
            source_type="openephys",
            subsession_names=[_openephys_subsession_name(p) for p in recording_paths],
            recording_paths=recording_paths,
            recording_stream_names=recording_stream_names,
            ttl_event_paths=ttl_event_paths,
            amplifier_paths=continuous_paths,
            analogin_paths=[],
            digitalin_paths=[],
            auxiliary_paths=[],
            supply_paths=[],
            time_paths=[],
            sample_counts=sample_counts,
            sampling_frequency=sampling_frequency,
            amplifier_channels=int(amplifier_channels or n_amplifier_channels),
            auxiliary_input_channels=0,
            supply_voltage_channels=0,
            board_adc_channels=0,
            board_digital_input_channels=dig_ch,
            board_digital_word_channels=dig_word_ch,
            board_digital_output_channels=0,
            temperature_sensor_channels=0,
            board_adc_native_orders=[],
            board_digital_input_native_orders=dig_native_orders,
        )

    itemsize = np.dtype(dtype).itemsize
    sample_counts = [int(p.stat().st_size // (n_amplifier_channels * itemsize)) for p in amplifier_paths]

    analogin_paths: list[Path] = []
    digitalin_paths: list[Path] = []
    auxiliary_paths: list[Path] = []
    supply_paths: list[Path] = []
    time_paths: list[Path] = []

    for p in amplifier_paths:
        d = p.parent
        analog = d / "analogin.dat"
        digital = d / "digitalin.dat"
        aux = d / "auxiliary.dat"
        supply = d / "supply.dat"
        tdat = d / "time.dat"

        if analog.exists():
            analogin_paths.append(analog)
        if digital.exists():
            digitalin_paths.append(digital)
        if aux.exists():
            auxiliary_paths.append(aux)
        if supply.exists():
            supply_paths.append(supply)
        if tdat.exists():
            time_paths.append(tdat)

    aux_ch = _infer_channels_from_file(auxiliary_paths[0], sample_counts[0]) if auxiliary_paths else 0
    supply_ch = _infer_channels_from_file(supply_paths[0], sample_counts[0]) if supply_paths else 0
    adc_ch = _infer_channels_from_file(analogin_paths[0], sample_counts[0]) if analogin_paths else 0
    adc_native_orders: list[int] = []

    if intan_header is not None:
        if intan_header.num_aux_input_channels > 0:
            aux_ch = int(intan_header.num_aux_input_channels)
        if intan_header.num_supply_voltage_channels > 0:
            supply_ch = int(intan_header.num_supply_voltage_channels)
        if intan_header.num_board_adc_channels > 0:
            adc_ch = int(intan_header.num_board_adc_channels)
        if intan_header.board_adc_native_orders:
            adc_native_orders = [int(ch) for ch in intan_header.board_adc_native_orders]
    if not adc_native_orders and adc_ch > 0:
        adc_native_orders = list(range(int(adc_ch)))

    dig_ch = 0
    dig_word_ch = 0
    dig_native_orders: list[int] = []
    if digitalin_paths:
        raw_words = _infer_channels_from_file(digitalin_paths[0], sample_counts[0])
        dig_word_ch = raw_words if raw_words > 0 else 1
        dig_ch = 16 if raw_words in (0, 1, 16) else raw_words
    if intan_header is not None:
        if intan_header.num_board_dig_in_channels > 0:
            dig_ch = int(intan_header.num_board_dig_in_channels)
            if dig_word_ch <= 0:
                dig_word_ch = 1
        if intan_header.board_dig_in_native_orders:
            dig_native_orders = [int(ch) for ch in intan_header.board_dig_in_native_orders]
        board_dig_out_ch = int(intan_header.num_board_dig_out_channels)
        temp_sensor_ch = int(intan_header.num_temp_sensor_channels)
    else:
        board_dig_out_ch = 0
        temp_sensor_ch = 0
    if not dig_native_orders and dig_ch > 0:
        dig_native_orders = list(range(int(dig_ch)))

    return AcquisitionCatalog(
        source_type="intan",
        subsession_names=[p.parent.name for p in amplifier_paths],
        recording_paths=[p.parent for p in amplifier_paths],
        recording_stream_names=[None for _ in amplifier_paths],
        ttl_event_paths=[],
        amplifier_paths=amplifier_paths,
        analogin_paths=analogin_paths,
        digitalin_paths=digitalin_paths,
        auxiliary_paths=auxiliary_paths,
        supply_paths=supply_paths,
        time_paths=time_paths,
        sample_counts=sample_counts,
        sampling_frequency=None,
        amplifier_channels=n_amplifier_channels,
        auxiliary_input_channels=aux_ch,
        supply_voltage_channels=supply_ch,
        board_adc_channels=adc_ch,
        board_digital_input_channels=dig_ch,
        board_digital_word_channels=dig_word_ch,
        board_digital_output_channels=board_dig_out_ch,
        temperature_sensor_channels=temp_sensor_ch,
        board_adc_native_orders=adc_native_orders,
        board_digital_input_native_orders=dig_native_orders,
    )


def print_catalog_summary(catalog: AcquisitionCatalog) -> None:
    print(f"Found {catalog.amplifier_channels} amplifier channels.")
    print(f"Found {catalog.auxiliary_input_channels} auxiliary input channels.")
    print(f"Found {catalog.supply_voltage_channels} supply voltage channels.")
    print(f"Found {catalog.board_adc_channels} board ADC channels.")
    print(f"Found {catalog.board_digital_input_channels} board digital input channels.")
    print(f"Found {catalog.board_digital_output_channels} board digital output channels.")
    print(f"Found {catalog.temperature_sensor_channels} temperature sensor channels.")


def save_params_and_manifest(
    config: PreprocessConfig,
    result: PreprocessResult,
    output_dir: Path,
    script_path: Path | None = None,
) -> None:
    def _sanitize_for_mat(value):
        if isinstance(value, dict):
            return {k: _sanitize_for_mat(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize_for_mat(v) for v in value]
        if isinstance(value, tuple):
            return [_sanitize_for_mat(v) for v in value]
        if value is None:
            return ""
        return value

    cfg = asdict(config)
    cfg["basepath"] = str(config.basepath)
    cfg["localpath"] = str(config.localpath) if config.localpath else None
    cfg["output_dir"] = str(config.output_dir) if config.output_dir else None
    cfg["chanmap_mat_path"] = str(config.chanmap_mat_path) if config.chanmap_mat_path else None
    cfg["sorter_path"] = str(config.sorter_path) if config.sorter_path else None
    cfg["sorter_config_path"] = str(config.sorter_config_path) if config.sorter_config_path else None
    cfg["matlab_path"] = str(config.matlab_path) if config.matlab_path else None

    if config.save_params_json:
        with open(output_dir / "preprocessSession_params.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    manifest = {
        "basepath": str(result.basepath),
        "basename": result.basename,
        "local_output_dir": str(result.local_output_dir),
        "dat_path": str(result.dat_path) if result.dat_path else None,
        "lfp_path": str(result.lfp_path) if result.lfp_path else None,
        "session_mat_path": str(result.session_mat_path),
        "mergepoints_mat_path": str(result.mergepoints_mat_path),
        "analog_event_paths": [str(p) for p in result.analog_event_paths],
        "digital_event_paths": [str(p) for p in result.digital_event_paths],
        "intermediate_dat_paths": {k: str(v) for k, v in result.intermediate_dat_paths.items()},
        "n_channels": result.n_channels,
        "sr": result.sr,
        "sr_lfp": result.sr_lfp,
        "bad_channels_0based": result.bad_channels_0based,
        "bad_channels_1based": result.bad_channels_1based,
        "subsession_paths": [str(p) for p in result.subsession_paths],
        "subsession_sample_counts": result.subsession_sample_counts,
        "sorter": result.sorter,
        "sorter_output_dir": str(result.sorter_output_dir) if result.sorter_output_dir else None,
        "state_score_paths": [str(p) for p in result.state_score_paths],
        "state_score_figure_paths": [str(p) for p in result.state_score_figure_paths],
    }

    if config.save_manifest_json:
        with open(output_dir / "preprocessSession_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if config.save_log_mat:
        cfg_for_mat = _sanitize_for_mat(cfg)
        savemat(output_dir / "preprocessSession_params.mat", {"results": cfg_for_mat}, do_compression=True)
        if script_path is not None and Path(script_path).exists():
            copy2(script_path, output_dir / "preprocessSession.log")


def set_tree_world_rw(root: Path) -> None:
    root = Path(root)
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        try:
            if path.is_symlink():
                continue
            mode = path.stat().st_mode
            if path.is_dir():
                path.chmod(mode | 0o777)
            elif path.is_file():
                path.chmod(mode | 0o666)
        except Exception as exc:
            print(f"Warning: failed to update permissions for {path}: {exc}")


def set_paths_world_rw(paths: list[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        current = Path(path)
        if current.name == "@eaDir":
            continue
        if current in seen or not current.exists():
            continue
        seen.add(current)
        try:
            if current.is_symlink():
                continue
            mode = current.stat().st_mode
            if current.is_dir():
                current.chmod(mode | 0o777)
            elif current.is_file():
                current.chmod(mode | 0o666)
        except Exception as exc:
            print(f"Warning: failed to update permissions for {current}: {exc}")


def copy_results_to_basepath(
    *,
    local_output_dir: Path,
    basepath: Path,
    delete_local: bool = False,
) -> Path:
    src = Path(local_output_dir).resolve()
    dst = Path(basepath).resolve()

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Local output directory does not exist: {src}")
    if not dst.exists() or not dst.is_dir():
        raise NotADirectoryError(f"Basepath does not exist or is not a directory: {dst}")
    if src == dst:
        raise ValueError(f"Source and destination are identical: {src}")

    changed_paths: list[Path] = []

    def _copy_path_contents_no_metadata(source_path: Path, target_path: Path) -> bool:
        copied = False
        if source_path.is_dir():
            created = not target_path.exists()
            target_path.mkdir(parents=True, exist_ok=True)
            if created:
                changed_paths.append(target_path)
            for nested in source_path.iterdir():
                copied = _copy_path_contents_no_metadata(nested, target_path / nested.name) or copied
            return copied
        if source_path.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            changed_paths.append(target_path)
            return True
        return False

    copied_any = False
    for child in src.iterdir():
        copied_any = _copy_path_contents_no_metadata(child, dst / child.name) or copied_any

    if copied_any:
        set_paths_world_rw(changed_paths)

    if delete_local:
        shutil.rmtree(src)

    return dst


def convert_dual_side_map(
    chan_map_file: str | Path,
    x_shift: float = 6.0,
    pairs_to_merge: list[tuple[int, int]] | None = None,
    custom_shank_positions: dict[int, float] | None = None,
) -> None:
    """Merge paired front/back shanks in chanMap and apply optional x-offset overrides."""
    p = Path(chan_map_file)
    if not p.exists():
        print(f"File not found: {p}")
        return

    data = loadmat(p)
    x = np.asarray(data["xcoords"]).flatten()
    k = np.asarray(data["kcoords"]).flatten()
    x_shape = data["xcoords"].shape
    k_shape = data["kcoords"].shape

    unique_shanks = np.unique(k)
    unique_shanks.sort()
    if pairs_to_merge is None:
        n_pairs = len(unique_shanks) // 2
        pairs_to_merge = [(int(unique_shanks[2 * i]), int(unique_shanks[2 * i + 1])) for i in range(n_pairs)]
        print(f"Auto-detected {len(pairs_to_merge)} pairs to merge.")
    else:
        print(f"Using provided list of {len(pairs_to_merge)} pairs to merge.")

    for s_back, s_front in pairs_to_merge:
        idx_back = k == s_back
        idx_front = k == s_front
        if not np.any(idx_back) or not np.any(idx_front):
            print(f"Warning: Missing channels for pair ({s_back}, {s_front}).")
            continue

        k[idx_front] = s_back
        mean_x_back = float(np.mean(x[idx_back]))
        mean_x_front = float(np.mean(x[idx_front]))
        current_offset = mean_x_front - mean_x_back
        x[idx_front] = x[idx_front] - current_offset + x_shift
        print(f"Merged Shank {s_front} into {s_back}. Shifted X by {-current_offset + x_shift:.2f} um.")

    if custom_shank_positions:
        print(f"Applying custom positions to {len(custom_shank_positions)} shanks.")
        for s_id, target_x in custom_shank_positions.items():
            idx_group = k == s_id
            if np.any(idx_group):
                start_mean = float(np.mean(x[idx_group]))
                x[idx_group] += target_x - start_mean
                print(f"  Shank {s_id}: Moved to {target_x:.1f} (shift {target_x - start_mean:.1f})")

    data["xcoords"] = x.reshape(x_shape)
    data["kcoords"] = k.reshape(k_shape)
    savemat(p, data)
    print(f"Updated {p} with dual-side conversion.")


def load_rez(rez_path: str | Path):
    """
    Load MATLAB v7.3 rez.mat via h5py and convert nested groups to dicts.

    For multi-dimensional arrays, reverse axes to recover MATLAB ordering.
    """
    import h5py

    def h5_to_dict(obj):
        if isinstance(obj, h5py.Group):
            return {k: h5_to_dict(obj[k]) for k in obj.keys()}
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            if isinstance(data, np.ndarray):
                if data.ndim > 1:
                    data = np.ascontiguousarray(np.transpose(data, axes=range(data.ndim - 1, -1, -1)))
                data = np.squeeze(data)
            return data
        return obj

    with h5py.File(rez_path, "r") as f:
        if "rez" in f:
            return h5_to_dict(f["rez"])
        return h5_to_dict(f)


def rezToPhy(rez: dict, save_path: str | Path) -> None:
    """Extract Kilosort rez fields and write Phy-compatible .npy files."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        "amplitudes.npy",
        "channel_map.npy",
        "channel_positions.npy",
        "pc_features.npy",
        "pc_feature_ind.npy",
        "similar_templates.npy",
        "spike_clusters.npy",
        "spike_templates.npy",
        "spike_times.npy",
        "templates.npy",
        "templates_ind.npy",
        "template_features.npy",
        "template_feature_ind.npy",
        "whitening_mat.npy",
        "whitening_mat_inv.npy",
    ]
    for filename in outputs:
        fp = save_dir / filename
        if fp.exists():
            fp.unlink()
    phy_dir = save_dir / ".phy"
    if phy_dir.exists():
        shutil.rmtree(phy_dir)

    st3 = np.asarray(rez["st3"])
    spike_times = st3[:, 0].astype(np.uint64)
    spike_templates = (st3[:, 1] - 1).astype(np.uint32)
    spike_clusters = (st3[:, 4] - 1).astype(np.int32) if st3.shape[1] > 4 else spike_templates.astype(np.int32)
    amplitudes = st3[:, 2]

    ops = rez["ops"]
    chan_map = np.atleast_1d(ops["chanMap"]).flatten()
    chan_map_0ind = (chan_map - 1).astype(np.int32)

    connected = np.atleast_1d(rez["connected"]).flatten().astype(bool)
    xcoords = np.atleast_1d(rez["xcoords"]).flatten()
    ycoords = np.atleast_1d(rez["ycoords"]).flatten()

    U = np.asarray(rez["U"])
    W = np.asarray(rez["W"])
    templates = np.einsum("cfr, tfr -> ftc", U, W)
    n_templates = templates.shape[0]
    n_chan = U.shape[0]
    templates_inds = np.tile(np.arange(n_chan), (n_templates, 1)).astype(np.int32)

    pc_features = np.asarray(rez["cProjPC"])
    pc_feature_inds = (np.atleast_2d(rez["iNeighPC"]) - 1).astype(np.int32)
    template_features = np.asarray(rez["cProj"])
    template_feature_inds = (np.atleast_2d(rez["iNeigh"]) - 1).astype(np.int32)

    if pc_feature_inds.shape[0] != n_templates:
        pc_feature_inds = pc_feature_inds.T
    if template_feature_inds.shape[0] != n_templates:
        template_feature_inds = template_feature_inds.T

    np.save(save_dir / "spike_times.npy", np.ascontiguousarray(spike_times))
    np.save(save_dir / "spike_templates.npy", np.ascontiguousarray(spike_templates))
    np.save(save_dir / "spike_clusters.npy", np.ascontiguousarray(spike_clusters))
    np.save(save_dir / "amplitudes.npy", np.ascontiguousarray(amplitudes))
    np.save(save_dir / "templates.npy", np.ascontiguousarray(templates.astype(np.float32)))
    np.save(save_dir / "templates_ind.npy", np.ascontiguousarray(templates_inds))

    np.save(save_dir / "channel_map.npy", np.ascontiguousarray(chan_map_0ind[connected]))
    channel_positions = np.column_stack((xcoords[connected], ycoords[connected]))
    np.save(save_dir / "channel_positions.npy", np.ascontiguousarray(channel_positions))

    np.save(save_dir / "template_features.npy", np.ascontiguousarray(template_features.astype(np.float32)))
    np.save(save_dir / "template_feature_ind.npy", np.ascontiguousarray(template_feature_inds))
    np.save(save_dir / "pc_features.npy", np.ascontiguousarray(pc_features.astype(np.float32)))
    np.save(save_dir / "pc_feature_ind.npy", np.ascontiguousarray(pc_feature_inds))

    whitening_matrix = np.asarray(rez["Wrot"]) / 200
    whitening_matrix_inv = np.linalg.pinv(whitening_matrix)
    np.save(save_dir / "whitening_mat.npy", np.ascontiguousarray(whitening_matrix.astype(np.float32)))
    np.save(save_dir / "whitening_mat_inv.npy", np.ascontiguousarray(whitening_matrix_inv.astype(np.float32)))

    if "simScore" in rez:
        np.save(save_dir / "similar_templates.npy", np.ascontiguousarray(np.asarray(rez["simScore"]).astype(np.float32)))

    params_path = save_dir / "params.py"
    fb_val = ops.get("fbinary", "recording.dat")
    if isinstance(fb_val, np.ndarray) and fb_val.dtype.kind in "ui":
        fbinary = "".join([chr(int(c)) for c in fb_val.flatten()])
    else:
        fbinary = str(fb_val)

    dat_path = "../" + os.path.basename(fbinary)
    n_chan_tot = int(np.atleast_1d(ops["NchanTOT"]).flatten()[0])
    fs = float(np.atleast_1d(ops["fs"]).flatten()[0])
    with open(params_path, "w", encoding="utf-8") as f:
        f.write(f"dat_path = r'{dat_path}'\n")
        f.write(f"n_channels_dat = {n_chan_tot}\n")
        f.write("dtype = 'int16'\n")
        f.write("offset = 0\n")
        f.write(f"sample_rate = {fs}\n")
        f.write("hp_filtered = False\n")

    print(f"Done! Phy files saved to {save_dir}")
