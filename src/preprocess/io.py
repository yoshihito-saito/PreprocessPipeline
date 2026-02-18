from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
from shutil import copy2
import json
import re
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat, savemat

from .metafile import AcquisitionCatalog, PreprocessConfig, PreprocessResult, XmlMeta


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


def _load_xml_groups_for_chanmap(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    anat_grps: list[list[int]] = []
    skipped_channels: list[int] = []
    anat_desc = root.find("anatomicalDescription")
    if anat_desc is not None:
        ch_grps = anat_desc.find("channelGroups")
        if ch_grps is not None:
            for group in ch_grps.findall("group"):
                channels: list[int] = []
                tags = group.findall("n")
                if not tags:
                    tags = group.findall("channel")
                for ch in tags:
                    try:
                        if ch.text and ch.text.strip():
                            val = int(ch.text)
                            channels.append(val)
                            if ch.get("skip") == "1":
                                skipped_channels.append(val)
                    except (ValueError, TypeError):
                        continue
                if channels:
                    anat_grps.append(channels)

    spk_channels: list[int] = []
    spk_desc = root.find("spikeDetection")
    has_spk_groups = False
    if spk_desc is not None:
        ch_grps = spk_desc.find("channelGroups")
        if ch_grps is not None:
            groups = ch_grps.findall("group")
            if groups:
                has_spk_groups = True
                for group in groups:
                    channels_container = group.find("channels")
                    tags = []
                    if channels_container is not None:
                        tags = channels_container.findall("n")
                        if not tags:
                            tags = channels_container.findall("channel")
                    else:
                        tags = group.findall("n")
                        if not tags:
                            tags = group.findall("channel")
                    for ch in tags:
                        try:
                            if ch.text and ch.text.strip():
                                spk_channels.append(int(ch.text))
                        except (ValueError, TypeError):
                            continue

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


def ensure_xml(basepath: Path, local_output_dir: Path, basename: str) -> Path:
    target = local_output_dir / f"{basename}.xml"
    if target.exists():
        return target

    parent_xml = sorted(basepath.parent.glob("*.xml"))
    if parent_xml:
        copy2(parent_xml[0], target)
        return target

    local_xml = sorted(basepath.glob("*.xml"))
    if local_xml:
        copy2(local_xml[0], target)
        return target

    nested_xml = sorted(basepath.rglob("*.xml"))
    if nested_xml:
        copy2(nested_xml[0], target)
        return target

    raise FileNotFoundError(f"No xml file found in parent/basepath for {basepath}")


def load_xml_metadata(xml_path: Path) -> XmlMeta:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sr_tag = root.find(".//sampleRate")
    if sr_tag is None:
        sr_tag = root.find(".//samplingRate")
    if sr_tag is None or sr_tag.text is None:
        raise ValueError(f"Could not parse sampling rate from {xml_path}")
    sr = float(sr_tag.text)

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

    skipped = []
    anat_desc = root.find("anatomicalDescription")
    if anat_desc is not None:
        ch_grps = anat_desc.find("channelGroups")
        if ch_grps is not None:
            for group in ch_grps.findall("group"):
                for ch in group.findall("n") + group.findall("channel"):
                    if ch.get("skip") == "1" and ch.text and ch.text.strip():
                        skipped.append(int(ch.text.strip()))

    return XmlMeta(sr=sr, n_channels=n_channels, skipped_channels_0based=sorted(set(skipped)))


def discover_subsessions(
    basepath: Path,
    sort_files: bool,
    alt_sort: list[int] | None,
    ignore_folders: list[str] | None,
) -> list[Path]:
    if sort_files and alt_sort:
        raise ValueError("sort_files=True cannot be used with alt_sort")

    ignore_folders = ignore_folders or []

    amp = list(basepath.rglob("amplifier.dat"))
    cont = list(basepath.rglob("continuous.dat"))
    paths = sorted(set(amp + cont))

    filtered = []
    for p in paths:
        pstr = str(p).lower()
        if any(tok.lower() in pstr for tok in ignore_folders):
            continue
        filtered.append(p)

    if not filtered:
        return []

    if sort_files:
        filtered = sorted(filtered, key=_subsession_sort_key)
    elif alt_sort:
        idx = _normalize_alt_sort_indices(alt_sort, len(filtered))
        filtered = [filtered[i] for i in idx]

    return filtered


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

    intan_match = re.search(r"(\d{6}_\d{6})", name)
    if intan_match:
        dt = datetime.strptime(intan_match.group(1), "%y%m%d_%H%M%S")
        return int(dt.strftime("%Y%m%d%H%M%S")), str(path)

    oe_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
    if oe_match:
        dt = datetime.strptime(oe_match.group(1), "%Y-%m-%d_%H-%M-%S")
        return int(dt.strftime("%Y%m%d%H%M%S")), str(path)

    return 0, str(path)


def _infer_channels_from_file(path: Path, sample_count: int, bytes_per_sample: int = 2) -> int:
    if sample_count <= 0 or not path.exists():
        return 0
    return int(path.stat().st_size // (sample_count * bytes_per_sample))


def build_acquisition_catalog(
    amplifier_paths: list[Path],
    n_amplifier_channels: int,
    dtype: str,
) -> AcquisitionCatalog:
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

    dig_ch = 0
    if digitalin_paths:
        raw_words = _infer_channels_from_file(digitalin_paths[0], sample_counts[0])
        dig_ch = 16 if raw_words in (0, 1, 16) else raw_words

    return AcquisitionCatalog(
        subsession_names=[p.parent.name for p in amplifier_paths],
        amplifier_paths=amplifier_paths,
        analogin_paths=analogin_paths,
        digitalin_paths=digitalin_paths,
        auxiliary_paths=auxiliary_paths,
        supply_paths=supply_paths,
        time_paths=time_paths,
        sample_counts=sample_counts,
        amplifier_channels=n_amplifier_channels,
        auxiliary_input_channels=aux_ch,
        supply_voltage_channels=supply_ch,
        board_adc_channels=adc_ch,
        board_digital_input_channels=dig_ch,
        board_digital_output_channels=0,
        temperature_sensor_channels=0,
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
    }

    if config.save_manifest_json:
        with open(output_dir / "preprocessSession_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if config.save_log_mat:
        cfg_for_mat = _sanitize_for_mat(cfg)
        savemat(output_dir / "preprocessSession_params.mat", {"results": cfg_for_mat}, do_compression=True)
        if script_path is not None and Path(script_path).exists():
            copy2(script_path, output_dir / "preprocessSession.log")
