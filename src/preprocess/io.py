from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import os
import shutil
import uuid
import filecmp
from pathlib import Path
from shutil import copy2
import json
import re
import warnings
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatWriteError

from .intan_rhd import IntanRhdHeader, read_intan_rhd_header
from .metafile import (
    AcquisitionCatalog,
    PreprocessConfig,
    PreprocessResult,
    SessionXmlMeta,
    XmlMeta,
)
from .paths import find_project_root


_OPENEPHYS_DATETIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
_OPENEPHYS_RECORD_NODE_NAME = "Record Node 101"
_DAY_PREFIX_PATTERN = re.compile(r"^(?:day|d)(\d+)", re.IGNORECASE)
_OPENEPHYS_EPHYS_CHANNEL_PATTERN = re.compile(r"^CH\d+$", re.IGNORECASE)
_OPENEPHYS_ADC_CHANNEL_PATTERN = re.compile(r"^ADC\d+$", re.IGNORECASE)
A5X12_16_BUZ_LIN_PROBE_TYPE = "A5x12-16-Buz-Lin-5mm-100-200-160-177"


def atomic_write_path(
    target: Path,
    writer,
    *,
    validator=None,
) -> Path:
    """Publish a file only after its same-directory temporary output validates.

    The canonical target is never opened for writing.  A writer receives the
    temporary path and may raise; in that case an earlier canonical output is
    deliberately left untouched.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial-{uuid.uuid4().hex}{target.suffix}")
    try:
        writer(temporary)
        if validator is not None:
            validation_result = validator(temporary)
            if validation_result is False:
                raise ValueError(f"New output failed validation: {temporary}")
        elif not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError(f"New output is missing or empty: {temporary}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


_MAT_V5_SAFE_PAYLOAD_BYTES = (2**31) - (64 * 1024**2)


def _mat_payload_nbytes(value: object) -> int:
    if isinstance(value, dict):
        return sum(_mat_payload_nbytes(item) for item in value.values())
    if isinstance(value, str):
        return len(value.encode("utf-16-le"))
    try:
        return int(np.asarray(value).nbytes)
    except (TypeError, ValueError):
        return 0


def _matlab_class_for_dtype(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return "logical"
    if dtype.kind == "f":
        return "single" if dtype.itemsize == 4 else "double"
    if dtype.kind in {"i", "u"}:
        return dtype.name
    raise TypeError(f"Unsupported MATLAB v7.3 numeric dtype: {dtype}")


def _write_mat73_fields_attribute(group, field_names: list[str]) -> None:
    import h5py

    field_dtype = h5py.vlen_dtype(np.dtype("S1"))
    fields = np.empty((len(field_names),), dtype=object)
    for index, name in enumerate(field_names):
        fields[index] = np.frombuffer(name.encode("ascii"), dtype="S1")
    group.attrs.create("MATLAB_fields", fields, dtype=field_dtype)


def _write_mat73_value(parent, name: str, value: object) -> None:
    if isinstance(value, dict):
        group = parent.create_group(name)
        group.attrs["MATLAB_class"] = np.bytes_("struct")
        group.attrs["H5PATH"] = np.bytes_(parent.name)
        field_names = list(value)
        _write_mat73_fields_attribute(group, field_names)
        for field_name in field_names:
            _write_mat73_value(group, field_name, value[field_name])
        return

    if isinstance(value, str):
        array = np.fromiter((ord(char) for char in value), dtype=np.uint16).reshape(1, -1)
        dataset = parent.create_dataset(name, data=array.T)
        dataset.attrs["MATLAB_class"] = np.bytes_("char")
        dataset.attrs["MATLAB_int_decode"] = np.int64(2)
        dataset.attrs["H5PATH"] = np.bytes_(parent.name)
        return

    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    matlab_class = _matlab_class_for_dtype(array.dtype)
    if matlab_class == "logical":
        array = array.astype(np.uint8, copy=False)
    if array.size == 0:
        dataset = parent.create_dataset(
            name,
            data=np.asarray(array.shape, dtype=np.uint64),
        )
        dataset.attrs["MATLAB_empty"] = np.uint8(1)
    else:
        stored = np.transpose(array, axes=tuple(reversed(range(array.ndim))))
        options: dict[str, object] = {}
        if array.nbytes >= 1024**2:
            options.update(compression="gzip", compression_opts=4, shuffle=True)
        dataset = parent.create_dataset(name, data=stored, **options)
    dataset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
    dataset.attrs["H5PATH"] = np.bytes_(parent.name)


def _write_mat73(path: Path, payload: dict) -> None:
    import h5py

    with h5py.File(path, "w", userblock_size=512) as handle:
        for name, value in payload.items():
            _write_mat73_value(handle, str(name), value)
        handle.flush()
    description = (
        "MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: "
        f"{datetime.now().strftime('%a %b %d %H:%M:%S %Y')} HDF5 schema 1.00 ."
    )
    header = description.encode("ascii")[:116].ljust(116, b" ")
    header += b"\x00" * 8 + b"\x00\x02IM"
    with Path(path).open("r+b") as stream:
        stream.write(header)
        stream.flush()
        os.fsync(stream.fileno())


def _read_mat73_value(node) -> object:
    import h5py

    if isinstance(node, h5py.Group):
        return {name: _read_mat73_value(child) for name, child in node.items()}
    matlab_class = node.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii", errors="replace")
    if int(node.attrs.get("MATLAB_empty", 0)):
        shape = tuple(int(value) for value in np.asarray(node).reshape(-1))
        dtype = np.float64 if matlab_class == "double" else np.dtype(str(matlab_class))
        return np.empty(shape, dtype=dtype)
    array = np.asarray(node)
    if array.ndim:
        array = np.transpose(array, axes=tuple(reversed(range(array.ndim))))
    if matlab_class == "char":
        return "".join(chr(int(value)) for value in array.reshape(-1))
    if matlab_class == "logical":
        return array.astype(bool, copy=False)
    return array


def _decode_mat73_class(node) -> str:
    value = node.attrs.get("MATLAB_class", b"")
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace")
    return str(value)


def _decode_mat73_fields(value: object) -> list[str]:
    fields: list[str] = []
    for item in np.asarray(value, dtype=object).reshape(-1):
        chars = np.asarray(item).reshape(-1)
        fields.append(b"".join(bytes(char) for char in chars).decode("ascii"))
    return fields


def _validate_mat73_node(node) -> None:
    import h5py

    matlab_class = _decode_mat73_class(node)
    if isinstance(node, h5py.Group):
        if matlab_class != "struct":
            raise ValueError(f"group {node.name} is not a MATLAB struct")
        if "MATLAB_fields" not in node.attrs:
            raise ValueError(f"struct {node.name} has no MATLAB_fields metadata")
        fields = _decode_mat73_fields(node.attrs["MATLAB_fields"])
        if fields != list(node.keys()):
            # HDF5 iterates children lexically while MATLAB_fields preserves
            # struct order, so compare membership separately from ordering.
            if set(fields) != set(node.keys()):
                raise ValueError(f"struct {node.name} field metadata do not match datasets")
        for child in node.values():
            _validate_mat73_node(child)
        return

    supported = {
        "char", "logical", "double", "single",
        "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    }
    if matlab_class not in supported:
        raise ValueError(f"dataset {node.name} has unsupported MATLAB_class {matlab_class!r}")
    if int(node.attrs.get("MATLAB_empty", 0)):
        dimensions = np.asarray(node).reshape(-1)
        if node.dtype != np.dtype(np.uint64) or np.any(dimensions < 0):
            raise ValueError(f"dataset {node.name} has invalid MATLAB empty dimensions")
        return
    if matlab_class == "char" and node.dtype != np.dtype(np.uint16):
        raise ValueError(f"dataset {node.name} has invalid MATLAB char dtype")
    if matlab_class == "logical" and node.dtype not in {np.dtype(np.uint8), np.dtype(bool)}:
        raise ValueError(f"dataset {node.name} has invalid MATLAB logical dtype")
    if matlab_class not in {"char", "logical"}:
        expected = np.dtype(matlab_class)
        if node.dtype.kind != expected.kind or node.dtype.itemsize != expected.itemsize:
            raise ValueError(f"dataset {node.name} dtype does not match MATLAB_class")
    if node.size:
        first = tuple(0 for _ in node.shape)
        node[first]
        last = tuple(size - 1 for size in node.shape)
        if last != first:
            node[last]


def validate_mat_output(
    path: Path, required_key: str, *, load_payload: bool = True
) -> dict:
    """Load a MATLAB output and require its producer's top-level payload."""
    try:
        import h5py

        is_hdf5 = h5py.is_hdf5(path)
    except (ImportError, OSError):
        is_hdf5 = False
    if is_hdf5:
        try:
            with h5py.File(path, "r") as handle:
                if required_key not in handle:
                    raise ValueError(
                        f"MAT output {path} does not contain required "
                        f"{required_key!r} payload"
                    )
                _validate_mat73_node(handle[required_key])
                value = (
                    _read_mat73_value(handle[required_key])
                    if load_payload
                    else None
                )
        except Exception as exc:
            if isinstance(exc, ValueError) and "does not contain required" in str(exc):
                raise
            raise ValueError(f"Invalid MATLAB v7.3 output {path}: {exc}") from exc
        return {required_key: value}
    try:
        loaded = loadmat(path, simplify_cells=True)
    except Exception as exc:
        raise ValueError(f"Invalid MAT output {path}: {exc}") from exc
    if required_key not in loaded:
        raise ValueError(f"MAT output {path} does not contain required {required_key!r} payload")
    return loaded


def atomic_savemat(path: Path, payload: dict, *, required_key: str) -> Path:
    use_v73 = _mat_payload_nbytes(payload) > _MAT_V5_SAFE_PAYLOAD_BYTES

    def writer(temporary: Path) -> None:
        if use_v73:
            _write_mat73(temporary, payload)
            return
        try:
            savemat(temporary, payload, do_compression=True)
        except MatWriteError as exc:
            if "too large" not in str(exc).lower():
                raise
            _write_mat73(temporary, payload)

    return atomic_write_path(
        path,
        writer,
        validator=lambda temporary: validate_mat_output(
            temporary,
            required_key,
            load_payload=not h5py_is_hdf5(temporary),
        ),
    )


def h5py_is_hdf5(path: Path) -> bool:
    try:
        import h5py

        return bool(h5py.is_hdf5(path))
    except (ImportError, OSError):
        return False


def atomic_write_json(path: Path, payload: object) -> Path:
    return atomic_write_path(
        path,
        lambda temporary: temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"),
        validator=lambda temporary: json.loads(temporary.read_text(encoding="utf-8")),
    )


def atomic_save_figure(path: Path, figure, **kwargs) -> Path:
    return atomic_write_path(
        path,
        lambda temporary: figure.savefig(temporary, **kwargs),
        validator=lambda temporary: temporary.is_file() and temporary.stat().st_size > 0
        or (_ for _ in ()).throw(ValueError(f"Invalid figure output: {temporary}")),
    )


@dataclass(frozen=True)
class OpenEphysStreamInfo:
    continuous_dat: Path
    stream_name: str
    ttl_path: Path
    total_channels: int
    sampling_frequency: float
    ephys_channel_indices: list[int]
    adc_channel_indices: list[int]
    ephys_channel_names: list[str]
    adc_channel_names: list[str]

    def __iter__(self):
        """Preserve the historical 5-tuple unpacking contract."""
        yield self.continuous_dat
        yield self.stream_name
        yield self.ttl_path
        yield self.total_channels
        yield self.sampling_frequency


def extract_datetime(path: str) -> datetime:
    m = re.search(r"(\d{6}_\d{6})", path)
    if m:
        return datetime.strptime(m.group(1), "%y%m%d_%H%M%S")
    return datetime.min


def _natural_sort_key(text: str) -> tuple[tuple[int, int | str], ...]:
    parts = [part for part in re.split(r"(\d+)", text) if part]
    key_parts: list[tuple[int, int | str]] = []
    for part in parts:
        if part.isdigit():
            key_parts.append((0, int(part)))
        else:
            key_parts.append((1, part.lower()))
    return tuple(key_parts)


def _fallback_session_name_sort_key(name: str) -> tuple[int, int | tuple[tuple[int, int | str], ...], str]:
    day_match = _DAY_PREFIX_PATTERN.match(name)
    if day_match is not None:
        return 0, int(day_match.group(1)), name.lower()
    return 1, _natural_sort_key(name), name.lower()


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
    manual_xml_path: str | Path | None = None,
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

    xml_path = ensure_xml(
        basepath=basepath,
        local_output_dir=local_output_dir,
        basename=basename,
        explicit_xml_path=Path(manual_xml_path) if manual_xml_path else None,
    )
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


def _electrode_type_from_xml(root: ET.Element, default: str = "staggered") -> str:
    description = root.find("generalInfo/description")
    if description is None or not description.text:
        return default
    value = description.text.strip().lower()
    if "neuropixel" in value:
        return "NeuroPixel"
    compact_value = re.sub(r"[\s_/-]+", "", value).replace("×", "x")
    if "buzsaki5x12" in compact_value or "buz5x12" in compact_value:
        return "Buzsaki 5x12"
    if "a5x1216buzlin5mm100200160177" in compact_value:
        return A5X12_16_BUZ_LIN_PROBE_TYPE
    if any(token in value for token in ("flex-g5", "flex_g5", "flex g5", "flex_probe", "flex probe", "flexprobe")):
        return "flex-G5"
    if "staggered" in value:
        return "staggered"
    if "neurogrid" in value or "grid" in value:
        return "neurogrid"
    if "poly2" in value or "poly 2" in value:
        return "poly2"
    if "poly3" in value:
        return "poly3"
    if "poly5" in value:
        return "poly5"
    return default


def derive_probe_assignments_from_xml(
    xml_path: Path | str,
) -> tuple[list[dict[str, object]], list[int]]:
    """Return the default GUI probe assignment and XML-skipped channels.

    XML channel groups are the source of truth for group membership. The
    optional general-info description selects the same geometry used by
    ``build_channel_map_data`` when no explicit GUI assignment is supplied.
    """
    anat_grps, _spk_channels, _has_spk_groups, skipped_channels, root = (
        _load_xml_groups_for_chanmap(Path(xml_path).expanduser())
    )
    if not anat_grps:
        raise ValueError(f"No anatomical channel groups found in XML: {xml_path}")
    electrode_type = _electrode_type_from_xml(root)
    return [
        {
            "type": electrode_type,
            "groups": list(range(len(anat_grps))),
            "x_offset": 0,
        }
    ], skipped_channels


def _normalize_chanmap_layout(layout: str | None) -> str:
    text = str(layout or "").strip()
    key = text.lower().replace("-", "_").replace(" ", "")
    compact_key = re.sub(r"[\s_/-]+", "", text.lower()).replace("×", "x")
    if key in {"neuropixel", "neuro_pixel"}:
        return "NeuroPixel"
    if compact_key in {"buzsaki5x12", "buz5x12"}:
        return "Buzsaki 5x12"
    if compact_key == "a5x1216buzlin5mm100200160177":
        return A5X12_16_BUZ_LIN_PROBE_TYPE
    if key in {"flex", "flexg5", "flex_g5", "flexprobe", "flex_probe"}:
        return "flex-G5"
    if key in {"poly2", "poly3", "poly5"}:
        return key
    if key in {"linear", "edge", "staggered", "neurogrid", "twohundred"}:
        return key
    if key == "doublesided":
        return "double_sided"
    return text or "staggered"


def _cell_explorer_layout_coords(
    n_channels: int,
    layout: str,
    group_index: int,
    *,
    vertical_spacing: float = 20.0,
    shank_spacing: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return channel coordinates matching CellExplorer generateChanCoords.m."""
    layout = _normalize_chanmap_layout(layout)
    x = np.full(n_channels, np.nan, dtype=float)
    y = np.full(n_channels, np.nan, dtype=float)

    if layout in {"linear", "edge"}:
        x[:] = 0.0
        y = -np.arange(n_channels, dtype=float) * vertical_spacing
        x = x + group_index * shank_spacing
        return x, y

    if layout == "staggered":
        max_offset = max(2080.0, 17.0 + 4.0 * max(n_channels, 1))
        horz_offset = np.flip(
            np.concatenate(
                (
                    np.asarray([0.0, 8.5]),
                    np.arange(17.0, max_offset + 0.1, 4.0),
                )
            )
        )
        horz_offset[::2] = -horz_offset[::2]
        x = horz_offset[-n_channels:].astype(float)
        y = -np.arange(n_channels, dtype=float) * vertical_spacing
        x = x + group_index * shank_spacing
        return x, y

    if layout in {"poly2", "poly3", "poly5"}:
        columns = int(layout[-1])
        extrachannels = n_channels % columns
        polyline = np.arange(1, n_channels - extrachannels + 1) % columns

        if layout == "poly2":
            x[np.where(polyline == 1)[0] + extrachannels] = 0.0
            x[np.where(polyline == 0)[0] + extrachannels] = 20.0
            x[:extrachannels] = 0.0
            for xpos, y_offset in [(0.0, 0.0), (20.0, -vertical_spacing / 2.0)]:
                mask = x == xpos
                y[mask] = -np.arange(np.sum(mask), dtype=float) * vertical_spacing + y_offset
        elif layout == "poly3":
            x[np.where(polyline == 1)[0] + extrachannels] = -18.0
            x[np.where(polyline == 2)[0] + extrachannels] = 0.0
            x[np.where(polyline == 0)[0] + extrachannels] = 18.0
            x[:extrachannels] = 0.0
            for xpos, y_offset in [
                (18.0, 0.0),
                (0.0, -vertical_spacing / 2.0 + extrachannels * vertical_spacing),
                (-18.0, 0.0),
            ]:
                mask = x == xpos
                y[mask] = -np.arange(np.sum(mask), dtype=float) * vertical_spacing + y_offset
        else:
            x[np.where(polyline == 1)[0] + extrachannels] = -36.0
            x[np.where(polyline == 2)[0] + extrachannels] = -18.0
            x[np.where(polyline == 3)[0] + extrachannels] = 0.0
            x[np.where(polyline == 4)[0] + extrachannels] = 18.0
            x[np.where(polyline == 0)[0] + extrachannels] = 36.0
            if extrachannels > 0:
                x[:extrachannels] = 18.0 * ((-1.0) ** np.arange(1, extrachannels + 1))
            for xpos, y_offset in [
                (36.0, 0.0),
                (18.0, -vertical_spacing / 2.0),
                (0.0, 0.0),
                (-18.0, -vertical_spacing / 2.0),
                (-36.0, 0.0),
            ]:
                mask = x == xpos
                y[mask] = -np.arange(np.sum(mask), dtype=float) * vertical_spacing + y_offset

        x = x + group_index * shank_spacing
        return x, y

    if layout == "neurogrid":
        x = np.asarray([n_channels - (i + 1) for i in range(n_channels)], dtype=float)
        y = -np.arange(n_channels, dtype=float) * vertical_spacing
        x = x + group_index * vertical_spacing
        return x, y

    if layout == "twohundred":
        x[:] = 0.0
        y[:] = 0.0
        y[1::2] = shank_spacing
        x = x + group_index * shank_spacing
        return x, y

    raise ValueError(f"Unsupported probe geometry layout: {layout}")


def _middle_finger_layout_coords(
    n_channels: int,
    group_index: int,
    n_groups: int,
    *,
    side_n_channels: int | None = None,
    vertical_spacing: float = 20.0,
    shank_spacing: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometry for NeuroNexus A5x12-16-Buz-Lin middle-finger probes.

    Side shanks use the CellExplorer poly2 layout. The center shank is a
    16-site linear shank extending 1200 um deeper than the side-shank tips.
    From the tip, the lower 13 sites are spaced at 100 um and the remaining
    upper sites are spaced at 500 um.
    """
    center_index = n_groups // 2
    if group_index != center_index:
        return _cell_explorer_layout_coords(
            n_channels,
            "poly2",
            group_index,
            vertical_spacing=vertical_spacing,
            shank_spacing=shank_spacing,
        )

    side_count = side_n_channels if side_n_channels is not None else 12
    side_lowest_contact_y = -float(max(side_count - 1, 0)) * vertical_spacing / 2.0
    side_tip_y = side_lowest_contact_y - 35.0
    center_tip_y = side_tip_y - 1200.0
    center_lowest_contact_y = center_tip_y + 50.0
    bottom_dense_count = min(n_channels, 13)
    bottom_offsets = 100.0 * np.arange(bottom_dense_count, dtype=float)
    upper_offsets = (
        bottom_offsets[-1] + 500.0 * np.arange(1, n_channels - bottom_dense_count + 1, dtype=float)
        if n_channels > bottom_dense_count
        else np.asarray([], dtype=float)
    )
    y_from_bottom = center_lowest_contact_y + np.concatenate((bottom_offsets, upper_offsets))

    x = np.full(
        n_channels,
        group_index * shank_spacing + 10.0,
        dtype=float,
    )
    y = y_from_bottom[::-1]
    return x, y


def _buzsaki_5x12_layout_coords(
    n_channels: int,
    group_index: int,
    n_groups: int,
    *,
    vertical_spacing: float = 20.0,
    shank_spacing: float = 200.0,
    linear_spacing: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometry for the 64-channel NeuroNexus Buzsaki 5x12 probe.

    Each side shank has 12 poly2 sites. The center shank adds four
    single-column sites at 200 um pitch above its 12-site poly2 tip cluster.
    Center-shank channel order follows that top-to-bottom sequence, as in
    ``middle_finger``.
    """
    if n_groups != 5:
        raise ValueError(
            f"Buzsaki 5x12 requires 5 channel groups; received {n_groups}"
        )
    center_index = n_groups // 2
    expected_channels = 16 if group_index == center_index else 12
    if n_channels != expected_channels:
        raise ValueError(
            "Buzsaki 5x12 requires group sizes [12, 12, 16, 12, 12]; "
            f"group {group_index} has {n_channels} channels"
        )
    if group_index != center_index:
        return _cell_explorer_layout_coords(
            n_channels,
            "poly2",
            group_index,
            vertical_spacing=vertical_spacing,
            shank_spacing=shank_spacing,
        )

    linear_count = 4
    tip_count = 12
    tip_x, tip_y = _cell_explorer_layout_coords(
        tip_count,
        "poly2",
        group_index,
        vertical_spacing=vertical_spacing,
        shank_spacing=shank_spacing,
    )
    center_x = group_index * shank_spacing + 10.0
    linear_x = np.full(linear_count, center_x, dtype=float)
    linear_y = linear_spacing * np.arange(linear_count, 0, -1, dtype=float)
    return np.concatenate((linear_x, tip_x)), np.concatenate((linear_y, tip_y))


def _flex_g5_layout_coords(
    n_channels: int,
    group_index: int,
    *,
    lateral_spacing: float = 21.5,
    row_spacing: float = 15.0,
    side_spacing: float = 80.0,
    block_spacing: float = 1000.0,
    top_anchor_spacing: float = 800.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates for the 64-channel flex-G5 layout."""
    left = [
        (15, 0.0, 0),
        (49, 0.0, 1),
        (14, -1.0, 2),
        (48, 1.0, 2),
        (13, 0.0, 3),
        (12, -1.0, 4),
        (51, 1.0, 4),
        (50, 0.0, 5),
        (11, -1.0, 6),
        (53, 1.0, 6),
        (10, 0.0, 7),
        (9, -1.0, 8),
        (52, 1.0, 8),
        (55, 0.0, 9),
        (8, -1.0, 10),
        (54, 1.0, 10),
        (7, 0.0, 11),
        (6, -1.0, 12),
        (57, 1.0, 12),
        (56, 0.0, 13),
        (5, -1.0, 14),
        (59, 1.0, 14),
        (4, 0.0, 15),
        (3, -1.0, 16),
        (58, 1.0, 16),
        (61, 0.0, 17),
        (2, -1.0, 18),
        (60, 1.0, 18),
        (1, 0.0, 19),
        (0, -1.0, 20),
        (63, 1.0, 20),
        (62, 0.0, 21),
    ]
    right = [
        (16, 0.0, 0),
        (46, 0.0, 1),
        (47, -1.0, 2),
        (17, 1.0, 2),
        (18, 0.0, 3),
        (44, -1.0, 4),
        (19, 1.0, 4),
        (45, 0.0, 5),
        (42, -1.0, 6),
        (20, 1.0, 6),
        (21, 0.0, 7),
        (43, -1.0, 8),
        (22, 1.0, 8),
        (40, 0.0, 9),
        (41, -1.0, 10),
        (23, 1.0, 10),
        (24, 0.0, 11),
        (38, -1.0, 12),
        (25, 1.0, 12),
        (39, 0.0, 13),
        (36, -1.0, 14),
        (26, 1.0, 14),
        (27, 0.0, 15),
        (37, -1.0, 16),
        (28, 1.0, 16),
        (34, 0.0, 17),
        (35, -1.0, 18),
        (29, 1.0, 18),
        (32, 0.0, 19),
        (33, -1.0, 20),
        (30, 1.0, 20),
        (31, 0.0, 21),
    ]

    template_x = np.empty(64, dtype=float)
    template_y = np.empty(64, dtype=float)
    for channel, x_multiplier, row in left:
        template_x[channel] = x_multiplier * lateral_spacing
        template_y[channel] = top_anchor_spacing if row == 0 else -float(row - 1) * row_spacing
    for channel, x_multiplier, row in right:
        template_x[channel] = side_spacing + x_multiplier * lateral_spacing
        template_y[channel] = top_anchor_spacing if row == 0 else -float(row - 1) * row_spacing

    contact_idx = np.arange(n_channels, dtype=int)
    block_idx = contact_idx // 64
    template_idx = contact_idx % 64

    x = template_x[template_idx] + (group_index + block_idx) * block_spacing
    y = template_y[template_idx]
    return x, y


def _flex_probe_layout_coords(n_channels: int, group_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for the flex-G5 geometry helper."""
    return _flex_g5_layout_coords(n_channels, group_index)


def _center_channel_coords_at_x(
    coords: list[dict[str, float | int]], target_x: float | int
) -> None:
    """Place one probe assignment's horizontal bounding-box center at target_x."""
    finite_x = [
        float(coord["x"])
        for coord in coords
        if np.isfinite(float(coord["x"]))
    ]
    if not finite_x:
        return
    current_center = (min(finite_x) + max(finite_x)) / 2.0
    shift = float(target_x) - current_center
    for coord in coords:
        coord["x"] = float(coord["x"]) + shift


def build_channel_map_data(
    basepath: Path | str,
    basename: str | None = None,
    electrode_type: str | None = None,
    reject_channels: list[int] | None = None,
    probe_assignments: list[dict] | None = None,
    xml_path: Path | str | None = None,
    emit_warnings: bool = False,
) -> dict[str, np.ndarray | str] | None:
    reject_channels = reject_channels or []

    base_path = Path(basepath)
    if basename is None:
        basename = base_path.name
        if basename.endswith(".xml"):
            basename = os.path.splitext(basename)[0]

    resolved_xml_path = Path(xml_path).expanduser() if xml_path is not None else base_path / f"{basename}.xml"
    if not resolved_xml_path.exists():
        if emit_warnings:
            print(f"Warning: XML file {resolved_xml_path} not found. Set XML first.")
        return None

    anat_grps, spk_channels, has_spk_groups, skipped_channels, root = _load_xml_groups_for_chanmap(resolved_xml_path)

    ngroups = len(anat_grps)
    if ngroups == 0:
        print("Warning: No anatomical groups found in XML.")
        return None

    if electrode_type is None:
        electrode_type = _electrode_type_from_xml(root)

    if not probe_assignments:
        probe_assignments = [
            {"type": electrode_type, "groups": list(range(ngroups)), "x_offset": 0}
        ]

    channel_coords = []
    for probe_idx, probe in enumerate(probe_assignments):
        probe_coord_start = len(channel_coords)
        p_type = _normalize_chanmap_layout(probe.get("type", electrode_type))
        p_groups = probe.get("groups", [])
        p_x_offset = probe.get("x_offset", 0)
        center_group_index = len(p_groups) // 2
        side_group_lengths = [
            len(anat_grps[group_idx])
            for local_group_idx, group_idx in enumerate(p_groups)
            if local_group_idx != center_group_index and 0 <= group_idx < ngroups
        ]
        side_n_channels = side_group_lengths[0] if side_group_lengths else None

        if p_type == "flex-G5":
            valid_groups = [int(g_idx) for g_idx in p_groups if 0 <= int(g_idx) < ngroups]
            local_block_idx = 0
            group_pos = 0
            while group_pos < len(valid_groups):
                first_group = valid_groups[group_pos]
                block_groups = [first_group]
                first_size = len(anat_grps[first_group])
                if (
                    first_size == 32
                    and group_pos + 1 < len(valid_groups)
                    and len(anat_grps[valid_groups[group_pos + 1]]) == 32
                ):
                    block_groups.append(valid_groups[group_pos + 1])
                    group_pos += 2
                else:
                    group_pos += 1

                tchannels = [ch for group_idx in block_groups for ch in anat_grps[group_idx]]
                n_ch = len(tchannels)
                template_x, template_y = _flex_g5_layout_coords(64, local_block_idx)
                k_val = local_block_idx + 1

                for ch in tchannels:
                    template_idx = int(ch) % 64
                    channel_coords.append(
                        {
                            "id": ch,
                            "x": template_x[template_idx] + p_x_offset,
                            "y": template_y[template_idx],
                            "k": k_val,
                            "p": probe_idx + 1,
                        }
                    )
                local_block_idx += max(1, int(np.ceil(n_ch / 64.0)))
            _center_channel_coords_at_x(channel_coords[probe_coord_start:], p_x_offset)
            continue

        for local_idx, g_idx in enumerate(p_groups):
            if g_idx < 0 or g_idx >= ngroups:
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
            elif p_type == A5X12_16_BUZ_LIN_PROBE_TYPE:
                x, y = _middle_finger_layout_coords(
                    n_ch,
                    local_idx,
                    len(p_groups),
                    side_n_channels=side_n_channels,
                )
            elif p_type == "Buzsaki 5x12":
                x, y = _buzsaki_5x12_layout_coords(
                    n_ch,
                    local_idx,
                    len(p_groups),
                )
            else:
                x, y = _cell_explorer_layout_coords(n_ch, p_type, local_idx)

            x = x + p_x_offset
            k_val = g_idx + 1

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
        _center_channel_coords_at_x(channel_coords[probe_coord_start:], p_x_offset)

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

    chanMap = (real_channels + 1).reshape(1, -1)
    chanMap0ind = real_channels.reshape(1, -1)
    save_dict = {
        "chanMap": chanMap.astype(float),
        "chanMap0ind": chanMap0ind.astype(float),
        "connected": connected.reshape(-1, 1).astype(float),
        "xcoords": xcoords.reshape(-1, 1).astype(float),
        "ycoords": ycoords.reshape(-1, 1).astype(float),
        "kcoords": kcoords.reshape(-1, 1).astype(float),
        "probe_ids": pcoords.reshape(-1, 1).astype(float),
        "probe_assignments_json": json.dumps(probe_assignments),
    }

    return save_dict


def create_channel_map(
    basepath: Path | str,
    outputDir: Path | str,
    basename: str | None = None,
    electrode_type: str | None = None,
    reject_channels: list[int] | None = None,
    probe_assignments: list[dict] | None = None,
    xml_path: Path | str | None = None,
    overwrite: bool = True,
) -> Path | None:
    save_dict = build_channel_map_data(
        basepath=basepath,
        basename=basename,
        electrode_type=electrode_type,
        reject_channels=reject_channels,
        probe_assignments=probe_assignments,
        xml_path=xml_path,
    )
    if save_dict is None:
        return None

    out_file = Path(outputDir) / "chanMap.mat"
    if out_file.exists() and not overwrite:
        existing = validate_mat_output(out_file, "chanMap")
        required = ("chanMap", "connected", "xcoords", "ycoords", "kcoords")
        if any(key not in existing for key in required):
            raise ValueError(f"Existing chanMap is incompatible with current output schema: {out_file}")
        for key in required:
            if not _nan_equal(np.asarray(existing[key]), np.asarray(save_dict[key])):
                raise ValueError(f"Existing chanMap is incompatible with current configuration: {out_file}")
        return out_file
    atomic_savemat(out_file, save_dict, required_key="chanMap")
    print(f"Successfully saved chanMap.mat to {out_file}")
    return out_file


def save_cell_explorer_chan_coords(
    *,
    chanmap_data: dict,
    output_dir: Path | str,
    basename: str,
    source: str = "PreprocessPipeline chanMap.mat",
    overwrite: bool = True,
) -> Path:
    chan_map = np.asarray(chanmap_data["chanMap"]).reshape(-1).astype(int)
    xcoords = np.asarray(chanmap_data["xcoords"]).reshape(-1).astype(float)
    ycoords = np.asarray(chanmap_data["ycoords"]).reshape(-1).astype(float)
    n = min(len(chan_map), len(xcoords), len(ycoords))
    if n == 0:
        raise ValueError("Cannot write chanCoords: chanMap has no channels.")

    n_channels = int(np.nanmax(chan_map[:n]))
    x = np.full((n_channels, 1), np.nan, dtype=float)
    y = np.full((n_channels, 1), np.nan, dtype=float)
    for channel_1based, x_value, y_value in zip(chan_map[:n], xcoords[:n], ycoords[:n]):
        channel = int(channel_1based)
        if 1 <= channel <= n_channels:
            x[channel - 1, 0] = float(x_value)
            y[channel - 1, 0] = float(y_value)

    chan_coords = {
        "x": x,
        "y": y,
        "source": source,
        "layout": "custom",
        "shankSpacing": np.asarray([[np.nan]], dtype=float),
        "verticalSpacing": np.asarray([[np.nan]], dtype=float),
    }
    output_path = Path(output_dir) / f"{basename}.chanCoords.channelInfo.mat"
    if output_path.exists() and not overwrite:
        existing = validate_mat_output(output_path, "chanCoords")["chanCoords"]
        try:
            compatible = _nan_equal(np.asarray(existing["x"]), x) and _nan_equal(np.asarray(existing["y"]), y)
        except Exception as exc:
            raise ValueError(f"Invalid existing chanCoords output: {output_path}") from exc
        if not compatible:
            raise ValueError(f"Existing chanCoords is incompatible with current chanMap: {output_path}")
        return output_path
    return atomic_savemat(output_path, {"chanCoords": chan_coords}, required_key="chanCoords")


def _nan_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """MATLAB geometry equality that treats matching NaNs as compatible."""
    left = np.asarray(left).squeeze()
    right = np.asarray(right).squeeze()
    if left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        return bool(np.allclose(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def prepare_chanmap(
    *,
    basepath: Path,
    basename: str,
    local_output_dir: Path,
    probe_assignments: list[dict],
    reject_channels: list[int] | None = None,
    xml_path: Path | None = None,
    overwrite: bool = True,
) -> tuple[Path, list[int]]:
    chanmap_path = create_channel_map(
        basepath=basepath,
        basename=basename,
        outputDir=local_output_dir,
        probe_assignments=probe_assignments,
        reject_channels=reject_channels or [],
        xml_path=xml_path,
        overwrite=overwrite,
    )
    if chanmap_path is None:
        raise RuntimeError("Failed to create chanMap.mat")

    chan = loadmat(chanmap_path)
    save_cell_explorer_chan_coords(
        chanmap_data=chan,
        output_dir=local_output_dir,
        basename=basename,
        overwrite=overwrite,
    )
    connected = np.asarray(chan["connected"]).flatten().astype(int)
    device_ch_inds = np.asarray(chan.get("chanMap0ind", np.asarray(chan["chanMap"]).flatten() - 1)).flatten().astype(int)
    n = min(connected.size, device_ch_inds.size)
    bad_ch_ids = device_ch_inds[:n][connected[:n] == 0].astype(int).tolist()
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
    device_ch_inds = np.asarray(
        chanmap.get("chanMap0ind", np.asarray(chanmap["chanMap"]).flatten() - 1)
    ).flatten().astype(int)
    connected = np.asarray(chanmap["connected"]).flatten().astype(int)
    n = min(connected.size, device_ch_inds.size)
    bad_ch_ids = device_ch_inds[:n][connected[:n] == 0].astype(int).tolist()

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
        root = find_project_root() / "sorting_temp"
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


def _openephys_channel_name(channel: dict) -> str:
    for key in ("name", "channel_name", "label", "source_name"):
        value = channel.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _is_openephys_ephys_channel(channel: dict) -> bool:
    identifier = str(channel.get("identifier", "")).lower()
    units = str(channel.get("units", "")).strip().lower()
    name = _openephys_channel_name(channel)
    return ".ephys" in identifier or (
        units in {"uv", "µv", "μv"}
        and _OPENEPHYS_EPHYS_CHANNEL_PATTERN.match(name) is not None
    )


def _is_openephys_adc_channel(channel: dict) -> bool:
    identifier = str(channel.get("identifier", "")).lower()
    units = str(channel.get("units", "")).strip().lower()
    name = _openephys_channel_name(channel)
    return ".adc" in identifier or (
        units == "v"
        and _OPENEPHYS_ADC_CHANNEL_PATTERN.match(name) is not None
    )


def _openephys_stream_name(entry: dict) -> str:
    folder_name = str(entry["folder_name"]).rstrip("/\\")
    recorded_processor = str(entry.get("recorded_processor", "Record Node")).strip()
    recorded_processor_id = entry.get("recorded_processor_id")
    if recorded_processor_id is not None:
        return f"{recorded_processor} {recorded_processor_id}#{folder_name}"
    return folder_name


def _resolve_openephys_stream_info(recording_root: Path) -> OpenEphysStreamInfo:
    structure = _load_openephys_structure(recording_root)
    continuous_entries = structure.get("continuous", [])
    if not continuous_entries:
        raise FileNotFoundError(f"No continuous streams found in {recording_root / 'structure.oebin'}")

    ephys_candidates: list[tuple[dict, list[int], list[int], list[str], list[str]]] = []
    for entry in continuous_entries:
        folder_name = str(entry.get("folder_name", "")).rstrip("/\\")
        if "memory_usage" in folder_name.lower():
            continue

        channels = entry.get("channels", []) or []
        total_channels = int(entry.get("num_channels", len(channels)))
        ephys_indices: list[int] = []
        adc_indices: list[int] = []
        ephys_names: list[str] = []
        adc_names: list[str] = []
        if channels:
            for index, channel in enumerate(channels):
                if not isinstance(channel, dict):
                    continue
                channel_name = _openephys_channel_name(channel)
                if _is_openephys_ephys_channel(channel):
                    ephys_indices.append(index)
                    ephys_names.append(channel_name or f"CH{index + 1}")
                elif _is_openephys_adc_channel(channel):
                    adc_indices.append(index)
                    # Preserve raw names for identity resolution and diagnostics.
                    # Synthesizing ADC<number> here would turn an unnamed layout
                    # into falsely authoritative metadata.
                    adc_names.append(channel_name)
        elif total_channels > 0:
            ephys_indices = list(range(total_channels))
            ephys_names = [f"CH{index + 1}" for index in ephys_indices]

        if ephys_indices:
            ephys_candidates.append((entry, ephys_indices, adc_indices, ephys_names, adc_names))

    if not ephys_candidates:
        raise ValueError(
            "No Open Ephys continuous stream with ephys channels was found in "
            f"{recording_root / 'structure.oebin'}"
        )
    if len(ephys_candidates) > 1:
        stream_names = ", ".join(_openephys_stream_name(entry) for entry, *_ in ephys_candidates)
        raise ValueError(
            "Multiple Open Ephys continuous streams with ephys channels are unsupported: "
            f"{stream_names}"
        )

    entry, ephys_indices, adc_indices, ephys_names, adc_names = ephys_candidates[0]
    folder_name = str(entry["folder_name"]).rstrip("/\\")
    continuous_dat = recording_root / "continuous" / folder_name / "continuous.dat"
    if not continuous_dat.exists():
        raise FileNotFoundError(f"Missing Open Ephys continuous.dat: {continuous_dat}")

    ttl_path = recording_root / "events" / folder_name / "TTL"
    total_channels = int(entry.get("num_channels", len(entry.get("channels", []))))
    sampling_frequency = float(entry["sample_rate"])
    return OpenEphysStreamInfo(
        continuous_dat=continuous_dat,
        stream_name=_openephys_stream_name(entry),
        ttl_path=ttl_path,
        total_channels=total_channels,
        sampling_frequency=sampling_frequency,
        ephys_channel_indices=ephys_indices,
        adc_channel_indices=adc_indices,
        ephys_channel_names=ephys_names,
        adc_channel_names=adc_names,
    )


def _resolve_openephys_adc_native_orders(info: OpenEphysStreamInfo, recording_root: Path) -> tuple[list[int], str]:
    """Resolve physical ADC identities from an OE stream's channel names."""
    parsed: list[int | None] = []
    for name in info.adc_channel_names:
        match = re.fullmatch(r"\s*adc(\d+)\s*", str(name), flags=re.IGNORECASE)
        parsed.append(int(match.group(1)) - 1 if match is not None else None)
    if not parsed:
        return [], "none"
    if all(order is None for order in parsed):
        warnings.warn(
            "Open Ephys ADC identities are unavailable; using positional identities for "
            f"{recording_root} stream {info.stream_name}",
            RuntimeWarning,
            stacklevel=2,
        )
        return list(range(len(parsed))), "inferred_positional"
    if any(order is None for order in parsed):
        raise ValueError(
            "Ambiguous Open Ephys ADC identities (only some channel names are ADC<number>) "
            f"for {recording_root} stream {info.stream_name}: {info.adc_channel_names}"
        )
    orders = [int(order) for order in parsed]
    if any(order < 0 for order in orders) or len(set(orders)) != len(orders):
        raise ValueError(
            "Ambiguous Open Ephys ADC identities (duplicate or invalid ADC<number> names) "
            f"for {recording_root} stream {info.stream_name}: {info.adc_channel_names}"
        )
    return orders, "openephys_name"


def _find_local_intan_rhd(recording_dir: Path) -> Path | None:
    """Return only an RHD adjacent to one selected Intan recording."""
    preferred = recording_dir / "info.rhd"
    if preferred.exists():
        return preferred
    matches = sorted(recording_dir.glob("*.rhd"))
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class WildMergedInfo:
    ephys_sampling_frequency: float
    ephys_num_channels: int
    ephys_sample_count: int
    analog_sampling_frequency: float
    analog_num_channels: int
    analog_sample_count: int


def _resolve_wild_merged_info(recording_dir: Path) -> WildMergedInfo | None:
    manifest_path = recording_dir / "wild_preprocess_run.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        merge = payload["merge"]
        ephys_rate = float(merge["fs"])
        ephys_channels = int(merge["n_channels"])
        ephys_samples = int(merge["n_samples"])
        analog_channels = int(merge["analog_channels"])
        analog_samples = int(merge["analog_samples"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid WILD preprocess manifest: {manifest_path}: {exc}") from exc
    if min(ephys_rate, ephys_channels, ephys_samples, analog_channels, analog_samples) <= 0:
        raise ValueError(f"WILD preprocess manifest has non-positive merged dimensions: {manifest_path}")
    analog_rate = 1250.0
    expected_analog_samples = int(round(ephys_samples * analog_rate / ephys_rate))
    if abs(analog_samples - expected_analog_samples) > 1:
        raise ValueError(
            f"WILD analog duration is inconsistent with 1250 Hz metadata: "
            f"manifest={analog_samples}, expected={expected_analog_samples} ({manifest_path})"
        )
    return WildMergedInfo(
        ephys_sampling_frequency=ephys_rate,
        ephys_num_channels=ephys_channels,
        ephys_sample_count=ephys_samples,
        analog_sampling_frequency=analog_rate,
        analog_num_channels=analog_channels,
        analog_sample_count=analog_samples,
    )


def _resolve_intan_adc_layout(
    *, analog_path: Path | None, sample_count: int, recording_dir: Path
) -> tuple[int, list[int], str]:
    """Resolve one Intan sidecar width and identity layout without root-RHD fallback."""
    if analog_path is None or not analog_path.exists():
        return 0, [], "none"
    width = _infer_channels_from_file(analog_path, sample_count)
    if width <= 0:
        return 0, [], "none"
    rhd_path = _find_local_intan_rhd(recording_dir)
    header: IntanRhdHeader | None = None
    if rhd_path is not None:
        try:
            header = read_intan_rhd_header(rhd_path)
        except Exception as exc:
            warnings.warn(
                f"Could not read local Intan ADC layout metadata for {recording_dir} ({rhd_path}): {exc}; "
                "using positional identities.",
                RuntimeWarning,
                stacklevel=2,
            )
    if header is not None and header.num_board_adc_channels > 0:
        orders = [int(order) for order in header.board_adc_native_orders]
        if header.num_board_adc_channels != width or len(orders) != width:
            raise ValueError(
                "Contradictory local Intan ADC metadata for "
                f"{recording_dir}: inferred width={width}, reported count="
                f"{header.num_board_adc_channels}, reported identities={orders}"
            )
        if any(order < 0 for order in orders) or len(set(orders)) != len(orders):
            raise ValueError(f"Invalid local Intan ADC identities for {recording_dir}: {orders}")
        return width, orders, "intan_rhd"
    warnings.warn(
        "Local Intan ADC layout metadata are missing, unreadable, or report zero channels; "
        f"using positional identities for {recording_dir}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return width, list(range(width)), "inferred_positional"


def _copy_if_different(source: Path, target: Path, *, overwrite: bool = False) -> Path:
    if source.resolve() == target.resolve():
        return target
    if target.exists() and not overwrite:
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            return target
        raise FileExistsError(
            f"Existing metadata output conflicts with selected source: {target}. Set overwrite=True to replace it."
        )
    atomic_write_path(
        target,
        lambda temporary: copy2(source, temporary),
        validator=lambda temporary: temporary.is_file() and temporary.stat().st_size == source.stat().st_size,
    )
    return target


def ensure_xml(
    basepath: Path,
    local_output_dir: Path,
    basename: str,
    *,
    explicit_xml_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    target = local_output_dir / f"{basename}.xml"

    if explicit_xml_path is not None:
        explicit = Path(explicit_xml_path).expanduser().resolve()
        if not explicit.exists() or not explicit.is_file():
            raise FileNotFoundError(f"Selected XML file does not exist: {explicit}")
        return _copy_if_different(explicit, target, overwrite=overwrite)

    base_xml = basepath / f"{basename}.xml"
    if base_xml.exists():
        return _copy_if_different(base_xml, target, overwrite=overwrite)

    raise FileNotFoundError(
        f"No XML file selected and no basename XML found. Expected {base_xml}; "
        "use Load XML to select one before running."
    )


def find_rhd_source(basepath: Path, basename: str, *, use_first_child_match: bool = False) -> Path | None:
    preferred = [basepath / f"{basename}.rhd", basepath / "info.rhd"]

    for src in preferred:
        if src is not None and src.exists():
            return src

    child_matches = _direct_child_file_candidates(basepath, "info.rhd", f"{basename}.rhd")
    if len(child_matches) == 1:
        return child_matches[0]
    if len(child_matches) > 1:
        ordered_matches = sorted(
            child_matches,
            key=lambda path: _subsession_sort_key(path.parent / "amplifier.dat"),
        )
        return ordered_matches[0]

    local_rhd = sorted(basepath.glob("*.rhd"))
    if len(local_rhd) == 1:
        return local_rhd[0]
    if len(local_rhd) > 1:
        local_rhd = sorted(local_rhd, key=lambda path: _fallback_session_name_sort_key(path.stem))
        return local_rhd[0]

    return None


def ensure_rhd(
    basepath: Path,
    local_output_dir: Path,
    basename: str,
    *,
    use_first_child_match: bool = False,
    overwrite: bool = False,
) -> Path | None:
    target = local_output_dir / f"{basename}.rhd"
    src = find_rhd_source(basepath, basename, use_first_child_match=use_first_child_match)
    if src is not None:
        return _copy_if_different(src, target, overwrite=overwrite)

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
    subsession_order: list[str] | None = None,
) -> list[Path]:
    ignore_folders = ignore_folders or []

    paths: list[Path] = []
    if (basepath / "structure.oebin").exists():
        paths.append(basepath)
    else:
        paths.extend(_discover_openephys_recordings(basepath, ignore_folders))
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
        direct_amp = basepath / "amplifier.dat"
        direct_cont = basepath / "continuous.dat"
        if direct_amp.exists():
            paths = [direct_amp]
        elif direct_cont.exists():
            paths = [direct_cont]

    if not paths:
        return []

    if subsession_order:
        paths = _apply_explicit_subsession_order(
            paths,
            basepath=basepath,
            subsession_order=subsession_order,
        )
    elif alt_sort:
        idx = _normalize_alt_sort_indices(alt_sort, len(paths))
        paths = [paths[i] for i in idx]
    else:
        paths = sorted(paths, key=_subsession_sort_key)

    return paths


def _apply_explicit_subsession_order(
    discovered_paths: list[Path],
    *,
    basepath: Path,
    subsession_order: list[str],
) -> list[Path]:
    base = Path(basepath).expanduser().resolve()

    def resolve_order_entry(value: str) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("subsession_order must not contain empty paths")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()

    discovered_by_path = {
        Path(path).expanduser().resolve(): Path(path) for path in discovered_paths
    }
    requested = [resolve_order_entry(value) for value in subsession_order]
    if len(set(requested)) != len(requested):
        raise ValueError("subsession_order contains duplicate paths")

    requested_set = set(requested)
    discovered_set = set(discovered_by_path)
    if requested_set != discovered_set:
        missing = sorted(str(path) for path in discovered_set - requested_set)
        unexpected = sorted(str(path) for path in requested_set - discovered_set)
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise ValueError(
            "subsession_order must contain every discovered recording exactly once: "
            + "; ".join(details)
        )
    return [discovered_by_path[path] for path in requested]


def _normalize_alt_sort_indices(alt_sort: list[int], n: int) -> list[int]:
    if not alt_sort:
        return list(range(n))
    idx = [int(i) for i in alt_sort]
    if sorted(idx) != list(range(n)):
        raise ValueError(
            f"Invalid alt_sort for {n} subsessions: {alt_sort}. "
            f"Expected a 0-based permutation of [0, {n - 1}]."
        )
    return idx


def _subsession_sort_key(
    path: Path,
) -> tuple[int, tuple[tuple[int, int | tuple[tuple[int, int | str], ...] | str], ...], str]:
    name = path.parent.name
    oe_dt_dir = _find_openephys_datetime_ancestor(path)
    if oe_dt_dir is not None:
        dt = _extract_openephys_datetime(oe_dt_dir.name)
        if dt is not None:
            return 0, ((0, int(dt.strftime("%Y%m%d%H%M%S"))),), str(path)

    intan_match = re.search(r"(\d{6}_\d{6})", name)
    if intan_match:
        dt = datetime.strptime(intan_match.group(1), "%y%m%d_%H%M%S")
        return 0, ((0, int(dt.strftime("%Y%m%d%H%M%S"))),), str(path)

    oe_dt = _extract_openephys_datetime(name)
    if oe_dt is not None:
        return 0, ((0, int(oe_dt.strftime("%Y%m%d%H%M%S"))),), str(path)

    fallback_rank, fallback_value, fallback_name = _fallback_session_name_sort_key(name)
    if isinstance(fallback_value, tuple):
        fallback_key = fallback_value
    else:
        fallback_key = ((0, int(fallback_value)),)
    return 1, ((0, int(fallback_rank)),) + fallback_key + ((1, fallback_name),), str(path)


def _infer_channels_from_file(path: Path, sample_count: int, bytes_per_sample: int = 2) -> int:
    if sample_count <= 0 or not path.exists():
        return 0
    denom = int(sample_count) * int(bytes_per_sample)
    size = int(path.stat().st_size)
    if denom <= 0 or size % denom != 0:
        raise ValueError(
            f"Cannot infer sidecar channel count for {path}: size={size}, "
            f"sample_count={sample_count}, bytes_per_sample={bytes_per_sample}"
        )
    return int(size // denom)


def _infer_sample_count_from_binary(path: Path, *, n_channels: int, dtype: str) -> int:
    itemsize = np.dtype(dtype).itemsize
    frame_bytes = int(n_channels) * int(itemsize)
    if frame_bytes <= 0:
        raise ValueError(f"Invalid binary frame size for {path}: n_channels={n_channels}, dtype={dtype}")
    size = int(path.stat().st_size)
    if size % frame_bytes != 0:
        raise ValueError(
            f"Binary file size is not divisible by frame size: {path} "
            f"size={size}, frame_bytes={frame_bytes}"
        )
    return int(size // frame_bytes)


def _validate_intan_recording(path: Path, *, n_channels: int, dtype: str) -> int:
    """Check each raw recording before trusting XML dimensions or exporting data.

    A session/root header may describe a different epoch, so only an adjacent
    RHD can validate this recording. File sizes alone cannot prove its width.
    """
    directory = path.parent
    repair_hint = (
        "Back up the originals and reconcile the raw files, local RHD metadata, "
        "and XML before rerunning. Do not fix this by changing the XML alone."
    )
    rhd_path = _find_local_intan_rhd(directory)
    if rhd_path is None and list(directory.glob("*.rhd")):
        raise ValueError(f"Ambiguous local Intan RHD metadata in {directory}. {repair_hint}")
    if rhd_path is not None:
        try:
            header = read_intan_rhd_header(rhd_path)
        except Exception as exc:
            raise ValueError(
                f"Cannot validate Intan channels: unreadable local RHD {rhd_path}: {exc}. {repair_hint}"
            ) from exc
        if header.num_amplifier_channels != n_channels:
            disabled = ", ".join(header.disabled_amplifier_channels) or "none listed"
            raise ValueError(
                f"Intan amplifier channel mismatch in {directory}: XML expects {n_channels}, "
                f"but {rhd_path.name} enables {header.num_amplifier_channels}. "
                f"Disabled amplifier channels: {disabled}. {repair_hint}"
            )
    else:
        warnings.warn(
            f"No local RHD in {directory}; cannot independently verify enabled amplifier "
            f"channels against XML ({n_channels}). Checking available file sizes only.",
            RuntimeWarning,
            stacklevel=2,
        )

    samples = _infer_sample_count_from_binary(path, n_channels=n_channels, dtype=dtype)
    if samples == 0:
        raise ValueError(f"Empty Intan amplifier recording: {path}. {repair_hint}")
    # time.dat is int32; digital files pack all enabled bits into one uint16
    # word per amplifier sample, regardless of the number of enabled lines.
    for name, width in (("time.dat", 4), ("digitalin.dat", 2), ("digitalout.dat", 2)):
        sidecar = directory / name
        if not sidecar.exists():
            continue
        size = sidecar.stat().st_size
        expected = samples * width
        if size != expected:
            count, remainder = divmod(size, width)
            raise ValueError(
                f"Intan stream length mismatch in {directory}: {path.name} has {samples} "
                f"samples at {n_channels} channels; {name} has {count} samples "
                f"and {remainder} trailing bytes ({size} bytes; expected {expected}). "
                f"Possible channel-count mismatch or incomplete recording. {repair_hint}"
            )
    return samples


def build_acquisition_catalog(
    amplifier_paths: list[Path],
    n_amplifier_channels: int,
    dtype: str,
    intan_header: IntanRhdHeader | None = None,
) -> AcquisitionCatalog:
    source_types: list[str] = []
    subsession_names: list[str] = []
    recording_paths: list[Path] = []
    recording_stream_names: list[str | None] = []
    ttl_event_paths_by_subsession: list[Path | None] = []
    resolved_amplifier_paths: list[Path] = []
    sample_counts: list[int] = []
    source_total_channels: list[int] = []
    source_ephys_channels: list[int] = []
    source_adc_channels: list[int] = []
    ephys_channel_indices_by_subsession: list[list[int] | None] = []
    adc_channel_indices_by_subsession: list[list[int]] = []
    adc_channel_names_by_subsession: list[list[str]] = []
    adc_native_orders_by_subsession: list[list[int]] = []
    adc_output_indices_by_subsession: list[list[int]] = []
    adc_layout_sources_by_subsession: list[str] = []
    analogin_source_paths: list[Path | None] = []
    analogin_paths: list[Path] = []
    ephys_sampling_frequencies_by_subsession: list[float | None] = []
    analog_sample_counts_by_subsession: list[int | None] = []
    analog_sampling_frequencies_by_subsession: list[float | None] = []
    digitalin_paths: list[Path] = []
    auxiliary_paths: list[Path] = []
    supply_paths: list[Path] = []
    time_paths: list[Path] = []
    openephys_sampling_frequency: float | None = None
    intan_paths: list[Path] = []
    intan_sample_counts: list[int] = []

    for raw_path in amplifier_paths:
        path = Path(raw_path)
        if path.is_dir() and (path / "structure.oebin").exists():
            info = _resolve_openephys_stream_info(path)
            ephys_channel_count = len(info.ephys_channel_indices)
            if openephys_sampling_frequency is None:
                openephys_sampling_frequency = info.sampling_frequency
            elif not np.isclose(openephys_sampling_frequency, info.sampling_frequency):
                raise ValueError(
                    "Open Ephys recordings with mismatched sampling frequencies are unsupported: "
                    f"{openephys_sampling_frequency} vs {info.sampling_frequency} ({path})"
                )

            source_types.append("openephys")
            subsession_names.append(_openephys_subsession_name(path))
            recording_paths.append(path)
            recording_stream_names.append(info.stream_name)
            ttl_event_paths_by_subsession.append(info.ttl_path)
            resolved_amplifier_paths.append(info.continuous_dat)
            sample_counts.append(
                _infer_sample_count_from_binary(
                    info.continuous_dat,
                    n_channels=info.total_channels,
                    dtype=dtype,
                )
            )
            source_total_channels.append(int(info.total_channels))
            source_ephys_channels.append(ephys_channel_count)
            source_adc_channels.append(len(info.adc_channel_indices))
            ephys_channel_indices_by_subsession.append(list(info.ephys_channel_indices))
            adc_channel_indices_by_subsession.append(list(info.adc_channel_indices))
            adc_channel_names_by_subsession.append(list(info.adc_channel_names))
            adc_orders, adc_layout_source = _resolve_openephys_adc_native_orders(info, path)
            adc_native_orders_by_subsession.append(adc_orders)
            adc_output_indices_by_subsession.append([])
            adc_layout_sources_by_subsession.append(adc_layout_source)
            analogin_source_paths.append(info.continuous_dat if info.adc_channel_indices else None)
            ephys_sampling_frequencies_by_subsession.append(float(info.sampling_frequency))
            analog_sample_counts_by_subsession.append(
                sample_counts[-1] if info.adc_channel_indices else None
            )
            analog_sampling_frequencies_by_subsession.append(
                float(info.sampling_frequency) if info.adc_channel_indices else None
            )
            continue

        d = path.parent
        wild_info = _resolve_wild_merged_info(d)
        if wild_info is None:
            sample_count = _validate_intan_recording(
                path, n_channels=n_amplifier_channels, dtype=dtype
            )
        else:
            sample_count = _infer_sample_count_from_binary(
                path, n_channels=n_amplifier_channels, dtype=dtype
            )
        analog = d / "analogin.dat"
        digital = d / "digitalin.dat"
        aux = d / "auxiliary.dat"
        supply = d / "supply.dat"
        tdat = d / "time.dat"

        intan_adc_channels = 0
        if analog.exists():
            analogin_paths.append(analog)
        if wild_info is not None:
            if sample_count != wild_info.ephys_sample_count:
                raise ValueError(
                    f"WILD amplifier sample count disagrees with {d / 'wild_preprocess_run.json'}: "
                    f"{sample_count} != {wild_info.ephys_sample_count}"
                )
            if int(n_amplifier_channels) != wild_info.ephys_num_channels:
                raise ValueError(
                    f"WILD amplifier channel count disagrees with XML: "
                    f"manifest={wild_info.ephys_num_channels}, xml={n_amplifier_channels}"
                )
            intan_adc_channels = wild_info.analog_num_channels if analog.exists() else 0
            intan_adc_orders = list(range(intan_adc_channels))
            intan_adc_layout_source = "wild_preprocess_manifest"
            if analog.exists():
                expected_bytes = (
                    wild_info.analog_sample_count * wild_info.analog_num_channels * np.dtype("int16").itemsize
                )
                if analog.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"WILD analogin.dat size disagrees with manifest: "
                        f"{analog.stat().st_size} != {expected_bytes} ({analog})"
                    )
        else:
            intan_adc_channels, intan_adc_orders, intan_adc_layout_source = _resolve_intan_adc_layout(
                analog_path=analog if analog.exists() else None,
                sample_count=sample_count,
                recording_dir=d,
            )
        if digital.exists():
            digitalin_paths.append(digital)
        if aux.exists():
            auxiliary_paths.append(aux)
        if supply.exists():
            supply_paths.append(supply)
        if tdat.exists():
            time_paths.append(tdat)

        source_types.append("wild" if wild_info is not None else "intan")
        subsession_names.append(d.name)
        recording_paths.append(d)
        recording_stream_names.append(None)
        ttl_event_paths_by_subsession.append(None)
        resolved_amplifier_paths.append(path)
        sample_counts.append(sample_count)
        source_total_channels.append(int(n_amplifier_channels))
        source_ephys_channels.append(int(n_amplifier_channels))
        source_adc_channels.append(intan_adc_channels)
        ephys_channel_indices_by_subsession.append(None)
        adc_channel_indices_by_subsession.append(list(range(intan_adc_channels)))
        adc_channel_names_by_subsession.append([f"ADC{i + 1}" for i in range(intan_adc_channels)])
        adc_native_orders_by_subsession.append(intan_adc_orders)
        adc_output_indices_by_subsession.append([])
        adc_layout_sources_by_subsession.append(intan_adc_layout_source)
        analogin_source_paths.append(analog if analog.exists() else None)
        ephys_sampling_frequencies_by_subsession.append(
            wild_info.ephys_sampling_frequency if wild_info is not None else None
        )
        analog_sample_counts_by_subsession.append(
            wild_info.analog_sample_count
            if wild_info is not None and analog.exists()
            else (sample_count if analog.exists() else None)
        )
        analog_sampling_frequencies_by_subsession.append(
            wild_info.analog_sampling_frequency
            if wild_info is not None and analog.exists()
            else None
        )
        if wild_info is None:
            intan_paths.append(path)
            intan_sample_counts.append(sample_count)

    sidecar_sample_counts = {p.parent: int(n) for p, n in zip(intan_paths, intan_sample_counts, strict=True)}

    def _infer_channels_for_sidecar(paths: list[Path]) -> int:
        if not paths:
            return 0
        p = paths[0]
        return _infer_channels_from_file(p, sidecar_sample_counts.get(p.parent, sample_counts[0] if sample_counts else 0))

    aux_ch = _infer_channels_for_sidecar(auxiliary_paths)
    supply_ch = _infer_channels_for_sidecar(supply_paths)
    # ADC width is resolved from each source's native identities below.  In
    # particular, WILD analogin.dat uses 1250 Hz and cannot be inferred using
    # the amplifier sample count.
    adc_ch = 0
    adc_native_orders: list[int] = []

    if intan_header is not None:
        if intan_header.num_aux_input_channels > 0:
            aux_ch = int(intan_header.num_aux_input_channels)
        if intan_header.num_supply_voltage_channels > 0:
            supply_ch = int(intan_header.num_supply_voltage_channels)
    canonical_adc_orders = sorted({
        int(order)
        for orders in adc_native_orders_by_subsession
        for order in orders
    })
    for idx, orders in enumerate(adc_native_orders_by_subsession):
        destinations = [canonical_adc_orders.index(order) for order in orders]
        if len(destinations) != len(orders) or len(set(destinations)) != len(destinations):
            raise ValueError(f"Invalid ADC output mapping for {subsession_names[idx]}: {orders}")
        adc_output_indices_by_subsession[idx] = destinations
    adc_ch = len(canonical_adc_orders)
    adc_native_orders = canonical_adc_orders

    dig_ch = 0
    dig_word_ch = 0
    dig_native_orders: list[int] = []
    if digitalin_paths:
        raw_words = _infer_channels_for_sidecar(digitalin_paths)
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
    has_openephys_ttl = any(path is not None and path.exists() for path in ttl_event_paths_by_subsession)
    if has_openephys_ttl:
        dig_ch = max(dig_ch, 16)
        dig_word_ch = max(dig_word_ch, 1)
        if not dig_native_orders:
            dig_native_orders = list(range(16))

    unique_ephys_channels = sorted(set(int(n) for n in source_ephys_channels))
    if len(unique_ephys_channels) > 1:
        raise ValueError(
            "Recordings with mismatched ephys channel counts are unsupported: "
            f"{unique_ephys_channels}"
        )

    if source_types and all(source_type == "openephys" for source_type in source_types):
        catalog_source_type = "openephys"
        sampling_frequency = openephys_sampling_frequency
        ttl_event_paths = [path for path in ttl_event_paths_by_subsession if path is not None]
        amplifier_channels = unique_ephys_channels[0] if unique_ephys_channels else int(n_amplifier_channels)
    elif source_types and all(source_type == "wild" for source_type in source_types):
        catalog_source_type = "wild"
        sampling_frequency = None
        ttl_event_paths = []
        amplifier_channels = int(n_amplifier_channels)
    else:
        catalog_source_type = "intan"
        sampling_frequency = openephys_sampling_frequency
        ttl_event_paths = (
            ttl_event_paths_by_subsession
            if any(source_type == "openephys" for source_type in source_types)
            else []
        )
        amplifier_channels = int(n_amplifier_channels)

    return AcquisitionCatalog(
        source_type=catalog_source_type,
        subsession_names=subsession_names,
        recording_paths=recording_paths,
        recording_stream_names=recording_stream_names,
        ttl_event_paths=ttl_event_paths,
        amplifier_paths=resolved_amplifier_paths,
        analogin_paths=analogin_paths,
        digitalin_paths=digitalin_paths,
        auxiliary_paths=auxiliary_paths,
        supply_paths=supply_paths,
        time_paths=time_paths,
        sample_counts=sample_counts,
        sampling_frequency=sampling_frequency,
        amplifier_channels=amplifier_channels,
        auxiliary_input_channels=aux_ch,
        supply_voltage_channels=supply_ch,
        board_adc_channels=adc_ch,
        board_digital_input_channels=dig_ch,
        board_digital_word_channels=dig_word_ch,
        board_digital_output_channels=board_dig_out_ch,
        temperature_sensor_channels=temp_sensor_ch,
        board_adc_native_orders=adc_native_orders,
        board_digital_input_native_orders=dig_native_orders,
        source_types=source_types,
        source_total_channels=source_total_channels,
        source_ephys_channels=source_ephys_channels,
        source_adc_channels=source_adc_channels,
        ephys_channel_indices_by_subsession=ephys_channel_indices_by_subsession,
        adc_channel_indices_by_subsession=adc_channel_indices_by_subsession,
        adc_channel_names_by_subsession=adc_channel_names_by_subsession,
        adc_native_orders_by_subsession=adc_native_orders_by_subsession,
        adc_output_indices_by_subsession=adc_output_indices_by_subsession,
        adc_layout_sources_by_subsession=adc_layout_sources_by_subsession,
        analogin_source_paths=analogin_source_paths,
        ephys_sampling_frequencies_by_subsession=ephys_sampling_frequencies_by_subsession,
        analog_sample_counts_by_subsession=analog_sample_counts_by_subsession,
        analog_sampling_frequencies_by_subsession=analog_sampling_frequencies_by_subsession,
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
    def _sanitize_for_json(value):
        if isinstance(value, os.PathLike):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_sanitize_for_json(v) for v in value]
        if isinstance(value, set):
            return [_sanitize_for_json(v) for v in sorted(value, key=str)]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _sanitize_for_mat(value):
        if isinstance(value, dict):
            return {k: _sanitize_for_mat(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize_for_mat(v) for v in value]
        if isinstance(value, tuple):
            return [_sanitize_for_mat(v) for v in value]
        if isinstance(value, os.PathLike):
            return str(value)
        if value is None:
            return ""
        return value

    cfg = _sanitize_for_json(asdict(config))
    cfg["basepath"] = str(config.basepath)
    cfg["localpath"] = str(config.localpath) if config.localpath else None
    cfg["output_dir"] = str(config.output_dir) if config.output_dir else None
    cfg["chanmap_mat_path"] = str(config.chanmap_mat_path) if config.chanmap_mat_path else None
    cfg["xml_path"] = str(config.xml_path) if config.xml_path else None
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
        "sorter_output_dirs": [str(p) for p in result.sorter_output_dirs],
        "sorter_partition_manifest_path": (
            str(result.sorter_partition_manifest_path) if result.sorter_partition_manifest_path else None
        ),
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
    if "chanMap0ind" in ops:
        chan_map_0ind = np.atleast_1d(ops["chanMap0ind"]).flatten().astype(np.int32)
    else:
        chan_map = np.atleast_1d(ops["chanMap"]).flatten()
        chan_map_0ind = (chan_map - 1).astype(np.int32)

    connected = np.atleast_1d(rez["connected"]).flatten().astype(bool)
    xcoords = np.atleast_1d(rez["xcoords"]).flatten()
    ycoords = np.atleast_1d(rez["ycoords"]).flatten()
    n_chan_meta = min(chan_map_0ind.size, connected.size, xcoords.size, ycoords.size)
    chan_map_0ind = chan_map_0ind[:n_chan_meta]
    connected = connected[:n_chan_meta]
    xcoords = xcoords[:n_chan_meta]
    ycoords = ycoords[:n_chan_meta]

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
