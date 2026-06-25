from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from math import isclose
import re
import shutil
from pathlib import Path

from .io import (
    _find_openephys_datetime_ancestor,
    _infer_sample_count_from_binary,
    _subsession_sort_key,
    discover_subsessions,
    find_rhd_source,
    load_xml_metadata,
)


MULTI_DAY_MANIFEST = "multi_day_manifest.json"


@dataclass(frozen=True)
class MultiDaySubepoch:
    session_index: int
    subepoch_index: int
    session_name: str
    source_session_path: str
    source_subepoch_path: str
    source_dat_path: str
    staged_subepoch_path: str
    source_type: str
    binary_n_channels: int
    binary_sampling_frequency: float | None
    sample_count: int


@dataclass(frozen=True)
class SourceBinaryInfo:
    path: Path
    source_type: str
    n_channels: int
    sampling_frequency: float | None


@dataclass(frozen=True)
class MultiDayStagingResult:
    name: str
    server_basepath: Path
    local_basepath: Path
    manifest_path: Path
    subepochs: list[MultiDaySubepoch]


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "session"


def default_multi_day_name(session_paths: list[Path]) -> str:
    if not session_paths:
        raise ValueError("At least one session path is required for multi-day staging.")
    ordered = [_safe_name(path.name) for path in session_paths]
    if len(ordered) == 1:
        return f"multiday_{ordered[0]}"
    return f"multiday_{ordered[0]}_to_{ordered[-1]}"


def _common_parent(paths: list[Path]) -> Path:
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if len(resolved) == 1:
        return resolved[0].parent
    try:
        import os

        common = Path(os.path.commonpath([str(path) for path in resolved]))
        if str(common) == common.anchor:
            return resolved[0].parent
        return common
    except Exception:
        return resolved[0].parent


def _find_session_xml(session_path: Path) -> Path:
    preferred = session_path / f"{session_path.name}.xml"
    if preferred.exists():
        return preferred
    raise FileNotFoundError(
        "No XML file selected and no basename XML found for multi-day session. "
        f"Expected {preferred}; use Load XML to select one before running."
    )


def _source_subepoch_folder(discovered_path: Path) -> Path:
    if discovered_path.is_dir() and (discovered_path / "structure.oebin").exists():
        return _find_openephys_datetime_ancestor(discovered_path) or discovered_path
    return discovered_path.parent


def _source_binary_info(
    discovered_path: Path,
    *,
    xml_n_channels: int,
    xml_sampling_frequency: float,
) -> SourceBinaryInfo:
    if discovered_path.is_dir() and (discovered_path / "structure.oebin").exists():
        from .io import _resolve_openephys_stream_info

        continuous_path, _stream_name, _ttl_path, n_channels, sr = _resolve_openephys_stream_info(discovered_path)
        return SourceBinaryInfo(
            path=continuous_path,
            source_type="openephys",
            n_channels=int(n_channels),
            sampling_frequency=float(sr),
        )
    return SourceBinaryInfo(
        path=discovered_path,
        source_type="intan",
        n_channels=int(xml_n_channels),
        sampling_frequency=float(xml_sampling_frequency),
    )


def _replace_symlink(target: Path, source: Path, *, overwrite: bool) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if not overwrite:
            raise FileExistsError(
                f"Staged multi-day subepoch already exists: {target}. "
                "Enable overwrite to replace it."
            )
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_multi_day_basepath(
    *,
    session_paths: list[Path],
    local_root: Path,
    name: str | None = None,
    server_root: Path | None = None,
    xml_path: Path | None = None,
    dtype: str = "int16",
    overwrite: bool = False,
) -> MultiDayStagingResult:
    sessions = [Path(path).expanduser().resolve() for path in session_paths]
    if len(sessions) < 2:
        raise ValueError("Multi-day staging requires at least two session folders.")
    for session in sessions:
        if not session.exists() or not session.is_dir():
            raise NotADirectoryError(f"Invalid multi-day session folder: {session}")

    multiday_name = _safe_name(name) if name else default_multi_day_name(sessions)
    root = Path(server_root).expanduser().resolve() if server_root is not None else _common_parent(sessions)
    server_basepath = (root / multiday_name).resolve()
    local_basepath = (Path(local_root).expanduser().resolve() / multiday_name).resolve()
    server_basepath.mkdir(parents=True, exist_ok=True)
    local_basepath.mkdir(parents=True, exist_ok=True)
    if not server_basepath.is_dir():
        raise FileNotFoundError(f"Multi-day server basepath was not created: {server_basepath}")
    if not local_basepath.is_dir():
        raise FileNotFoundError(f"Multi-day local basepath was not created: {local_basepath}")

    staged_xml = server_basepath / f"{multiday_name}.xml"
    has_authoritative_xml = xml_path is not None or staged_xml.exists()
    if xml_path is not None:
        selected_xml = Path(xml_path).expanduser().resolve()
        if not selected_xml.exists() or not selected_xml.is_file():
            raise FileNotFoundError(f"Selected multi-day XML file does not exist: {selected_xml}")
        if selected_xml != staged_xml.resolve():
            shutil.copy2(selected_xml, staged_xml)
        reference_xml = staged_xml
    elif staged_xml.exists():
        reference_xml = staged_xml
    else:
        reference_xml = _find_session_xml(sessions[0])
        shutil.copy2(reference_xml, staged_xml)

    reference_meta = load_xml_metadata(reference_xml)
    if not has_authoritative_xml:
        for session in sessions:
            meta = load_xml_metadata(_find_session_xml(session))
            if int(meta.n_channels) != int(reference_meta.n_channels) or float(meta.sr) != float(reference_meta.sr):
                raise ValueError(
                    "Multi-day sessions must share channel count and sampling rate. "
                    f"{reference_xml} has n_channels={reference_meta.n_channels}, sr={reference_meta.sr}; "
                    f"{session} has n_channels={meta.n_channels}, sr={meta.sr}."
                )

    first_rhd = find_rhd_source(sessions[0], sessions[0].name, use_first_child_match=True)
    if first_rhd is not None:
        shutil.copy2(first_rhd, server_basepath / f"{multiday_name}.rhd")

    subepochs: list[MultiDaySubepoch] = []
    openephys_stream_channels: int | None = None
    openephys_sampling_frequency: float | None = None
    for session_index, session in enumerate(sessions, start=1):
        discovered = discover_subsessions(
            basepath=session,
            sort_files=True,
            alt_sort=None,
            ignore_folders=[],
        )
        if not discovered:
            raise FileNotFoundError(f"No subepochs found for multi-day session: {session}")
        discovered = sorted(discovered, key=_subsession_sort_key)
        for subepoch_index, discovered_path in enumerate(discovered, start=1):
            source_folder = _source_subepoch_folder(discovered_path)
            source_binary = _source_binary_info(
                discovered_path,
                xml_n_channels=int(reference_meta.n_channels),
                xml_sampling_frequency=float(reference_meta.sr),
            )
            if source_binary.source_type == "openephys":
                if openephys_stream_channels is None:
                    openephys_stream_channels = source_binary.n_channels
                elif openephys_stream_channels != source_binary.n_channels:
                    raise ValueError(
                        "Open Ephys recordings with mismatched channel counts are unsupported: "
                        f"{openephys_stream_channels} vs {source_binary.n_channels} ({discovered_path})"
                    )
                if openephys_sampling_frequency is None:
                    openephys_sampling_frequency = source_binary.sampling_frequency
                elif source_binary.sampling_frequency is not None and not isclose(
                    openephys_sampling_frequency,
                    source_binary.sampling_frequency,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        "Open Ephys recordings with mismatched sampling frequencies are unsupported: "
                        f"{openephys_sampling_frequency} vs {source_binary.sampling_frequency} ({discovered_path})"
                    )
            staged_name = (
                f"{len(subepochs) + 1:03d}_"
                f"{_safe_name(session.name)}_"
                f"{_safe_name(source_folder.name)}"
            )
            staged_folder = server_basepath / staged_name
            _replace_symlink(staged_folder, source_folder, overwrite=overwrite)
            sample_count = _infer_sample_count_from_binary(
                source_binary.path,
                n_channels=source_binary.n_channels,
                dtype=dtype,
            )
            subepochs.append(
                MultiDaySubepoch(
                    session_index=session_index,
                    subepoch_index=subepoch_index,
                    session_name=session.name,
                    source_session_path=str(session),
                    source_subepoch_path=str(source_folder),
                    source_dat_path=str(source_binary.path),
                    staged_subepoch_path=str(staged_folder),
                    source_type=source_binary.source_type,
                    binary_n_channels=source_binary.n_channels,
                    binary_sampling_frequency=source_binary.sampling_frequency,
                    sample_count=sample_count,
                )
            )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "name": multiday_name,
        "server_basepath": str(server_basepath),
        "local_basepath": str(local_basepath),
        "source_sessions": [str(path) for path in sessions],
        "xml_path": str(reference_xml),
        "sampling_frequency": float(reference_meta.sr),
        "n_channels": int(reference_meta.n_channels),
        "dtype": dtype,
        "subepochs": [asdict(item) for item in subepochs],
    }
    manifest_path = server_basepath / MULTI_DAY_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (local_basepath / MULTI_DAY_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return MultiDayStagingResult(
        name=multiday_name,
        server_basepath=server_basepath,
        local_basepath=local_basepath,
        manifest_path=manifest_path,
        subepochs=subepochs,
    )
