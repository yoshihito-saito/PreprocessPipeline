from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import os
from math import isclose
import re
import shutil
import uuid
from pathlib import Path

from .io import (
    atomic_write_json,
    atomic_write_path,
    _find_openephys_datetime_ancestor,
    _infer_sample_count_from_binary,
    _subsession_sort_key,
    discover_subsessions,
    find_rhd_source,
    load_xml_metadata,
)


MULTI_DAY_MANIFEST = "multi_day_manifest.json"
MULTI_DAY_SELECTED_SUBEPOCHS_CSV = "multi_day_selected_subepochs.csv"


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
    source_total_channels: int
    source_ephys_channels: int
    source_adc_channels: int
    binary_n_channels: int
    binary_sampling_frequency: float | None
    sample_count: int


@dataclass(frozen=True)
class MultiDayDiscoveredSubepoch:
    session_index: int
    subepoch_index: int
    session_name: str
    source_session_path: str
    discovered_path: str
    source_subepoch_path: str


@dataclass(frozen=True)
class SourceBinaryInfo:
    path: Path
    source_type: str
    binary_n_channels: int
    source_total_channels: int
    source_ephys_channels: int
    source_adc_channels: int
    sampling_frequency: float | None


@dataclass(frozen=True)
class MultiDayStagingResult:
    name: str
    server_basepath: Path
    local_basepath: Path
    manifest_path: Path
    selected_subepochs_csv_path: Path
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


def discover_multi_day_subepochs(
    session_paths: list[Path],
    *,
    require_subepochs: bool = False,
) -> list[MultiDayDiscoveredSubepoch]:
    rows: list[MultiDayDiscoveredSubepoch] = []
    sessions = [Path(path).expanduser().resolve() for path in session_paths]
    for session_index, session in enumerate(sessions, start=1):
        if not session.exists() or not session.is_dir():
            raise NotADirectoryError(f"Invalid multi-day session folder: {session}")
        discovered = discover_subsessions(
            basepath=session,
            sort_files=True,
            alt_sort=None,
            ignore_folders=[],
        )
        if not discovered:
            if require_subepochs:
                raise FileNotFoundError(f"No subepochs found for multi-day session: {session}")
            continue
        discovered = sorted(discovered, key=_subsession_sort_key)
        for subepoch_index, discovered_path in enumerate(discovered, start=1):
            source_folder = _source_subepoch_folder(discovered_path)
            rows.append(
                MultiDayDiscoveredSubepoch(
                    session_index=session_index,
                    subepoch_index=subepoch_index,
                    session_name=session.name,
                    source_session_path=str(session),
                    discovered_path=str(discovered_path),
                    source_subepoch_path=str(source_folder),
                )
            )
    return rows


def _source_binary_info(
    discovered_path: Path,
    *,
    xml_n_channels: int,
    xml_sampling_frequency: float,
) -> SourceBinaryInfo:
    if discovered_path.is_dir() and (discovered_path / "structure.oebin").exists():
        from .io import _resolve_openephys_stream_info

        info = _resolve_openephys_stream_info(discovered_path)
        source_total_channels = int(info.total_channels)
        source_ephys_channels = len(info.ephys_channel_indices)
        source_adc_channels = len(info.adc_channel_indices)
        return SourceBinaryInfo(
            path=info.continuous_dat,
            source_type="openephys",
            binary_n_channels=source_total_channels,
            source_total_channels=source_total_channels,
            source_ephys_channels=source_ephys_channels,
            source_adc_channels=source_adc_channels,
            sampling_frequency=float(info.sampling_frequency),
        )
    return SourceBinaryInfo(
        path=discovered_path,
        source_type="intan",
        binary_n_channels=int(xml_n_channels),
        source_total_channels=int(xml_n_channels),
        source_ephys_channels=int(xml_n_channels),
        source_adc_channels=0,
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
            temporary = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
            temporary.symlink_to(source.resolve(), target_is_directory=source.is_dir())
            os.replace(temporary, target)
            return
    temporary = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
    temporary.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    os.replace(temporary, target)


def _resolve_path_set(paths: list[Path] | None) -> set[str]:
    if not paths:
        return set()
    return {str(Path(path).expanduser().resolve()) for path in paths}


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _cleanup_stale_staged_subepochs(
    *,
    manifest_path: Path,
    active_staged_folders: set[Path],
    staging_root: Path,
    overwrite: bool,
) -> None:
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    active = {path.expanduser().absolute() for path in active_staged_folders}
    for entry in manifest.get("subepochs", []):
        staged_text = entry.get("staged_subepoch_path")
        if not staged_text:
            continue
        staged_path = Path(staged_text).expanduser().absolute()
        try:
            staged_path.relative_to(staging_root.resolve())
        except ValueError:
            raise ValueError(
                f"Refusing to delete manifest path outside multi-day staging root: {staged_path}"
            )
        if staged_path.parent.resolve() != staging_root.resolve():
            raise ValueError(
                f"Refusing to delete non-direct staged child from manifest: {staged_path}"
            )
        if staged_path in active or not (staged_path.exists() or staged_path.is_symlink()):
            continue
        if not overwrite:
            raise FileExistsError(
                f"Stale staged multi-day subepoch exists: {staged_path}. "
                "Enable overwrite to remove subepochs excluded from the current selection."
            )
        if staged_path.is_dir() and not staged_path.is_symlink():
            shutil.rmtree(staged_path)
        else:
            staged_path.unlink()


def _write_selected_subepochs_csv(path: Path, subepochs: list[MultiDaySubepoch]) -> None:
    fieldnames = [
        "staged_order",
        "session_index",
        "subepoch_index",
        "session_name",
        "source_session_path",
        "source_subepoch_path",
        "source_dat_path",
        "staged_subepoch_path",
        "source_type",
        "source_total_channels",
        "source_ephys_channels",
        "source_adc_channels",
        "binary_n_channels",
        "binary_sampling_frequency",
        "sample_count",
    ]
    def _write(temporary: Path) -> None:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for staged_order, subepoch in enumerate(subepochs, start=1):
                row = asdict(subepoch)
                row["staged_order"] = staged_order
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    atomic_write_path(
        path,
        _write,
        validator=lambda temporary: list(csv.DictReader(temporary.open(encoding="utf-8"))),
    )


def prepare_multi_day_basepath(
    *,
    session_paths: list[Path],
    selected_subepoch_paths: list[Path] | None = None,
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
    selected_paths = _resolve_path_set(selected_subepoch_paths)
    selected_path_objects = [Path(path) for path in selected_paths]
    active_session_entries = [
        (session_index, session)
        for session_index, session in enumerate(sessions, start=1)
        if not selected_path_objects
        or any(_path_is_within(path, session) for path in selected_path_objects)
    ]
    if selected_paths and not active_session_entries:
        preview = ", ".join(sorted(selected_paths)[:5])
        suffix = "" if len(selected_paths) <= 5 else f", ... ({len(selected_paths)} total)"
        raise ValueError(
            "Selected multi-day subepoch paths were not found under the selected sessions: "
            f"{preview}{suffix}"
        )
    active_sessions = [session for _session_index, session in active_session_entries]
    for session in active_sessions:
        if not session.exists() or not session.is_dir():
            raise NotADirectoryError(f"Invalid multi-day session folder: {session}")

    multiday_name = _safe_name(name) if name else default_multi_day_name(sessions)
    root = (
        Path(server_root).expanduser().resolve()
        if server_root is not None
        else _common_parent(active_sessions)
    )
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
    xml_copy_source: Path | None = None
    if xml_path is not None:
        selected_xml = Path(xml_path).expanduser().resolve()
        if not selected_xml.exists() or not selected_xml.is_file():
            raise FileNotFoundError(f"Selected multi-day XML file does not exist: {selected_xml}")
        reference_xml = selected_xml
        if selected_xml != staged_xml.resolve():
            xml_copy_source = selected_xml
    elif staged_xml.exists():
        reference_xml = staged_xml
    else:
        reference_xml = _find_session_xml(active_sessions[0])
        xml_copy_source = reference_xml

    reference_meta = load_xml_metadata(reference_xml)
    if not has_authoritative_xml:
        for session in active_sessions:
            meta = load_xml_metadata(_find_session_xml(session))
            if int(meta.n_channels) != int(reference_meta.n_channels) or float(meta.sr) != float(reference_meta.sr):
                raise ValueError(
                    "Multi-day sessions must share channel count and sampling rate. "
                    f"{reference_xml} has n_channels={reference_meta.n_channels}, sr={reference_meta.sr}; "
                    f"{session} has n_channels={meta.n_channels}, sr={meta.sr}."
                )

    first_session = active_sessions[0]
    first_rhd = find_rhd_source(
        first_session, first_session.name, use_first_child_match=True
    )
    staged_rhd = server_basepath / f"{multiday_name}.rhd"

    subepochs: list[MultiDaySubepoch] = []
    openephys_stream_channels: int | None = None
    openephys_sampling_frequency: float | None = None
    discovered_rows = discover_multi_day_subepochs(
        active_sessions, require_subepochs=True
    )
    original_session_indices = [
        session_index for session_index, _session in active_session_entries
    ]
    discovered_rows = [
        replace(
            row,
            session_index=original_session_indices[row.session_index - 1],
        )
        for row in discovered_rows
    ]
    discovered_source_paths = {
        str(Path(row.source_subepoch_path).expanduser().resolve()) for row in discovered_rows
    }
    if selected_paths:
        unmatched = sorted(selected_paths - discovered_source_paths)
        if unmatched:
            preview = ", ".join(unmatched[:5])
            suffix = "" if len(unmatched) <= 5 else f", ... ({len(unmatched)} total)"
            raise ValueError(
                "Selected multi-day subepoch paths were not found under the selected sessions: "
                f"{preview}{suffix}"
            )
    stage_plan: list[tuple[MultiDayDiscoveredSubepoch, Path, Path]] = []
    for staged_index, row in enumerate(
        [
            item
            for item in discovered_rows
            if not selected_paths
            or str(Path(item.source_subepoch_path).expanduser().resolve()) in selected_paths
        ],
        start=1,
    ):
        source_folder = Path(row.source_subepoch_path)
        session = Path(row.source_session_path)
        staged_name = (
            f"{staged_index:03d}_"
            f"{_safe_name(session.name)}_"
            f"{_safe_name(source_folder.name)}"
        )
        stage_plan.append((row, source_folder, server_basepath / staged_name))

    if not stage_plan:
        raise ValueError("No multi-day subepochs were selected for staging.")

    def _same_file(left: Path, right: Path) -> bool:
        if not left.is_file() or not right.is_file() or left.stat().st_size != right.stat().st_size:
            return False
        import hashlib

        def digest(path: Path) -> str:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            return hasher.hexdigest()
        return digest(left) == digest(right)

    # Validate every canonical conflict before copying XML/RHD or changing a
    # link.  Under overwrite=False exact existing files/links are reusable;
    # any difference fails without mutating the established staging tree.
    planned_links = {staged for _row, _source, staged in stage_plan}
    for _row, source, staged in stage_plan:
        if staged.exists() or staged.is_symlink():
            same_link = staged.is_symlink() and staged.resolve() == source.resolve()
            if not same_link and not overwrite:
                raise FileExistsError(
                    f"Staged multi-day subepoch already exists: {staged}. "
                    "Enable overwrite to replace it."
                )
    if xml_copy_source is not None and (staged_xml.exists() or staged_xml.is_symlink()):
        if not _same_file(xml_copy_source, staged_xml) and not overwrite:
            raise FileExistsError(
                f"Multi-day XML already exists with different content: {staged_xml}. "
                "Enable overwrite to replace it."
            )
    if first_rhd is not None and (staged_rhd.exists() or staged_rhd.is_symlink()):
        if not _same_file(first_rhd, staged_rhd) and not overwrite:
            raise FileExistsError(
                f"Multi-day RHD already exists with different content: {staged_rhd}. "
                "Enable overwrite to replace it."
            )

    manifest_path = server_basepath / MULTI_DAY_MANIFEST
    stale_links: list[Path] = []
    if manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot safely update invalid multi-day manifest: {manifest_path}") from exc
        for entry in prior_manifest.get("subepochs", []):
            staged_text = entry.get("staged_subepoch_path")
            if not staged_text:
                continue
            stale = Path(staged_text).expanduser().absolute()
            if stale.parent.resolve() != server_basepath or stale in planned_links:
                if stale not in planned_links:
                    raise ValueError(f"Refusing manifest path outside direct staging children: {stale}")
                continue
            if stale.exists() or stale.is_symlink():
                if not overwrite:
                    raise FileExistsError(
                        f"Stale staged multi-day subepoch exists: {stale}. "
                        "Enable overwrite to remove subepochs excluded from the current selection."
                    )
                stale_links.append(stale)

    manifest_xml_path = staged_xml if xml_copy_source is not None or staged_xml.exists() else reference_xml

    for row, source_folder, staged_folder in stage_plan:
        discovered_path = Path(row.discovered_path)
        session = Path(row.source_session_path)
        session_index = row.session_index
        subepoch_index = row.subepoch_index
        source_binary = _source_binary_info(
            discovered_path,
            xml_n_channels=int(reference_meta.n_channels),
            xml_sampling_frequency=float(reference_meta.sr),
        )
        # Once build_acquisition_catalog exposes per-subepoch channel metadata on
        # this branch, this is the narrow integration point for replacing the
        # local structure.oebin fallback with catalog source_* fields.
        if source_binary.source_type == "openephys":
            if openephys_stream_channels is None:
                openephys_stream_channels = source_binary.source_ephys_channels
            elif openephys_stream_channels != source_binary.source_ephys_channels:
                raise ValueError(
                    "Open Ephys recordings with mismatched channel counts (ephys) are unsupported: "
                    f"{openephys_stream_channels} vs {source_binary.source_ephys_channels} ({discovered_path})"
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
        sample_count = _infer_sample_count_from_binary(
            source_binary.path,
            n_channels=source_binary.binary_n_channels,
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
                source_total_channels=source_binary.source_total_channels,
                source_ephys_channels=source_binary.source_ephys_channels,
                source_adc_channels=source_binary.source_adc_channels,
                binary_n_channels=source_binary.binary_n_channels,
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
        "xml_path": str(manifest_xml_path),
        "sampling_frequency": float(reference_meta.sr),
        "n_channels": int(reference_meta.n_channels),
        "dtype": dtype,
        "subepochs": [asdict(item) for item in subepochs],
    }
    selected_subepochs_csv_path = server_basepath / MULTI_DAY_SELECTED_SUBEPOCHS_CSV
    transaction_id = uuid.uuid4().hex
    # os.replace is only atomic within one filesystem.  Server and local
    # staging commonly reside on distinct mounts, so each side gets its own
    # staging/backup root while the journal below coordinates both swaps.
    server_transaction_root = server_basepath / f".multiday-publish-{transaction_id}"
    local_transaction_root = local_basepath / f".multiday-publish-{transaction_id}"
    server_transaction_root.mkdir()
    local_transaction_root.mkdir()
    try:
        staged_server = server_transaction_root / "staged"
        staged_local = local_transaction_root / "staged"
        staged_server.mkdir()
        staged_local.mkdir()
        server_backups = server_transaction_root / "backups"
        local_backups = local_transaction_root / "backups"
        server_backups.mkdir()
        local_backups.mkdir()
        publications: list[tuple[Path, Path | None, Path]] = []
        for _row, source, target in stage_plan:
            staged = staged_server / target.name
            staged.symlink_to(source.resolve(), target_is_directory=True)
            publications.append((target, staged, server_backups))
        if xml_copy_source is not None:
            staged = staged_server / staged_xml.name
            shutil.copy2(xml_copy_source, staged)
            publications.append((staged_xml, staged, server_backups))
        if first_rhd is not None:
            staged = staged_server / staged_rhd.name
            shutil.copy2(first_rhd, staged)
            publications.append((staged_rhd, staged, server_backups))
        server_manifest_stage = staged_server / MULTI_DAY_MANIFEST
        local_manifest_stage = staged_local / MULTI_DAY_MANIFEST
        server_csv_stage = staged_server / MULTI_DAY_SELECTED_SUBEPOCHS_CSV
        local_csv_stage = staged_local / MULTI_DAY_SELECTED_SUBEPOCHS_CSV
        atomic_write_json(server_manifest_stage, manifest)
        atomic_write_json(local_manifest_stage, manifest)
        _write_selected_subepochs_csv(server_csv_stage, subepochs)
        _write_selected_subepochs_csv(local_csv_stage, subepochs)
        # Validate all staged metadata before touching canonical server/local views.
        json.loads(server_manifest_stage.read_text(encoding="utf-8"))
        json.loads(local_manifest_stage.read_text(encoding="utf-8"))
        for csv_stage in (server_csv_stage, local_csv_stage):
            list(csv.DictReader(csv_stage.open(encoding="utf-8")))
        publications.extend([
            (manifest_path, server_manifest_stage, server_backups),
            (selected_subepochs_csv_path, server_csv_stage, server_backups),
            (local_basepath / MULTI_DAY_MANIFEST, local_manifest_stage, local_backups),
            (local_basepath / MULTI_DAY_SELECTED_SUBEPOCHS_CSV, local_csv_stage, local_backups),
        ])
        publications.extend((stale, None, server_backups) for stale in stale_links)
        applied: list[tuple[Path, Path | None]] = []
        try:
            for index, (target, staged, backup_root) in enumerate(publications):
                backup: Path | None = None
                if target.exists() or target.is_symlink():
                    backup = backup_root / str(index)
                    os.replace(target, backup)
                applied.append((target, backup))
                if staged is not None:
                    os.replace(staged, target)
        except BaseException:
            for target, backup in reversed(applied):
                if target.exists() or target.is_symlink():
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                if backup is not None and backup.exists():
                    # Use rename for rollback so a failed/instrumented publish
                    # replace cannot prevent restoration of the prior view.
                    os.rename(backup, target)
            raise
    finally:
        for transaction_root in (server_transaction_root, local_transaction_root):
            if transaction_root.exists():
                shutil.rmtree(transaction_root, ignore_errors=True)
    return MultiDayStagingResult(
        name=multiday_name,
        server_basepath=server_basepath,
        local_basepath=local_basepath,
        manifest_path=manifest_path,
        selected_subepochs_csv_path=selected_subepochs_csv_path,
        subepochs=subepochs,
    )
