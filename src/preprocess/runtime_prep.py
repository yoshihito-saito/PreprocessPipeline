from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import prepare_chanmap
from .multiday import prepare_multi_day_basepath


def prepare_preprocess_settings(settings: Any) -> dict[str, Any]:
    """Resolve runtime-only staging inputs without modifying the saved Run intent."""
    payload: dict[str, Any] = {}
    if settings.multi_day_enabled:
        session_paths = [Path(path) for path in settings.multi_day_session_paths if str(path).strip()]
        if len(session_paths) < 2:
            raise ValueError(
                "Multi-day preprocessing requires at least two session folders. "
                "Use Browse for multi-days and select all session folders before running."
            )
        if not settings.multi_day_name.strip():
            raise ValueError(
                "Multi-day basepath name is required. "
                "Use Browse for multi-days or enter a Multi-day name before running."
            )
        selected_subepoch_paths = [
            Path(path)
            for path in settings.multi_day_selected_subepoch_paths
            if str(path).strip()
        ]
        staged = prepare_multi_day_basepath(
            session_paths=session_paths,
            selected_subepoch_paths=selected_subepoch_paths or None,
            local_root=settings.local_root_path,
            name=settings.multi_day_name.strip(),
            xml_path=settings.resolved_xml_path(),
            dtype="int16",
            overwrite=settings.preprocess.overwrite,
        )
        settings.basepath = str(staged.server_basepath)
        settings.multi_day_name = staged.name
        settings.xml_path = str(staged.server_basepath / f"{staged.name}.xml")
        payload["multi_day_result"] = {
            "name": staged.name,
            "server_basepath": str(staged.server_basepath),
            "local_basepath": str(staged.local_basepath),
            "manifest_path": str(staged.manifest_path),
            "selected_subepochs_csv_path": str(staged.selected_subepochs_csv_path),
            "subepoch_count": len(staged.subepochs),
            "selected_subepoch_count": len(selected_subepoch_paths) or len(staged.subepochs),
        }
        print(f"Prepared multi-day basepath: {staged.server_basepath}", flush=True)
        print(f"Multi-day manifest: {staged.manifest_path}", flush=True)
        print(f"Selected subepochs CSV: {staged.selected_subepochs_csv_path}", flush=True)
    if settings.basepath_path is None:
        raise ValueError("basepath is required.")
    # This runs inside the claimed persistent Stage worker. Rebuild the
    # canonical chanMap from the immutable Run settings so a parameter-change
    # rerun cannot silently consume a stale map, and the GUI never mutates
    # session outputs before acquiring the session claim.
    basepath = settings.preprocess_source_path or settings.basepath_path
    basename = settings.basename
    local_output_dir = settings.local_output_dir
    xml_path = settings.resolved_xml_path()
    if basepath is None or local_output_dir is None or xml_path is None:
        raise ValueError("Cannot resolve source, output, and XML paths for chanMap generation")
    chanmap_path, bad_channels = prepare_chanmap(
        basepath=basepath,
        basename=basename,
        local_output_dir=local_output_dir,
        probe_assignments=settings.preprocess.probe_assignments,
        reject_channels=settings.preprocess.reject_channels,
        xml_path=xml_path,
    )
    print(f"Prepared chanMap: {chanmap_path}", flush=True)
    print(f"Bad channels: {bad_channels}", flush=True)
    settings.chanmap_path = str(chanmap_path)
    return payload
