from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import prepare_chanmap, discover_subsessions
from .wildx_np import has_wildx_np_marker, session_recordings, prepare_session_metadata
from .multiday import prepare_multi_day_basepath


def prepare_preprocess_settings(settings: Any) -> dict[str, Any]:
    """Resolve runtime-only staging inputs without modifying the saved Run intent."""
    payload: dict[str, Any] = {}
    source_chanmap = settings.resolved_chanmap_path()
    if source_chanmap is not None and not source_chanmap.is_file():
        if settings.chanmap_path.strip():
            raise FileNotFoundError(f"Selected chanMap does not exist: {source_chanmap}")
        source_chanmap = None
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
    # Preserve the selected physical map. Template generation is used only
    # when no existing map was selected before any multi-day path rewrite.
    basepath = settings.preprocess_source_path or settings.basepath_path
    basename = settings.basename
    local_output_dir = settings.local_output_dir
    xml_path = settings.resolved_xml_path()
    if basepath is not None and local_output_dir is not None and has_wildx_np_marker(basepath):
        # Match run_preprocess_session so mapped drives and UNC paths do not
        # produce different provenance for the same source recordings.
        basepath = basepath.resolve()
        recordings = session_recordings(discover_subsessions(
            basepath, sort_files=True, alt_sort=None, ignore_folders=[],
            subsession_order=settings.subsession_order))
        if recordings:
            xml_path, chanmap_path = prepare_session_metadata(
                recordings, output_dir=local_output_dir, basename=basename,
                reject_channels=settings.preprocess.reject_channels,
                overwrite=settings.preprocess.overwrite, source_basepath=basepath,
                lfp_fs=settings.preprocess.lfp_fs,
                probe_assignments=settings.preprocess.probe_assignments)
            settings.xml_path = str(xml_path)
            settings.chanmap_path = str(chanmap_path)
            print(f"Detected acquisition: wildx_np; prepared chanMap: {chanmap_path}", flush=True)
            payload["source_type"] = "wildx_np"
            return payload
    if basepath is None or local_output_dir is None or xml_path is None:
        raise ValueError("Cannot resolve source, output, and XML paths for chanMap generation")
    chanmap_path, bad_channels = prepare_chanmap(
        basepath=basepath,
        basename=basename,
        local_output_dir=local_output_dir,
        probe_assignments=settings.preprocess.probe_assignments,
        reject_channels=settings.preprocess.reject_channels,
        xml_path=xml_path,
        overwrite=settings.preprocess.overwrite,
        source_chanmap_path=source_chanmap,
    )
    print(f"Prepared chanMap: {chanmap_path}", flush=True)
    print(f"Bad channels: {bad_channels}", flush=True)
    settings.chanmap_path = str(chanmap_path)
    return payload
