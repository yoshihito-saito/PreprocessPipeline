from __future__ import annotations

import argparse
import json
from os import PathLike
from pathlib import Path
import traceback
from typing import Any

from src.postprocess import PostprocessConfig, run_postprocess_session
from src.preprocess import prepare_chanmap, run_preprocess_session, select_paths_with_gui
from src.preprocess.multiday import prepare_multi_day_basepath

from .config_model import PipelineGuiSettings, RunMode


RESULT_PREFIX = "__PREPROCESS_GUI_RESULT__"
ERROR_PREFIX = "__PREPROCESS_GUI_ERROR__"


def _json_safe(value: Any) -> Any:
    """Convert GUI result payload values to JSON-serializable objects."""
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _postprocess_config_from_preprocess_result(
    settings: PipelineGuiSettings,
    pre_result: Any,
) -> PostprocessConfig:
    post_config = settings.to_postprocess_config()
    sorter_output_dirs = list(getattr(pre_result, "sorter_output_dirs", []) or [])
    local_output_dir = Path(pre_result.local_output_dir)
    manifest_path = local_output_dir / "sorter_partition_manifest.json"
    if len(sorter_output_dirs) > 1 or manifest_path.exists():
        post_config.sorting_phy_folder = None
    else:
        post_config.sorting_phy_folder = pre_result.sorter_output_dir or post_config.sorting_phy_folder
    post_config.sorting_search_root = local_output_dir
    post_config.dat_path = pre_result.dat_path
    post_config.sampling_frequency = pre_result.sr
    post_config.num_channels = pre_result.n_channels
    post_config.chanmap_mat_path = settings.resolved_chanmap_path()
    post_config.reject_channels = list(pre_result.bad_channels_0based)
    return post_config


def run_pipeline(settings: PipelineGuiSettings, mode: RunMode) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": mode}
    pre_result = None
    if mode in ("all", "preprocess"):
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
            staged = prepare_multi_day_basepath(
                session_paths=session_paths,
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
                "subepoch_count": len(staged.subepochs),
            }
            print(f"Prepared multi-day basepath: {staged.server_basepath}", flush=True)
            print(f"Multi-day manifest: {staged.manifest_path}", flush=True)
        if settings.basepath_path is None:
            raise ValueError("basepath is required.")
        chanmap = settings.resolved_chanmap_path()
        if chanmap is None or not chanmap.exists():
            print("chanMap is missing; generating before preprocess.", flush=True)
            basepath, basename, local_output_dir, _xml_path = select_paths_with_gui(
                use_gui=False,
                manual_basepath=settings.basepath_path,
                local_root=settings.local_root_path,
                manual_xml_path=settings.resolved_xml_path(),
            )
            chanmap_path, bad_channels = prepare_chanmap(
                basepath=basepath,
                basename=basename,
                local_output_dir=local_output_dir,
                probe_assignments=settings.preprocess.probe_assignments,
                reject_channels=settings.preprocess.reject_channels,
                xml_path=_xml_path,
            )
            print(f"Generated chanMap: {chanmap_path}", flush=True)
            print(f"Bad channels: {bad_channels}", flush=True)
            settings.chanmap_path = str(chanmap_path)
        pre_result = run_preprocess_session(settings.to_preprocess_config())
        payload["preprocess_result"] = {
            "sorter_output_dir": str(pre_result.sorter_output_dir) if pre_result.sorter_output_dir else "",
            "sorter_output_dirs": [
                str(path) for path in getattr(pre_result, "sorter_output_dirs", []) or []
            ],
            "sorter_partition_manifest_path": (
                str(pre_result.sorter_partition_manifest_path)
                if getattr(pre_result, "sorter_partition_manifest_path", None)
                else ""
            ),
            "local_output_dir": str(pre_result.local_output_dir),
            "dat_path": str(pre_result.dat_path) if pre_result.dat_path else "",
        }

    if mode in ("all", "postprocess", "noise_label"):
        if mode == "all" and pre_result is not None:
            post_config = _postprocess_config_from_preprocess_result(settings, pre_result)
        else:
            post_config = settings.to_postprocess_config()
        if mode == "noise_label":
            post_config.noise_label_only = True
        post_results = run_postprocess_session(post_config)
        post_result_list = list(post_results) if isinstance(post_results, list) else [post_results]
        analyzer_cache_dirs = [
            str(result.analyzer_cache_dir)
            for result in post_result_list
            if getattr(result, "analyzer_cache_dir", None)
        ]
        payload["postprocess_results"] = {
            "count": len(post_result_list),
            "analyzer_cache_dirs": analyzer_cache_dirs,
        }

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["all", "preprocess", "postprocess", "noise_label"])
    args = parser.parse_args(argv)
    try:
        settings = PipelineGuiSettings.load(Path(args.config))
        result = run_pipeline(settings, args.mode)
        print(f"{RESULT_PREFIX}{json.dumps(_json_safe(result), default=str, sort_keys=True)}", flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }
        print(f"{ERROR_PREFIX}{json.dumps(_json_safe(payload), default=str, sort_keys=True)}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
