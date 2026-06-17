from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.postprocess import PostprocessConfig, run_postprocess_session
from src.preprocess import prepare_chanmap, run_preprocess_session, select_paths_with_gui

from .config_model import PipelineGuiSettings, RunMode


RESULT_PREFIX = "__PREPROCESS_GUI_RESULT__"


def _postprocess_config_from_preprocess_result(
    settings: PipelineGuiSettings,
    pre_result: Any,
) -> PostprocessConfig:
    post_config = settings.to_postprocess_config()
    post_config.sorting_phy_folder = pre_result.sorter_output_dir or post_config.sorting_phy_folder
    post_config.sorting_search_root = pre_result.local_output_dir
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
        if settings.basepath_path is None:
            raise ValueError("basepath is required.")
        chanmap = settings.resolved_chanmap_path()
        if chanmap is None or not chanmap.exists():
            print("chanMap is missing; generating before preprocess.", flush=True)
            basepath, basename, local_output_dir, _xml_path = select_paths_with_gui(
                use_gui=False,
                manual_basepath=settings.basepath_path,
                local_root=settings.local_root_path,
            )
            chanmap_path, bad_channels = prepare_chanmap(
                basepath=basepath,
                basename=basename,
                local_output_dir=local_output_dir,
                probe_assignments=settings.preprocess.probe_assignments,
                reject_channels=settings.preprocess.reject_channels,
            )
            print(f"Generated chanMap: {chanmap_path}", flush=True)
            print(f"Bad channels: {bad_channels}", flush=True)
            settings.chanmap_path = str(chanmap_path)
        pre_result = run_preprocess_session(settings.to_preprocess_config())
        payload["preprocess_result"] = {
            "sorter_output_dir": str(pre_result.sorter_output_dir) if pre_result.sorter_output_dir else "",
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
        payload["postprocess_results"] = {
            "count": len(post_results) if hasattr(post_results, "__len__") else 1,
        }

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["all", "preprocess", "postprocess", "noise_label"])
    args = parser.parse_args(argv)
    settings = PipelineGuiSettings.load(Path(args.config))
    result = run_pipeline(settings, args.mode)
    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
