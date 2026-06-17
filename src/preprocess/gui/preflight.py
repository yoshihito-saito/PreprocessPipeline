from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.preprocess.io import build_channel_map_data, find_rhd_source
from src.preprocess.paths import resolve_project_path
from src.preprocess.recording import _local_reference_channels_without_neighbors

from .config_model import (
    PipelineGuiSettings,
    REPO_ROOT,
    RunMode,
    latest_sorting_folder,
    postprocess_output_folder_for_sorting,
)


@dataclass(frozen=True)
class CheckResult:
    label: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _check_path(label: str, path: Path | None, *, must_be_dir: bool = False) -> CheckResult:
    if path is None:
        return CheckResult(label, "error", "not set")
    if not path.exists():
        return CheckResult(label, "error", f"not found: {path}")
    if must_be_dir and not path.is_dir():
        return CheckResult(label, "error", f"not a directory: {path}")
    return CheckResult(label, "ok", str(path))


def _repo_relative_path(text: str) -> Path | None:
    stripped = text.strip()
    if not stripped:
        return None
    path = Path(stripped).expanduser()
    if not path.is_absolute():
        path = resolve_project_path(path, root=REPO_ROOT)
    return path.resolve()


def _local_cmr_check(settings: PipelineGuiSettings) -> CheckResult | None:
    p = settings.preprocess
    if not p.do_preprocess or str(p.reference).strip().lower() != "local":
        return None
    basepath = settings.basepath_path
    if basepath is None:
        return None

    try:
        data = build_channel_map_data(
            basepath=basepath,
            basename=settings.basename,
            probe_assignments=p.probe_assignments,
            reject_channels=p.reject_channels,
        )
    except Exception as exc:
        chanmap_path = settings.resolved_chanmap_path()
        if chanmap_path is not None and chanmap_path.exists():
            try:
                data = loadmat(chanmap_path)
            except Exception:
                return CheckResult("Local CMR radius", "warn", f"could not inspect chanMap geometry: {exc}")
        else:
            return CheckResult("Local CMR radius", "warn", f"could not inspect chanMap geometry: {exc}")

    if data is None:
        return None

    x = np.asarray(data["xcoords"]).reshape(-1)
    y = np.asarray(data["ycoords"]).reshape(-1)
    connected = np.asarray(data["connected"]).reshape(-1).astype(bool)
    device_ch = np.asarray(
        data.get("chanMap0ind", np.asarray(data["chanMap"]).reshape(-1) - 1)
    ).reshape(-1).astype(int)
    n = min(x.size, y.size, connected.size, device_ch.size)
    if n <= 1:
        return None

    x = x[:n][connected[:n]]
    y = y[:n][connected[:n]]
    channel_ids = device_ch[:n][connected[:n]].astype(int).tolist()
    if len(channel_ids) <= 1:
        return None

    locations = np.column_stack((x, y))
    isolated = _local_reference_channels_without_neighbors(
        channel_ids=channel_ids,
        locations=locations,
        local_radius_um=tuple(p.local_radius_um),
    )
    if not isolated:
        return CheckResult("Local CMR radius", "ok", f"all connected channels have local references within {tuple(p.local_radius_um)} um")

    diffs = locations[:, None, :] - locations[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = {
        ch: float(np.nanmin(dist[i]))
        for i, ch in enumerate(channel_ids)
        if ch in set(isolated) and np.isfinite(np.nanmin(dist[i]))
    }
    suggested_max = max(nearest.values()) if nearest else None
    shown = ", ".join(str(ch) for ch in isolated[:20])
    suffix = ", ..." if len(isolated) > 20 else ""
    suggestion = (
        f" Increase CMR radius max to at least {suggested_max:.0f} um, or use global/none."
        if suggested_max is not None
        else " Use global/none or add more reference channels."
    )
    return CheckResult(
        "Local CMR radius",
        "error",
        f"channels with no local reference in {tuple(p.local_radius_um)} um: {shown}{suffix}.{suggestion}",
    )


def run_preflight(settings: PipelineGuiSettings, mode: RunMode) -> list[CheckResult]:
    checks: list[CheckResult] = []
    basepath = settings.basepath_path
    output_dir = settings.local_output_dir
    chanmap_path = settings.resolved_chanmap_path()

    checks.append(_check_path("Basepath", basepath, must_be_dir=True))
    if output_dir is None:
        checks.append(CheckResult("Local output", "error", "basepath is required first"))
    else:
        checks.append(CheckResult("Local output", "ok" if output_dir.parent.exists() else "warn", str(output_dir)))

    if basepath is not None:
        xml = basepath / f"{basepath.name}.xml"
        rhd = find_rhd_source(basepath, basepath.name)
        if rhd is None and output_dir is not None:
            local_rhd = output_dir / f"{basepath.name}.rhd"
            if local_rhd.exists():
                rhd = local_rhd
        checks.append(
            CheckResult(
                "Session XML",
                "ok" if xml.exists() else "warn",
                str(xml) if xml.exists() else f"not found at expected path: {xml}",
            )
        )
        checks.append(
            CheckResult(
                "Intan info.rhd",
                "ok" if rhd is not None and rhd.exists() else "warn",
                str(rhd) if rhd is not None and rhd.exists() else "not found in basepath or direct child folders",
            )
        )

    if mode in ("all", "preprocess"):
        if chanmap_path is None:
            checks.append(CheckResult("chanMap", "warn", "not set"))
        else:
            checks.append(
                CheckResult(
                    "chanMap",
                    "ok" if chanmap_path.exists() else "warn",
                    str(chanmap_path) if chanmap_path.exists() else f"will be generated or expected at: {chanmap_path}",
                )
            )
        local_cmr = _local_cmr_check(settings)
        if local_cmr is not None:
            checks.append(local_cmr)
        sorter = settings.preprocess.sorter if settings.preprocess.run_sorter else None
        if sorter and sorter.lower() != "disabled":
            sorter_path = _repo_relative_path(settings.preprocess.sorter_path)
            sorter_config = _repo_relative_path(settings.preprocess.sorter_config_path)
            checks.append(_check_path("Sorter path", sorter_path, must_be_dir=True))
            checks.append(_check_path("Sorter config", sorter_config))
            if "kilosort" in sorter.lower() and sorter.lower() != "kilosort4":
                matlab_text = settings.preprocess.matlab_path.strip()
                if matlab_text:
                    checks.append(_check_path("MATLAB path", Path(matlab_text).expanduser()))
                else:
                    checks.append(CheckResult("MATLAB path", "ok", "auto-detect from PATH"))

    if mode in ("postprocess", "noise_label"):
        phy = settings.postprocess.sorting_phy_folder.strip()
        sorting_folder = Path(phy) if phy else latest_sorting_folder(output_dir)
        if phy:
            checks.append(_check_path("Postprocess sorting folder", Path(phy), must_be_dir=True))
        elif sorting_folder is not None:
            checks.append(
                CheckResult(
                    "Postprocess sorting folder",
                    "ok",
                    f"default: {sorting_folder}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "Postprocess sorting folder",
                    "error",
                    "select a sorting folder or run sorting first",
                )
            )

        if mode == "noise_label" or settings.postprocess.noise_label_only:
            if sorting_folder is not None:
                post_output = postprocess_output_folder_for_sorting(sorting_folder)
                metrics_path = post_output / "quality_metrics.csv"
                mapping_path = post_output / "cluster_si_unit_ids.tsv"
                if metrics_path.exists() and mapping_path.exists():
                    checks.append(CheckResult("Noise labeling metrics", "ok", str(metrics_path)))
                else:
                    missing = metrics_path if not metrics_path.exists() else mapping_path
                    checks.append(
                        CheckResult(
                            "Noise labeling metrics",
                            "error",
                            f"not found: {missing}. Run full postprocess once before using noise-labeling only.",
                        )
                    )
            else:
                checks.append(CheckResult("Noise labeling metrics", "error", "sorting folder is required first"))
        elif output_dir is not None and settings.basename:
            dat_path = output_dir / f"{settings.basename}.dat"
            if dat_path.exists():
                detail = str(dat_path)
                if settings.postprocess.apply_preprocess:
                    detail += " (raw legacy dat; preprocessing will be applied before postprocess)"
                else:
                    detail += " (treated as already preprocessed)"
                checks.append(CheckResult("basename.dat", "ok", detail))
            else:
                checks.append(
                    CheckResult(
                        "basename.dat",
                        "error",
                        f"not found: {dat_path}. Set spike sorting to skip/disabled, run preprocess only to create basename.dat, then run postprocess again.",
                    )
                )
        else:
            checks.append(CheckResult("basename.dat", "error", "basepath is required to resolve local_root/<basename>/<basename>.dat"))
    elif mode == "all":
        checks.append(CheckResult("Postprocess target", "ok", "will use sorter output from this run"))

    if settings.preprocess.overwrite or settings.postprocess.overwrite:
        checks.append(CheckResult("Overwrite", "warn", "one or more overwrite options are enabled"))
    else:
        checks.append(CheckResult("Overwrite", "ok", "overwrite disabled"))

    return checks
