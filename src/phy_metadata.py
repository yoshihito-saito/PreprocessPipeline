"""Read declarative Phy metadata without executing a params.py file."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def phy_params_path(folder_or_params: Path) -> Path:
    path = Path(folder_or_params)
    if path.name == "params.py":
        return path
    direct = path / "params.py"
    return direct if direct.exists() else path / "sorter_output" / "params.py"


def read_phy_params(folder_or_params: Path) -> dict[str, Any]:
    path = phy_params_path(folder_or_params)
    if not path.exists():
        return {}
    values: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path)).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                if target.id in {"dat_path", "n_channels_dat", "sample_rate", "dtype", "offset"}:
                    raise ValueError(f"{path}: {target.id} must be a literal value") from None
    return values


def resolve_phy_dat_path(folder: Path) -> Path | None:
    value = read_phy_params(folder).get("dat_path")
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if value is None or value == "None" or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Phy dat_path must name one binary file")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = phy_params_path(folder).parent / path
    return path.resolve()
