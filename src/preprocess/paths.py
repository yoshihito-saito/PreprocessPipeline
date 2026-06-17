from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT_ENV = "PREPROCESS_PIPELINE_ROOT"


def _candidate_ancestors(path: Path) -> Iterable[Path]:
    current = path.expanduser()
    if current.is_file():
        current = current.parent
    try:
        current = current.resolve()
    except OSError:
        current = current.absolute()
    yield current
    yield from current.parents


def is_project_root(path: Path) -> bool:
    return (path / "sorter").is_dir() and (
        (path / "config").is_dir()
        or (path / "pyproject.toml").is_file()
        or (path / "src" / "preprocess").is_dir()
    )


def find_project_root(start: Path | None = None) -> Path:
    override = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not is_project_root(root):
            raise FileNotFoundError(
                f"{PROJECT_ROOT_ENV} does not point to a PreprocessPipeline root with sorter/: {root}"
            )
        return root

    starts = [Path.cwd()]
    if start is not None:
        starts.append(start)
    starts.append(Path(__file__).resolve())

    seen: set[Path] = set()
    for candidate in starts:
        for root in _candidate_ancestors(candidate):
            if root in seen:
                continue
            seen.add(root)
            if is_project_root(root):
                return root

    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root or find_project_root()) / path
    return path.resolve()
