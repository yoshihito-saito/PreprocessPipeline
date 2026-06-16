from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the PreprocessPipeline Qt GUI.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Default GUI config JSON to load/save. Relative paths are resolved "
            "from the repository root."
        ),
    )
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    temp_root = Path(tempfile.gettempdir())
    os.environ.setdefault("NUMBA_CACHE_DIR", str(temp_root / "preprocess_numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(temp_root / "preprocess_matplotlib"))
    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        os.environ["PREPROCESS_GUI_DEFAULT_CONFIG"] = str(config_path.resolve())

    sys.argv = [sys.argv[0]]

    from src.preprocess.gui.app import main as qt_main

    return qt_main()


if __name__ == "__main__":
    raise SystemExit(main())
