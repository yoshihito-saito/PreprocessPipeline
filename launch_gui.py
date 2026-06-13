from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the PreprocessPipeline web GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    temp_root = Path(tempfile.gettempdir())
    os.environ.setdefault("NUMBA_CACHE_DIR", str(temp_root / "preprocess_numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(temp_root / "preprocess_matplotlib"))

    from src.preprocess.gui.web_app import main as web_main

    web_args = ["--host", args.host, "--port", str(args.port)]
    if args.no_browser:
        web_args.append("--no-browser")
    return web_main(web_args)


if __name__ == "__main__":
    raise SystemExit(main())
