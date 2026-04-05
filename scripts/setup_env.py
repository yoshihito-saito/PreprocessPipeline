from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / "environment.yml"
WINDOWS_ENV_FILE = REPO_ROOT / "environment.windows.yml"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"+ {_format_cmd(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def _parse_env_name(env_file: Path) -> str:
    text = env_file.read_text(encoding="utf-8")
    match = re.search(r"^name:\s*(\S+)\s*$", text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"Could not find an environment name in {env_file}")
    return match.group(1)


def _conda_exe() -> str:
    conda = os.environ.get("CONDA_EXE")
    if conda:
        return conda
    return "conda"


def _conda_env_exists(conda: str, env_name: str) -> bool:
    output = _capture([conda, "env", "list", "--json"])
    data = json.loads(output)
    prefixes = [Path(prefix) for prefix in data.get("envs", [])]
    return any(prefix.name == env_name for prefix in prefixes)


def _create_or_update_env(conda: str, env_file: Path, env_name: str) -> None:
    if _conda_env_exists(conda, env_name):
        _run([conda, "env", "update", "--name", env_name, "--file", str(env_file), "--prune"], cwd=REPO_ROOT)
        return
    _run([conda, "env", "create", "--name", env_name, "--file", str(env_file)], cwd=REPO_ROOT)


def _remove_env(conda: str, env_name: str) -> None:
    if not _conda_env_exists(conda, env_name):
        return
    _run([conda, "remove", "--name", env_name, "--all", "-y"], cwd=REPO_ROOT)


def _conda_run(conda: str, env_name: str, cmd: list[str]) -> None:
    _run([conda, "run", "--no-capture-output", "-n", env_name, *cmd], cwd=REPO_ROOT)


def _verify_windows_stack(conda: str, env_name: str) -> None:
    checks = (
        "import torch; import numpy; import scipy; import spikeinterface; print('torch_then_numpy_ok')",
        "import numpy; import torch; import scipy; import spikeinterface; print('numpy_then_torch_ok')",
    )
    for code in checks:
        _conda_run(conda, env_name, ["python", "-c", code])


def _verify_unix_stack(conda: str, env_name: str) -> None:
    code = (
        "import torch; import numpy; import scipy; import spikeinterface; "
        "print('stack_ok', torch.__version__, torch.cuda.is_available())"
    )
    _conda_run(conda, env_name, ["python", "-c", code])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the preprocess environment and install PyTorch.")
    parser.add_argument("--env-name", help="Override the conda environment name from the yaml file.")
    parser.add_argument(
        "--platform",
        choices=("auto", "windows", "unix"),
        default="auto",
        help="Force the setup strategy. Auto uses the current OS.",
    )
    parser.add_argument(
        "--torch-channel",
        default="auto",
        help="auto, cpu, or an explicit wheel channel such as cu124 or cu130",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete the target conda environment first, then create it from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = platform.system()
    strategy = args.platform
    if strategy == "auto":
        strategy = "windows" if system == "Windows" else "unix"

    conda = _conda_exe()
    env_file = WINDOWS_ENV_FILE if strategy == "windows" else DEFAULT_ENV_FILE
    env_name = args.env_name or _parse_env_name(env_file)

    if args.force_recreate:
        _remove_env(conda, env_name)
    _create_or_update_env(conda, env_file, env_name)

    if strategy == "windows":
        _conda_run(conda, env_name, ["python", "scripts/install_torch.py", "--compute-platform", args.torch_channel])
        _conda_run(conda, env_name, ["python", "-m", "pip", "install", "-e", ".[dev,notebook]"])
        _verify_windows_stack(conda, env_name)
        return

    _conda_run(conda, env_name, ["python", "scripts/install_torch.py", "--compute-platform", args.torch_channel])
    _verify_unix_stack(conda, env_name)


if __name__ == "__main__":
    main()
