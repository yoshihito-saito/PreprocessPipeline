"""Install the pipeline into .venv with uv, independently of active Conda envs."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-backend", choices=("cu130", "cu128", "cu126", "cpu"),
                        default="cu130", help="PyTorch backend (default: cu130).")
    args = parser.parse_args()
    system = platform.system()
    if system not in {"Linux", "Windows"}:
        raise SystemExit("uv setup supports Linux and Windows.")
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv was not found. Install uv and put it on PATH, then rerun setup.")
    if shutil.which("git") is None:
        raise SystemExit("Git was not found on PATH; Phy and nelpy require Git.")
    sorter_root = REPO_ROOT / "sorter" / "Kilosort4"
    if not (sorter_root / "kilosort" / "__init__.py").is_file():
        raise SystemExit(f"Vendored Kilosort4 source is missing: {sorter_root}")
    venv = REPO_ROOT / ".venv"
    python = venv / ("Scripts/python.exe" if system == "Windows" else "bin/python")
    if not venv.exists():
        _run([uv, "venv", "--python", "3.11", str(venv)])
    elif not python.is_file() or not (venv / "pyvenv.cfg").is_file():
        raise SystemExit(f"Existing {venv} is not a usable virtual environment; inspect it before retrying.")
    # Never silently reuse an environment with different Python semantics.
    _run([str(python), "-c",
          "import sys; assert sys.version_info[:2] == (3, 11), 'This setup requires Python 3.11'"])
    _run([uv, "pip", "install", "--python", str(python),
          "--torch-backend", args.torch_backend, "--reinstall-package", "torch",
          "--reinstall-package", "torchvision", "-r", str(REPO_ROOT / "requirements.uv.txt")])
    _run([uv, "pip", "check", "--python", str(python)])
    code = (
        "import sys; from pathlib import Path; "
        "import torch, numpy, scipy, spikeinterface, phy, nelpy; "
        "from neuro_py.raw.spike_sorting import phy_log_to_epocharray; "
        f"root = Path({str(sorter_root.resolve())!r}); "
        "sys.path.insert(0, str(root)); import kilosort; "
        "assert Path(kilosort.__file__).resolve() == root / 'kilosort' / '__init__.py'; "
        "print('stack_ok', torch.__version__, 'cuda_available', torch.cuda.is_available()); "
        "print('vendored_kilosort_ok', kilosort.__file__)"
    )
    _run([str(python), "-c", code])
    print(f"Setup complete. Activate {venv} and run preprocess-gui.")


if __name__ == "__main__":
    main()
