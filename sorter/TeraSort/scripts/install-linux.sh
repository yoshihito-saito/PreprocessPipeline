#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This installer supports Linux; use install_and_start.bat on Windows." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver tools were not found. Install a CUDA-capable NVIDIA driver and verify nvidia-smi first." >&2
    exit 1
fi

python_arg="${1:-${TERASORT_PYTHON:-}}"
python_explicit=0
if [[ -n "$python_arg" ]]; then
    python_explicit=1
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [python-executable]" >&2
    exit 2
fi

python_works() {
    "$1" -c 'import struct,sys; sys.exit(0 if sys.version_info[:2] >= (3,10) and sys.version_info[:2] <= (3,14) and struct.calcsize("P") == 8 else 1)' >/dev/null 2>&1
}

base_python=""
if [[ -n "$python_arg" ]]; then
    if [[ "$python_arg" == */* ]]; then
        base_python="$python_arg"
    else
        base_python="$(command -v -- "$python_arg" || true)"
    fi
    if [[ -z "$base_python" || ! -x "$base_python" ]]; then
        echo "Python executable not found: $python_arg" >&2
        exit 1
    fi
    if ! python_works "$base_python"; then
        echo "TeraSort requires 64-bit Python 3.10 through 3.14: $base_python" >&2
        exit 1
    fi
else
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        candidate_path="$(command -v "$candidate")"
        if python_works "$candidate_path"; then
            base_python="$candidate_path"
            break
        fi
    done
fi

venv_python="$repo_root/.venv/bin/python"
if [[ -x "$venv_python" ]]; then
    if ! python_works "$venv_python"; then
        echo "The existing .venv is not 64-bit Python 3.10 through 3.14. Move or remove '$repo_root/.venv' before reinstalling." >&2
        exit 1
    fi
    if [[ "$python_explicit" -eq 1 ]]; then
        requested_version="$("$base_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        environment_version="$("$venv_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "$requested_version" != "$environment_version" ]]; then
            echo "The existing .venv uses Python $environment_version, but $requested_version was requested. Move or remove '$repo_root/.venv' before reinstalling." >&2
            exit 1
        fi
    fi
    echo "Using the existing virtual environment."
else
    if [[ -z "$base_python" ]]; then
        echo "No supported Python found. Install Python 3.10-3.14 or pass its path to this script." >&2
        exit 1
    fi
    echo "Creating the virtual environment with $base_python."
    "$base_python" -m venv "$repo_root/.venv" || {
        echo "Could not create .venv. On Debian/Ubuntu, install the matching python3-venv package." >&2
        exit 1
    }
fi

"$venv_python" -m pip install --upgrade pip setuptools wheel
if ! "$venv_python" -c 'import torch; raise SystemExit(0 if torch.__version__.split("+",1)[0] == "2.10.0" and torch.version.cuda == "12.8" else 1)' >/dev/null 2>&1; then
    echo "Installing PyTorch 2.10.0 with CUDA 12.8 support..."
    "$venv_python" -m pip install --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
fi

echo "Installing or repairing TeraSort and its Python packages..."
"$venv_python" -m pip install -e "$repo_root"
"$venv_python" -m pip check

echo "Checking CUDA access and compiling the detection kernel..."
"$venv_python" -c 'import cupy as cp,sys,torch,terasort; from terasort.candidates.detectors import CudaDetector; q=cp.zeros((16,2),cp.float32); q[3,0]=5; found=cp.asnumpy(CudaDetector().detect(q,floor=3)); print("TeraSort",terasort.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available(),"candidate",found.tolist()); sys.exit(0 if torch.cuda.is_available() and found.tolist()==[6] else 1)'
"$venv_python" "$repo_root/scripts/environment_check.py" progress-smoke
"$venv_python" -m terasort.cli backends
echo "Linux environment is ready. Start the dashboard with ./start_server.sh --no-browser."
