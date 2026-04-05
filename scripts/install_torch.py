from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys


TORCH_CHANNELS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("cu130", (13, 0)),
    ("cu128", (12, 8)),
    ("cu126", (12, 6)),
    ("cu124", (12, 4)),
    ("cu118", (11, 8)),
)


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run(cmd: list[str]) -> None:
    print(f"+ {_format_cmd(cmd)}")
    subprocess.run(cmd, check=True)


def _run_capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _parse_cuda_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_compute_platform() -> tuple[str, str]:
    if platform.system() == "Darwin":
        return "cpu", "macOS uses CPU wheels from the default pip index"

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "cpu", "nvidia-smi was not found; falling back to CPU wheels"

    try:
        output = _run_capture([nvidia_smi])
    except subprocess.CalledProcessError as exc:
        return "cpu", f"nvidia-smi failed ({exc.returncode}); falling back to CPU wheels"

    cuda_version = _parse_cuda_version(output)
    if cuda_version is None:
        return "cpu", "CUDA version was not reported by nvidia-smi; falling back to CPU wheels"

    for channel, minimum_version in TORCH_CHANNELS:
        if cuda_version >= minimum_version:
            return channel, f"detected driver CUDA compatibility {cuda_version[0]}.{cuda_version[1]}"
    return "cpu", f"detected CUDA {cuda_version[0]}.{cuda_version[1]}, below the supported wheel floor"


def resolve_compute_platform(requested: str) -> tuple[str, str]:
    env_override = os.environ.get("PREPROCESS_TORCH_CHANNEL")
    if env_override:
        requested = env_override

    valid_channels = {channel for channel, _ in TORCH_CHANNELS} | {"cpu"}
    if requested == "auto":
        return detect_compute_platform()
    if requested not in valid_channels:
        raise SystemExit(
            f"Unsupported compute platform '{requested}'. Choose auto, cpu, or one of: "
            + ", ".join(sorted(valid_channels))
        )
    return requested, "explicitly requested"


def build_install_command(channel: str) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio"]
    if platform.system() == "Darwin" and channel == "cpu":
        return cmd
    return [*cmd, "--index-url", f"https://download.pytorch.org/whl/{channel}"]


def verify_install() -> None:
    code = (
        "import torch; "
        "print('torch', torch.__version__); "
        "print('torch_cuda', torch.version.cuda); "
        "print('cuda_available', torch.cuda.is_available())"
    )
    _run([sys.executable, "-c", code])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install PyTorch with automatic CUDA wheel selection.")
    parser.add_argument(
        "--compute-platform",
        default="auto",
        help="auto, cpu, or an explicit wheel channel such as cu124 or cu130",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel, reason = resolve_compute_platform(args.compute_platform)
    print(f"Selected PyTorch channel: {channel} ({reason})")
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _run(build_install_command(channel))
    verify_install()


if __name__ == "__main__":
    main()
