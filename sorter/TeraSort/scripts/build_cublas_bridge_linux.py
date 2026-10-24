"""Build the Linux x86_64 cuBLAS bridge with the installed C++ compiler."""
from pathlib import Path
import json
import os
import platform
import subprocess


def build():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The native bridge build targets Linux x86_64")
    root = Path(__file__).resolve().parents[1]
    source = root / "src/terasort/cublas_bridge_linux.cpp"
    folder = root / "build/native"
    target = root / "src/terasort/_native/cublas_bridge.so"
    folder.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX", "g++")
    command = [compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-pthread",
               str(source), "-ldl", "-o", str(target)]
    result = subprocess.run(command, cwd=folder, text=True, capture_output=True)
    (folder / "linux_build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    (folder / "linux_build.json").write_text(json.dumps({"command": command}, indent=2) + "\n",
                                             encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
