"""Rebuild the bundled Windows x64 bridge using installed MSVC tools."""

from pathlib import Path
import json
import os
import subprocess


def build():
    root = Path(__file__).resolve().parents[1]
    source = root / "src/terasort/cublas_bridge.cpp"
    folder = root / "build/native"
    target = root / "src/terasort/_native/cublas_bridge.dll"
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    install = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    if not install:
        raise RuntimeError("MSVC C++ build tools are required")
    vcvars = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
    environment = subprocess.check_output(
        f'cmd.exe /d /s /c "call "{vcvars}" >nul && set"', text=True)
    env = {}
    for line in environment.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            env[key] = value
    folder.mkdir(parents=True, exist_ok=True)
    vc_tools = next(value for key, value in env.items() if key.lower() == "vctoolsinstalldir")
    compiler = Path(vc_tools) / "bin/Hostx64/x64/cl.exe"
    command = [str(compiler), "/nologo", "/LD", "/O2", "/EHsc", "/std:c++17", "/MD",
               str(source), "/link", "/OUT:" + str(target)]
    result = subprocess.run(command, env=env, cwd=folder, text=True, capture_output=True)
    (folder / "build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    (folder / "build.json").write_text(json.dumps({"command": command,
        "compiler_environment": install}, indent=2))
    return target


if __name__ == "__main__":
    print(build())
