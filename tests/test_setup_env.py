import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_env", ROOT / "scripts/setup_env.py")
setup_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_env)

uv_spec = importlib.util.spec_from_file_location("setup_uv", ROOT / "scripts/setup_uv.py")
setup_uv = importlib.util.module_from_spec(uv_spec)
uv_spec.loader.exec_module(setup_uv)


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_environment_supplies_vendored_dependencies_without_kilosort_distribution(platform):
    env = yaml.safe_load((ROOT / f"environment.{platform}.yml").read_text())
    requirements = []
    for entry in env["dependencies"]:
        requirements.extend(entry["pip"] if isinstance(entry, dict) else [entry])
    names = {entry.split("=")[0].split(">=")[0] for entry in requirements}
    tree = ast.parse((ROOT / "sorter/Kilosort4/setup.py").read_text())
    deps = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "install_deps" for t in node.targets))
    needed = {dep.split(">=")[0] for dep in deps} | {"packaging"}
    if platform == "windows":
        needed.remove("torch")  # Installed by install_torch.py after Conda.
    assert needed <= names
    assert not any(entry.lower().startswith("kilosort") for entry in requirements)
    assert "klustakwik2" not in names  # Optional legacy sorter, not a pipeline dependency.
    if platform == "linux":
        assert "nelpy @ git+https://github.com/nelpy/nelpy.git" in requirements
        assert not any(entry.startswith("nelpy==") for entry in requirements)
        assert "neuro-analysis-py==0.0.2" in requirements
        assert "git" in names
        assert "phy @ git+https://github.com/cortex-lab/phy.git@1ddcd015e0382c3fc0ba20cd99dd5b8771bb8702" in requirements
        assert not any(entry.startswith("phy==") for entry in requirements)
        assert "--extra-index-url https://download.pytorch.org/whl/cu130" in requirements
        assert "torch==2.9.1+cu130" in requirements


def test_missing_source_fails_early(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_env, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="Vendored Kilosort4 source is missing"):
        setup_env._vendored_kilosort_root()


def test_verification_prefers_checkout_over_installed_package(tmp_path, monkeypatch):
    root = tmp_path / "checkout with spaces"
    package = root / "sorter/Kilosort4/kilosort"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '4.1.2'\n")
    shadow = tmp_path / "site-packages/kilosort"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise RuntimeError('wrong Kilosort')\n")
    monkeypatch.setenv("PYTHONPATH", str(shadow.parent))
    monkeypatch.setattr(setup_env, "REPO_ROOT", root)
    def run(conda, env_name, cmd):
        result = subprocess.run([sys.executable, *cmd[1:]], check=True, capture_output=True, text=True)
        assert "vendored_kilosort_ok" in result.stdout
        assert str(package) in result.stdout
    monkeypatch.setattr(setup_env, "_conda_run", run)
    setup_env._verify_vendored_kilosort("conda", "preprocess")


def test_partial_environment_is_updated(monkeypatch):
    calls = []
    monkeypatch.setattr(setup_env, "_conda_env_exists", lambda *args: True)
    monkeypatch.setattr(setup_env, "_run", lambda cmd, **kwargs: calls.append(cmd))
    setup_env._create_or_update_env("conda", setup_env.LINUX_ENV_FILE, "preprocess")
    assert calls == [["conda", "env", "update", "--name", "preprocess", "--file",
                      str(setup_env.LINUX_ENV_FILE), "--prune"]]


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_setup_checks_vendored_source_after_stack(platform, monkeypatch):
    from argparse import Namespace
    events = []
    monkeypatch.setattr(setup_env, "parse_args", lambda: Namespace(
        platform=platform, env_name="preprocess", force_recreate=False, torch_channel="cpu"))
    monkeypatch.setattr(setup_env, "_create_or_update_env", lambda *a: events.append("create"))
    monkeypatch.setattr(setup_env, "_conda_run", lambda *a: events.append("install"))
    monkeypatch.setattr(setup_env, f"_verify_{platform}_stack", lambda *a: events.append("stack"))
    monkeypatch.setattr(setup_env, "_verify_vendored_kilosort", lambda *a: events.append("kilosort"))
    setup_env.main()
    assert events[0] == "create"
    assert events[-2:] == ["stack", "kilosort"]


@pytest.mark.parametrize("system,relative_python", [("Linux", "bin/python"), ("Windows", "Scripts/python.exe")])
@pytest.mark.parametrize("existing", [False, True])
def test_uv_targets_venv_and_preserves_existing_environment(tmp_path, monkeypatch, system, relative_python, existing):
    monkeypatch.setattr(setup_uv, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(setup_uv.platform, "system", lambda: system)
    monkeypatch.setattr(setup_uv.shutil, "which", lambda name: name)
    monkeypatch.setattr(sys, "argv", ["setup_uv.py", "--torch-backend", "cpu"])
    monkeypatch.setenv("CONDA_PREFIX", "/unrelated/conda")
    package = tmp_path / "sorter/Kilosort4/kilosort"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    python = tmp_path / ".venv" / relative_python
    if existing:
        python.parent.mkdir(parents=True)
        python.touch()
        (tmp_path / ".venv/pyvenv.cfg").touch()
    calls = []
    monkeypatch.setattr(setup_uv, "_run", lambda cmd: calls.append(cmd))
    setup_uv.main()
    assert any(cmd[:2] == ["uv", "venv"] for cmd in calls) is not existing
    install = next(cmd for cmd in calls if cmd[:3] == ["uv", "pip", "install"])
    assert install[install.index("--python") + 1] == str(python)
    assert install[install.index("--torch-backend") + 1] == "cpu"
    assert ["uv", "pip", "check", "--python", str(python)] in calls
    assert calls[-1][0] == str(python)
    assert "import kilosort" in calls[-1][-1]


def test_uv_missing_tool_fails_before_mutation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["setup_uv.py"])
    monkeypatch.setattr(setup_uv.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_uv.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup_uv, "_run", lambda cmd: pytest.fail("unexpected mutation"))
    with pytest.raises(SystemExit, match="uv was not found"):
        setup_uv.main()


def test_uv_refuses_to_replace_non_environment_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["setup_uv.py"])
    monkeypatch.setattr(setup_uv, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(setup_uv.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_uv.shutil, "which", lambda name: name)
    package = tmp_path / "sorter/Kilosort4/kilosort"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (tmp_path / ".venv").mkdir()
    marker = tmp_path / ".venv/keep.txt"
    marker.write_text("user data")
    monkeypatch.setattr(setup_uv, "_run", lambda cmd: pytest.fail("unexpected mutation"))
    with pytest.raises(SystemExit, match="not a usable virtual environment"):
        setup_uv.main()
    assert marker.read_text() == "user data"


def test_uv_requirements_cover_sorter_and_preserve_git_sources():
    from packaging.requirements import Requirement
    entries = [line.strip() for line in (ROOT / "requirements.uv.txt").read_text().splitlines()
               if line.strip() and not line.startswith("#")]
    assert "-e .[dev,notebook]" in entries
    reqs = {req.name.lower(): req for req in map(Requirement, (x for x in entries if not x.startswith("-e ")))}
    tree = ast.parse((ROOT / "sorter/Kilosort4/setup.py").read_text())
    deps = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "install_deps" for t in n.targets))
    assert {Requirement(d).name.lower() for d in deps} <= reqs.keys()
    assert str(reqs["torch"].specifier) == "==2.9.1"
    assert reqs["phy"].url.endswith("@1ddcd015e0382c3fc0ba20cd99dd5b8771bb8702")
    assert reqs["nelpy"].url == "git+https://github.com/nelpy/nelpy.git"
    assert not {"kilosort", "klustakwik2"} & reqs.keys()
