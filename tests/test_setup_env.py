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
