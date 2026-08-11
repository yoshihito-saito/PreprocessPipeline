# Default Environment Setup

## Date and Commit

- Date: 2026-06-16
- Commit: uncommitted

## Linked Plan

- [Implementation plan](../implementation_plan/2026-06-16-locked-env-default.md)

## What Changed

- Simplified README installation instructions to one setup entry point:
  `python scripts/setup_env.py`.
- Removed the Windows-specific `setup_env_windows.bat` setup entry point.
- Changed Linux setup to use `environment.linux.yml`.
- Kept Windows setup on `environment.windows.yml`.
- Removed `--use-spec` and the shorter `environment.yml` setup branch.
- Changed the final repository install step from editable install to
  `pip install . --no-deps` so setup does not leave
  `preprocess_pipeline.egg-info` in the repository root.
- Moved MATLAB shim output from repository `.matlab_shim` to the OS temp
  runtime directory.
- Removed the `env_locks/` naming from tracked setup files by renaming the
  Linux environment export to `environment.linux.yml`.
- Removed tracked audit files `env_locks/conda-linux-64.explicit.txt` and
  `env_locks/pip-freeze.txt`.
- Added `numba=0.62.1` and `llvmlite=0.45.1` to the Windows environment file
  because postprocess execution can require `numba` through the
  SpikeInterface stack.
- Added `numba>=0.62,<0.63` to package runtime dependencies so non-conda
  installs also declare the postprocess requirement.

## Why

The previous README exposed multiple setup paths and made users choose between
normal environment files and the Linux full environment export. The new flow
keeps the visible setup path short while still using the reproducible Linux
environment by default.

## Verification

- `python -m py_compile scripts/setup_env.py`
  - Passed.
- `python scripts/setup_env.py --help`
  - Passed.
- `rg -n "setup_env_windows|--unlocked|Locked Environment|locked environment|environment-win-64" README.md scripts .`
  - Passed: no remaining references.
- `python -c "... _resolve_env_file ..."`
  - Passed: Linux selects `environment.linux.yml` and Windows selects
    `environment.windows.yml`.
- `python -m py_compile scripts/setup_env.py src/preprocess/sorter_runner.py`
  - Passed.
- `git diff --check`
  - Passed.
- `git ls-files .matlab_shim env_locks preprocess_pipeline.egg-info`
  - Confirmed `.matlab_shim` and `preprocess_pipeline.egg-info` are not tracked.
- `python -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print('pyproject_ok')"`
  - Passed after adding the `numba` runtime dependency.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "import yaml, pathlib; data=yaml.safe_load(pathlib.Path('environment.windows.yml').read_text()); print(data['name'], [d for d in data['dependencies'] if isinstance(d, str) and ('numba' in d or 'llvmlite' in d)])"`
  - Passed: Windows environment includes `llvmlite=0.45.1` and
    `numba=0.62.1`.
- `git diff --check`
  - Passed after the Windows `numba` dependency update.

Additional attempted checks:

- `python -c "import yaml, pathlib; ..."`
  - Not run in the base Python because `yaml` is not installed there.

## Result

Environment setup now has one documented entry point and two platform files:
`environment.linux.yml` and `environment.windows.yml`. Normal setup no longer
creates repository-root `.matlab_shim` or `preprocess_pipeline.egg-info`
artifacts.

## Known Limitations and Next Steps

- Verify the Windows environment file on a Windows machine before publishing a
  Windows export with tighter pins.
