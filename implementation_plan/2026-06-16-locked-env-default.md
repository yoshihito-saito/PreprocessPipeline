# Default Environment Setup

## Goal and Motivation

Make the setup workflow use one entry point per OS-shaped environment file and
keep Linux/Windows setup structure aligned.

## Current Problem

The README currently presents multiple setup entry points and separates the
known working Linux environment export from the normal setup path. This makes
users choose between setup modes before they know which one is recommended.

The repository currently has a Linux full environment export but its filename
and location expose lock-file implementation details. The Windows environment
file already follows a clearer platform filename.

The setup script also uses editable install, which can leave
`preprocess_pipeline.egg-info` in the repository root. MATLAB sorter runs write
their shim files under `.matlab_shim` in the repository root. Both are generated
artifacts and should not appear in the working tree during normal use.

## Why This Is Needed Now

The standalone GUI is now the primary entry point. Setup instructions should be
short and reproducible so Windows and Linux users can install and launch the
GUI without choosing between multiple environment strategies.

## Affected Modules and Files

- `scripts/setup_env.py`
  - Select `environment.linux.yml` or `environment.windows.yml`.
  - Install the repository non-editably during setup to avoid root egg-info
    artifacts.
- `setup_env_windows.bat`
  - Remove this setup entry point so environment setup starts from one command.
- `README.md`
  - Present `python scripts/setup_env.py` as the single setup entry point.
  - Keep Windows and Linux installation sections short and parallel.
- `environment.linux.yml`
  - Replace `env_locks/environment-linux-64.lock.yml`.
- `environment.windows.yml`
  - Keep Windows setup aligned with runtime postprocess requirements.
- `pyproject.toml`
  - Declare runtime dependencies required by postprocess even when they are
    imported through third-party stack internals.
- `environment.yml`, `env_locks/`
  - Remove older Linux spec/lock naming.
- `src/preprocess/sorter_runner.py`
  - Move MATLAB shim output to a runtime temp directory.
- `implementation_plan/README.md`
- `change_log/`

## Public Parameters and API Changes

No package runtime API changes are planned.

## Expected Behavior

- Linux setup uses `environment.linux.yml`.
- Windows setup uses `environment.windows.yml` for now.
- Windows setup installs `numba`/`llvmlite`, because postprocess execution can
  require `numba` through the SpikeInterface stack.
- PyTorch auto-install remains active for Windows setup.
- Setup runs `pip install . --no-deps`, not editable install, so
  `preprocess_pipeline.egg-info` is not left in the repository root.
- MATLAB shim files are created under a temp runtime directory, not
  `.matlab_shim` in the repository root.

## Verification

- Compile `scripts/setup_env.py`.
- Inspect help output.
- Confirm dry helper behavior by reading the selected file logic in the diff.
- Compile `src/preprocess/sorter_runner.py`.

## Non-Goals

- Do not generate a Windows environment export on Linux.

## 2026-09-10: Portable vendored Kilosort4 setup

- Failure: the Linux export requests unpublished `kilosort==0.1.dev1513+g9f8e7052f`.
- Remove that distribution requirement. Runtime already imports `sorter/Kilosort4`
  directly; preserve that source and all sorting parameters unchanged.
- Keep its runtime dependencies explicit in platform environments; add missing
  Windows dependencies. Do not build the vendored setup.py, whose SCM version
  would depend on the enclosing pipeline checkout.
- Add the official CUDA 13.0 wheel index for existing Linux torch/torchvision pins.
- Verify the vendored import and its resolved file in the target environment
  after stack checks, and fail early if the source is missing.
- Document retrying setup after a partial Conda failure without deleting the env.
- Verify with setup regression tests (including source shadowing/missing source),
  environment dependency checks, and an actual vendored import in the available
  scientific environment. Full fresh Conda creation may require network access.

### Follow-up: unused klustakwik2 build failure

- The next reported setup failure is wheel build-requirement discovery for
  `klustakwik2==0.2.6`; the supplied log omits the underlying exception.
- Remove this unused Python distribution from the Linux environment. Repository
  runtime code does not import it. The inspected user Phy plugins execute the
  separate `~/klustakwik/KlustaKwik` binary; preserve that workflow.
- Document the distinction and extend the environment regression check to prevent
  accidental reintroduction from an environment export. Verify the test fails
  before removal and passes after removal. No clustering algorithm changes.
