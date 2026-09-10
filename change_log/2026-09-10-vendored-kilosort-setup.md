# Vendored Kilosort4 environment setup

- Date: 2026-09-10. Work is uncommitted.
- Plan: [default environment setup, September amendment](../implementation_plan/2026-06-16-locked-env-default.md).

## Changes

- Removed the unpublished Kilosort development distribution from the Linux
  environment. The existing runner continues importing the customized source
  in `sorter/Kilosort4`; no sorting code or parameters were changed by this task.
- Added the official CUDA 13.0 extra wheel index while preserving existing
  torch and torchvision versions. Source:
  https://pytorch.org/get-started/previous-versions/ (v2.9.1).
- Declared missing Windows vendored-sorter dependencies explicitly.
- Setup checks source presence before modifying the environment and verifies
  the actual imported source path after stack verification on both platforms.
- Documented retrying the setup command against a partially created environment.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/spn/bin/python -m pytest tests/test_setup_env.py -q`
  — 7 passed. Covers platform dependency declarations, removal of the invalid
  distribution, wheel index, missing source, source precedence with spaces in
  paths, updating partial environments, and final verification ordering.
- `python scripts/setup_env.py --help` — passed.
- `git diff --check` — passed; inspected task diff.
- Actual import attempted using:

```bash
/local/workdir/ys2375/miniforge3/envs/spn/bin/python -c 'import runpy, subprocess, sys; ns=runpy.run_path("scripts/setup_env.py"); ns["_verify_vendored_kilosort"].__globals__["_conda_run"] = lambda conda, env, cmd: subprocess.run([sys.executable, *cmd[1:]], check=True); ns["_verify_vendored_kilosort"]("conda", "spn")'
```

The check reached the vendored clustering module and failed because the existing
`spn` environment lacks `faiss`. This environment was not modified. Both platform
definitions declare `faiss-cpu`; an actual fresh Conda solve/install, complete
vendored import, Windows execution, and GPU sorting remain unverified.
The Linux environment remains a large pinned export; this change does not claim
to resolve every possible package availability or driver compatibility issue.
Unrelated pre-existing changes were left intact.

## Follow-up: klustakwik2 build failure (uncommitted)

- Removed `klustakwik2==0.2.6` from the Linux pip list. No pipeline code imports
  this legacy Python sorter. Session metadata recognizes Klustakwik file formats
  without using the package. Inspected local Phy plugins invoke the separate
  `~/klustakwik/KlustaKwik` executable, and installed Phy metadata does not require
  `klustakwik2`. No sorter code, binary, or parameters were modified.
- Updated README and the existing implementation plan; added an environment
  regression assertion against including this unused distribution.
- `/local/workdir/ys2375/miniforge3/envs/spn/bin/python -m pytest tests/test_setup_env.py -q`
  before removal: 1 failed, 6 passed; after removal: 7 passed.
- `git diff --check` passed; inspected final diff.
- The underlying wheel-build exception is absent from the supplied log, so its
  exact cause is unknown. Full fresh environment installation remains unverified.
  Users who independently need the Python klustakwik2 API must provision it
  separately; it is no longer installed by standard pipeline setup.
