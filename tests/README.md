# Project tests

Run commands from the repository root using the existing `preprocess` environment
with pytest installed. Tests import the checkout's `src` package, so use
`python -m pytest`. No runtime source changes are needed to select a test group.

Default pytest discovery is limited to `tests/` by the root `pytest.ini`.
Vendored tests under `external/` and `sorter/` are outside the project suite.
MATLAB tests are run separately.

## Groups

| Path | Existing coverage | Execution requirements |
| --- | --- | --- |
| `tests/cli/` | Pipeline/sorter CLI arguments, mode dispatch, JSON results/errors, channel identity and mixed acquisition inputs | Routine cases bypass sorting; two `integration` cases use real preprocessing, including Phy export or mixed Intan/OE/WILD binaries |
| `tests/preprocess/` | Input validation, disk budgeting, multi-day preparation, sorter partitions/channel mapping, state-scoring memory | Pipeline Python dependencies; synthetic inputs and existing test doubles |
| `tests/execution/` | Local backend lifecycle/identity, session claims, GPU selection, worker allocator setup | Some tests start subprocesses; Windows/POSIX tests retain their existing skip conditions; GPU queries are mocked |
| `tests/gui/` | GUI settings, XML/probe selection, run setup, output transfer and recovery | PySide6 and the pipeline dependencies; use Qt's offscreen platform |
| `tests/setup/` | Conda/uv setup configuration, vendored dependency selection and environment safety | PyYAML and pytest; setup/install calls are mocked; one test starts a Python subprocess |
| `tests/stability/` | Channel MAD computation, filtering, plots and multi-day channel metadata | NumPy, SciPy, pandas, Matplotlib and pipeline dependencies |
| `tests/matlab/` | MergePoints folder naming/path resolution and waveform preservation | MATLAB and the bundled MATLAB/CellExplorer helpers |

These are existing regression tests. They do not establish real-data pipeline,
GPU sorter, Slurm cluster, or interactive GUI correctness.

Useful regression cases are retained even when they need an optional environment.
The POSIX permission-bit check is skipped on Windows because it does not test
Windows ACLs. Symlink tests require permission to create symlinks.

## GitHub Actions

`.github/workflows/basic-tests.yml` runs Python 3.11 jobs on Ubuntu 24.04 and
Windows, named `Linux basic checks` and `Windows basic checks`, on pull requests
and pushes to `main`. Both jobs select the same cases:

- `tests/cli`, including the two small channel/Phy and mixed-acquisition cases.
- `tests/setup`, for environment configuration and setup safety.
- `tests/preprocess/test_intan_validation.py`, for malformed input rejection and
  preservation of input data.
- `tests/preprocess/test_disk_space.py`, for disk budgets and early rejection.
- `tests/preprocess/test_kilosort_partition_channels.py`, including native Phy
  shank/probe metadata alignment with simulated sorter output; MATLAB/GPU sorting
  is not launched. CLI export tests also check shank labels with external and
  copied binaries, reordered channels, and missing geometry.
- `tests/execution/test_input_identity.py`, for input identity and resume checks.
- `tests/execution/test_progress.py` and two named persistent GUI progress cases,
  for all three stages' progress settings, tiny binary writes with one/two workers,
  and readable incremental stdout/stderr in the main Log. Backend selection is
  simulated; no Slurm or GPU job is launched.
- Three named output-transfer tests: recovery paths/custom metadata/progress,
  copy-failure preservation, and equal-size copy-corruption rejection.

These checks run for incoming contributors' PRs as well as the maintainer's PRs.
Each job installs the project/dev dependencies from `pyproject.toml`, keeping
NumPy at the shared Linux/Windows environment baseline of 1.26.4. Pip downloads
are cached. A failure on one OS does not cancel the other job.
The project constrains Neo to `>=0.14.3,<0.14.5` because SpikeInterface 0.103.2
passes an Open Ephys reader argument removed in Neo 0.14.5. The Windows Conda
environment pins Neo 0.14.4; the existing Linux pin is 0.14.3.

Linux installs `libegl1`, required when importing Qt GUI modules even with Qt's
offscreen platform. Pytest cache is disabled; pytest and Numba use
separate subdirectories of a unique OS temporary directory removed after exit.
Each job has a 15-minute limit and its test step a five-minute limit. These jobs
do not run the full suite, GPU sorting, MATLAB, real sessions or interactive GUI
checks. Hosted Ubuntu coverage checks basic portability; it does not reproduce
the full Linux server environment or execute Slurm jobs.
See the PR Checks tab or the repository Actions tab for results. Manual dispatch
is available after this workflow is present on the default branch.

## Targeted Python execution

Choose a group, file, or named test deliberately. For example, the existing CLI
group is selected with `tests/cli`, and a single test with
`tests/cli/test_sorter_runner.py::test_run_sorter_cli_passes_active_and_excluded_channels`.
The GUI group is now selected with `tests/gui`, and environment setup with
`tests/setup`; sorter partition tests remain in `tests/preprocess`.

For a short routine check, select the existing CLI/setup tests and small input
validation tests together:

```text
python -m pytest tests/cli tests/setup tests/preprocess/test_intan_validation.py tests/preprocess/test_disk_space.py tests/execution/test_input_identity.py -m "not integration" -q -p no:cacheprovider --basetemp <unique-OS-temp-directory>
```

This selection does not launch a GUI, sorter, worker job, or Slurm job. The setup
tests include one short Python subprocess. The earlier version of this selection
ran 51 cases in 4.25 seconds; that result predates the added CLI cases. The latest
validation selection is recorded below. GUI, process lifecycle, scientific/plotting,
and MATLAB groups remain separately selectable. No exhaustive test matrix is added.

Keep pytest cache disabled and use a fresh temporary directory outside the
repository. Do not pass the temporary root itself as `--basetemp`.

PowerShell example (after activating `preprocess`):

```powershell
$testTemp = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString('N'))
$previousNumbaCache = $env:NUMBA_CACHE_DIR
$env:NUMBA_CACHE_DIR = Join-Path $testTemp 'n'
New-Item -ItemType Directory -Path $testTemp | Out-Null
try {
    python -m pytest tests/cli -m "not integration" -q -p no:cacheprovider --basetemp (Join-Path $testTemp 't')
    $testExitCode = $LASTEXITCODE
} finally {
    if ($null -eq $previousNumbaCache) {
        Remove-Item Env:NUMBA_CACHE_DIR -ErrorAction SilentlyContinue
    } else {
        $env:NUMBA_CACHE_DIR = $previousNumbaCache
    }
    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    $resolvedTemp = [System.IO.Path]::GetFullPath($testTemp)
    if (-not $resolvedTemp.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected temporary directory"
    }
    if (Test-Path -LiteralPath $resolvedTemp) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }
}
if ($testExitCode -ne 0) { throw "pytest failed with exit code $testExitCode" }
```

For GUI tests, set `$env:QT_QPA_PLATFORM = 'offscreen'` before the command.

POSIX example:

```bash
test_temp=$(mktemp -d "${TMPDIR:-/tmp}/preprocess-tests.XXXXXXXX")
python -m pytest tests/cli -m "not integration" -q -p no:cacheprovider --basetemp "$test_temp"
test_exit_code=$?
rm -rf -- "$test_temp"
(exit "$test_exit_code")
```

For GUI tests, prefix the Python command with `QT_QPA_PLATFORM=offscreen`.
Replace `tests/cli` with the selected group, file, or named test. Only select
`tests` for a full project run when that broader validation is intended.

## CLI coverage and optional launch checks

`tests/cli/test_sorter_runner.py` preserves the two existing tests:

- Explicit `--active-channels` and `--exclude-channels` values reach
  `execute_sorting_job` as zero-based integer lists.
- Omitting these options forwards `None` for both lists.

They call `build_parser()` and `run_sorter_cli()` directly and replace sorter
execution with a monkeypatch. They do not launch the module as a subprocess or
run a sorter. Subprocess import tests in `tests/execution/` and `tests/gui/`
test import behavior, rather than CLI parsing or full pipeline execution.

The routine cases check a missing XML before sorting, an all-excluded sorter
no-work result, a bad-channel ID beyond the binary width, invalid module CLI mode,
malformed configuration, and stage dispatch for `preprocess`, `postprocess` and
`noise_label`. They check exit codes and parse result/error JSON; dispatch cases
replace stage execution and verify the noise-label flag without creating outputs.

Select the cross-stage channel check separately with
`python -m pytest tests/cli -m integration` and the same cache/temp arguments.
One eight-channel synthetic recording combines manual exclusion, XML skip and
omission from spike groups. The check follows these contracts:

- Preprocessing retains all binary columns and zeroes bad columns; postprocessing
  selects good channels by their original zero-based IDs.
- Raw-reference Phy output keeps the full binary width and original column IDs.
  Copied-binary Phy output uses compact indices, while `channel_map_si.npy`
  preserves the original IDs. Both mappings address the correct binary samples.
- The CellExplorer input `session.mat` uses one-based bad/electrode channels and
  the full binary width; `chanCoords` rows agree with the corresponding chanMap.
- CellExplorer sampling rate, dtype, sample count and duration match the binary
  and Phy metadata. The raw and copied Phy binaries preserve sample count/timebase.
- Postprocess-only CLI execution respects a selected probe partition without
  treating original IDs as compact indices; source input samples remain intact.

This case uses real preprocessing, postprocess recording resolution, a small
analyzer and Phy export. It replaces scientific curation and disables exporter
PCA/amplitude computation and display-template re-estimation. MATLAB CellExplorer,
GUI interaction, GPU sorting and scientific feature validity are not verified.
The second integration case uses one Intan, one OE and one merged WILD epoch.
It checks selected epoch order, ephys samples excluding the embedded OE ADC,
bad-column zeroing, MergePoints boundaries, CellExplorer metadata, and analog
normalization to WILD's 1250 Hz rate. Intan/OE analog identities occupy the same
output columns, with absent ADC columns zero-filled. Source files remain unchanged.
Small rejection cases cover OE/WILD rates that disagree with the common XML
timebase and mismatched ephys channel counts before binary exports.

Synthetic Intan fixtures without local RHD metadata exercise the implementation's
documented warning and positional ADC identity path. The existing named Intan
validation check covers a local RHD header separately. Mixed recording tests do
not validate WILD hardware merging, TTL/camera synchronization, multi-day symlink
staging, calibration, or actual MATLAB loading.

The integration marker is registered but not excluded globally: choose the routine
or integration command deliberately. A captured stack from a stalled run identified
Numba cache-file creation during postprocess import. The PowerShell example uses
a short unique temporary root with separate Numba/pytest subdirectories and restores
the previous cache setting. This avoids writing caches into the installed environment;
it leaves JIT execution enabled and removes both temporary subdirectories afterward.

Latest focused validation on Windows selected the 15 CLI cases plus five existing
cases: `test_valid_recordings_and_packed_digital_lines`, the three parameterized
`test_wild_catalog_uses_manifest_analog_width_and_1250_hz` cases, and
`test_analog_concat_downsamples_non_wild_epochs_to_wild_rate`. The combined run
finished with 19 passed and one fixture failure in 18.18 seconds. XML anatomical
groups do not imply separate probes: after explicitly assigning two probes in
the partition fixture, the affected named test passed in 14.61 seconds (12.74
seconds in the test body). All 20 distinct selected cases have passing results
across these runs; no complete-suite result is claimed. The mixed-input normal
case took 0.53 seconds. Earlier fixture naming/order and Windows path-length
errors were corrected without changing data dimensions, expected mappings or
runtime source. These CLI results predate the source follow-up below; they are
not a rerun of the CLI group after that follow-up. Full GUI and MATLAB execution
remain unverified.

## Windows source follow-up

The user subsequently authorized source fixes. JSON metadata relocation now
decodes JSON and recursively rewrites its string keys/values, so Windows
backslash escaping does not leave references to the old local output path.
The existing named regression
`tests/gui/test_gui_move_outputs.py::test_move_rewrites_recovery_paths_and_copies_custom_destination_metadata`
passed on Windows in 2.85 seconds (0.11 seconds in its body), with offscreen Qt,
disabled pytest cache and an isolated temporary directory. It exercises filesystem
transfer and recovery metadata, not an interactive GUI session.

A separate user's Windows concatenated-dat stall has no available last log,
input format or storage details. The writer now prints flushed `writing`,
`validating`, `publishing` and `complete` messages, with failure-stage exception notes and the
output/partial paths, expected binary dimensions/bytes, requested workers/pool
engine and installed SpikeInterface version. Numerical processing, worker defaults,
pool selection and atomic publication are unchanged. This diagnostic-only writer
change was inspected; no new multiprocessing or full-pipeline run was performed.
The stall's cause and resolution in the other user's environment remain unknown.

For the next occurrence, retain the last phase message and SpikeInterface's own
progress output. The upstream writer preallocates the partial file before chunk
processing: a full-size partial file alone does not establish completed writing.
If a comparison run is needed, select one preprocessing worker in the GUI and
use a separate empty output location; retain the original inputs and outputs.
That comparison has not been executed here. The usual OE layout with ADCs after
the ephys channels passed the earlier mixed-recording case; interleaved-ADC reader
changes are deferred and no OE runtime source is changed in this follow-up.

When a launch/import check is needed, choose an entry point explicitly:

```powershell
python -m src.preprocess.sorter_runner --help
python -m src.execution.controller --help
python -m src.execution.worker --help
```

The installed console equivalents for the latter two are
`preprocess-run-controller --help` and `preprocess-stage-worker --help`.
Help checks confirm imports and parser startup only. They do not test controller
actions, worker execution, or sorting. The existing suite has no dedicated
controller/worker CLI argument tests; adding coverage is separate work.

Real execution uses data/output paths and can write results, submit jobs, retry
stages or cancel active work. Define the input, destination, command and expected
result separately before using those actions as an end-to-end test.

## MATLAB

From the repository root in MATLAB:

```matlab
results = runtests(fullfile('tests', 'matlab', 'test_mergepoints_foldernames.m'));
assertSuccess(results);
results = runtests(fullfile('tests', 'matlab', 'test_kilosort1_waveforms.m'));
assertSuccess(results);
```

The existing test adds its bundled helper paths temporarily and restores the
MATLAB path at teardown. Its synthetic files live in a MATLAB temporary folder.

The KiloSort1 tests run on CPU with tiny synthetic batches. They verify unchanged
historical 33/61-sample projections and explicit errors when initialization,
completed template fitting, or final matching detects no spikes. They do not
exercise CUDA MEX binaries or full-session sorting. A proposed alignment change
and first-pass abort were withdrawn after review of the legacy behavior.
