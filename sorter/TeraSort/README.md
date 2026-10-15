# TeraSort

TeraSort is an experimental GPU spike-sorting toolkit for long, high-channel-count electrophysiology recordings, including recordings stored over Ethernet. Its default dashboard workflow is a **bounded session sorter**: learn templates from a short preview, then sort every recording core with custom CUDA kernels and restartable output shards.

The goal is fast processing with working memory that does not grow with the full recording duration. The current route uses INT16 signal and template storage, floating-point fitting and scores, buffered reads, and selective waveform caching. It is not an entirely integer pipeline.

**Status:** usable for experimental runs, but not yet a validated replacement for Kilosort4. The latest development benchmark has higher spike F1 and lower recall/unit recovery than KS4. The separate Kilosort-compatible route remains available for comparison and Phy output. A successful hundred-terabyte run has not been demonstrated.

## Quick start on Windows

Requirements: Windows x64, Git, an NVIDIA GPU with a compatible driver, and internet access for installation. The bounded dashboard route targets a 16 GB GPU and checks for at least 12 GiB free VRAM before starting.

```powershell
git clone https://github.com/zifangzhao/TeraSort.git
Set-Location TeraSort
.\install_and_start.bat
```

The installer finds a supported 64-bit Python 3.10-3.14, or falls back to its bundled Python 3.11 runtime. It creates or repairs `.venv`, installs the pinned Kilosort 4.1.7 and PyTorch 2.10.0 CUDA 12.8 environment, checks dependencies, and compiles a CUDA smoke-test kernel. Keep the complete repository together; the batch files are launchers, not standalone installers.

For later starts:

```powershell
.\start_server.bat
```

Open **http://127.0.0.1:8765/**. Keep the server console open. Closing the browser does not stop a sorting worker.

If Python detection fails, select an executable explicitly:

```powershell
.\install_and_start.bat -Python 'C:\path\to\python.exe'
```

Run `install_and_start.bat` again to repair missing or incompatible dependencies. To check the installation without starting the server:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-server.ps1 -CheckOnly
```

## Linux NVIDIA server

Use a 64-bit Linux system with a working NVIDIA driver (`nvidia-smi`), Git, internet access, and Python 3.10-3.14 with `venv`. PyTorch and CuPy supply runtime components; a separate system CUDA toolkit is not required by the installer.

```bash
git clone https://github.com/zifangzhao/TeraSort.git
cd TeraSort
chmod +x install_and_start.sh start_server.sh scripts/install-linux.sh
./install_and_start.sh --no-browser
```

Later, run `./start_server.sh --no-browser`. An explicit interpreter can be selected with `./install_and_start.sh --python /path/to/python --no-browser`.

For remote access, forward localhost through SSH:

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

Then open http://127.0.0.1:8765/ on your workstation. Direct network binding requires `TERASORT_WEB_TOKEN` and `--host 0.0.0.0`; HTTP is unencrypted, so prefer an SSH tunnel or TLS proxy. The native cuBLAS bridge can be built on Linux x86_64 with `python scripts/build_cublas_bridge_linux.py`; Linux quality and long-recording validation remain open.

## Run a recording from the dashboard

1. **Select recordings.** Browse for interleaved INT16 `.dat` or `.bin` files. Select multiple segments in acquisition order to sort them as one session. **Add all .dat recursively** adds files from subfolders; review the list and exclude auxiliary files such as digital inputs or timestamps before submission. Native Intan `.rhd`/`.rhs` and compressed `.cbin` require conversion.
2. **Check XML.** TeraSort looks for a matching XML or `amplifier.xml` beside the first recording. Use **Read XML** to inspect sample rate, channel count, groups, and excluded contacts. Apply one XML only to segments sharing the same acquisition configuration.
3. **Enter the verified gain in microvolts per ADC count.** Gain is required by the bounded dashboard route. The tested Intan recording uses `0.195`; do not assume that value for every acquisition system.
4. **Use Bounded session sorter**, selected by default. Leave probe overrides empty for this dashboard route. It generates the channel map from XML.
5. **Choose a new output folder.** The default is a dedicated folder beside the first recording. Choose a writable local or network destination with enough output space. Source recordings and XML are read-only inputs.
6. Click **Queue sorting run** and select the task to monitor it.

XML supplies channel order and grouping, not measured probe coordinates. Automatic geometry follows the MATLAB wrapper's staggered convention: alternating -20/+20 um horizontal offsets, 20 um vertical steps, and 200 um between groups. Verify that this matches your probe. For explicit physical geometry, multiple probes, day boundaries, or measured gaps, use the session manifest CLI described below. The dashboard shortcut does not infer those boundaries from folder names.

The file picker browses files on the **server machine**; it does not upload raw recordings. On Windows, mapped drives must be visible to the account running the server. If a drive is missing, try its UNC path and confirm that the same account can access the share. Running a server under a different/elevated account can change mapped-drive visibility.

## Monitor, retry, and clean the task list

| Control or display | Behavior |
| --- | --- |
| Bounded progress | Reports the source-sample position through the dense pass. Calibration and shard finalization are separate work; ETA is an estimate. |
| Kilosort progress | Shows preprocessing, sorting, and post-processing bars with phase ETAs when supported by the pinned hooks. |
| Resource readings | Worker RAM is process-related; system RAM and GPU utilization/memory include other applications. |
| Cancel | Stops the queued or running attempt. Completed bounded shards remain available. |
| Retry with current code | Starts a new worker using the installed code. Bounded jobs with a run journal resume the same output; Kilosort retries use fresh output paths. |
| Click output path | Opens the result folder in Explorer on a supported Windows server. |
| Remove item | Removes one completed, failed, or cancelled job from the task list. |
| Clear finished items | Removes all completed, failed, and cancelled list entries. Running and queued jobs are protected. |
| Clear log text | Hides existing text in the selected log view and displays subsequent output. |

Removing task-list items preserves result files, saved logs, manifests, and job metadata. Removed entries stay hidden after refresh and server restart. It does not reclaim recording/output disk space.

The queue defaults to one concurrent session. Although the concurrency control accepts 1-4, a bounded sorting job reserves the GPU queue slot and is not launched alongside another GPU sorting job. The dashboard stores state under `%USERPROFILE%\.terasort\web` on Windows or `~/.terasort/web` on Linux; `terasort web --state-dir ...` selects another directory.

Progress snapshots are advisory: temporary Windows file locks trigger short retries and then skip that display update, rather than crashing sorting. Checkpoint/output errors remain fatal so corrupted results are not silently accepted.

## Selectable bounded KS1 engine

The dashboard also offers **Bounded KS1 (experimental)**. It learns a fixed
bank from a preview capped at 1 GiB, then streams every source batch through
KS1-style CUDA template fitting and residual subtraction. Models and shards
are pinned for exact restart; Kilosort/Phy output includes measured PCA features.
The default TeraSort engine stays selected.

Fresh KS1 learning requires a separate original KS1 checkout, compatible CUDA
MEX files and MATLAB. The dense pass does not require MATLAB or Torch's runtime.
`terasort ks1-sort --manifest session.json --output-root new_run --kilosort-path C:\tools\KiloSort`
starts it; add `--resume` to continue. A single-probe/day native `rez.mat` can be
provided with `--model` to reuse a previously learned bank.

**Quality caveat:** the first bounded-learning 600-second test reached 70.55%
F1 and 90/250 strict recovered units, versus original KS1's 92.40% and 218/250.
Its isolated full wall time was 285.5 seconds, peak process-tree RAM 4.19 GB and
total VRAM 2,370 MiB. It is not a quality replacement. The reduced learning
sample is part of that time/quality tradeoff; the dense pass is not subsampled.
See [setup and operation](docs/bounded_ks1.md) and
[measured evidence](docs/evidence/bounded_ks1_20261001.md).

## How the default bounded algorithm works

1. **Learn a bounded seed bank.** For each missing probe/day seed bank, Kilosort4 components process three windows totaling a requested 60 seconds, distributed over the session/day. Existing manifest-provided seeds are reused. Preview scratch is capped at 5 GiB per learning pass; oversized previews fail rather than silently processing an unbounded input.
2. **Sort every core.** Custom CUDA kernels perform fixed-point filtering, candidate detection, local template fitting, and residual subtraction. New sessions use the selected INT16 profile, three primary passes, two rescue passes plus a targeted collision retry, and rolling template updates. Selected unknown waveforms use a fixed three-component temporal PCA basis learned from the seed bank; selected observations align to raw peaks within two samples, and proposals require separate raw and temporal holdout evidence before enrollment. An additional fixed spatial/temporal pass proposes weak coherent spikes from the original core and fits the remaining residual. Its assignments enter the output while template updates and enrollment retain their baseline observations. Dense sorting is not subsampled, and recovery adds no raw-source pass.
3. **Keep working sets bounded.** The dashboard uses cores up to two seconds, a channel-scaled read buffer and bounded prefetch. Recovery further shortens wide-probe cores so its two INT32 response planes, including preprocessing halos, fit a combined 256 MiB budget. INT16 arrays reduce signal/residual/template storage; fitting and amplitude estimates still use floating point. The dashboard caps the CuPy pool at 10 GiB; this is not a cap on all GPU allocations, including Torch seed learning.
4. **Write recoverable results.** New default runs use one-minute per-probe shards with checkpoints. Candidate metadata are retained; waveform payloads are selected under a default budget of 5% of source bytes. Total derived storage also includes metadata, assignments, and checkpoints, so it can exceed 5%.

Seed reports, archives, and a generated manifest live in a sibling `<output-root>.seeds` directory. Automatic seed learning removes temporary raw previews and intermediate Kilosort output after reconstructing the seed bank. Keep the seed directory together with the sorting output for resume.

Seed reconstruction targets 32,768 selected waveforms, with a fixed hard cap of 40,960 so every calibration identity can receive the required ten observations. A capacity bug that discarded every seed in banks of 3,277-4,096 identities is fixed. Larger calibration-window experiments improve public-example recall but regress on drift and individual units; the selected learning policy is unchanged. See the [capacity fix and full calibration comparisons](docs/evidence/seed_capacity_20260930.md).

Bounded dashboard jobs and `session-sort` automatically write common Kilosort1/4â€“Phy arrays to `phy/<probe>/<day>/` after sorting. Jobs are marked complete only after this export finishes. HDF5 assignment/candidate shards and metadata are retained. To convert an older completed session manually without rereading raw voltage:

```powershell
.\.venv\Scripts\terasort.exe session-export --source-root F:\sorts\session-1 --output-root F:\sorts\session-1\phy
```

Open `phy/<probe>/<day>/params.py` in Phy. New bounded sessions automatically capture measured per-spike PCA features during the dense pass and export them for Phy views, Mahalanobis outlier removal, and K-means reclustering. No second raw-data read is needed. Native Kilosort training state is not exported; see [export format and limitations](docs/kilosort_export.md). Cross-day identity tracking remains experimental; a linked identity is not proof that the same neuron was recorded across days.

### Session CLI

Use a versioned manifest with ordered source files, geometry, gain, sample rate, and explicit day/gap metadata. See the [manifest and output reference](docs/session_sort.md).

```powershell
.\.venv\Scripts\terasort.exe session-sort --manifest C:\data\session.json --output-root F:\sorts\session-1
```

Resume an interrupted run with the same manifest and settings:

```powershell
.\.venv\Scripts\terasort.exe session-sort --manifest C:\data\session.json --output-root F:\sorts\session-1 --resume
```

Completed shards are retained; an incomplete shard is reprocessed. Source/configuration mismatches reject resume. Older runs retain their original calibration path, and their original nondefault settings must still be supplied. Changing scientific settings requires a new output run rather than mixing algorithms within a checkpoint sequence. Linux uses `.venv/bin/terasort` in place of the Windows executable.

### KS4 scientific settings and preprocessed input

`session-sort --seed-settings path/to/ks4-settings.json` accepts native KS4
scientific overrides for initial seed learning. `--seed-backend` selects `auto`,
`standard`, `deep_tiled` (the unchanged default), or `cublas`. Source channel
count, sampling rate, gain, paths and common `--nt` remain manifest-owned.
Seed drift correction stays disabled (`nblocks=0`). Explicit overrides/backend
changes are recorded in the seed plan/report and reject incompatible resume.
Empty overrides retain the existing default seed identity. Overrides cannot be
applied to existing external seed templates or legacy runs without KS4 learning.

The full KS4 `sort` CLI now accepts `--input-is-preprocessed`; the Python API
uses `run_kilosort(..., input_is_preprocessed=True)`. This reuses the pinned
KS4 4.1.7 identity high-pass guard, disables CAR and retains whitening/learning.
Do not provide `highpass_cutoff` for preprocessed input. LFP export requires raw
input and cannot be combined with this flag. Raw callers preserve their existing
filtering/reference behavior.

Single-session KS4 results write `terasort_settings.json` with effective native
settings, requested/selected backend and input preprocessing flags. Bounded seed
reports retain this record after the temporary KS4 preview directory is removed.
Nonfinite native limits are represented as strings in this metadata only.
The dedicated cuBLAS specialization requires `n_templates=6` and `nt=41` or
`nt=61`. Linux requires a locally built native bridge; measured speed and
recovery evidence currently covers the Windows `nt=61` route only.

Original `ks1-sort` remains a separate experimental original-KS1 CUDA/calibration
route. It rejects explicit source-column/preprocessed manifests. The `fast`
preset changes the bounded fitting/admission/cache policy; it is not a third
interchangeable KS4 sorting mode.

## Kilosort-compatible comparison route

Select **Kilosort4 (quality baseline)** explicitly in the dashboard when you need the whole-recording route and standard Kilosort/Phy output. This route has recording-sized feature/clustering allocations; its CUDA optimizations do not make it memory-bounded for arbitrarily long recordings. The dashboard requires an explicit override above 128 GiB of input; smaller input is not a guarantee against memory exhaustion.

Existing Python scripts can use:

```python
from terasort import run_kilosort

result = run_kilosort(
    settings={"n_chan_bin": 384, "fs": 30000, "scale": 0.05},
    probe=probe,  # acquisition-matched geometry/channel map
    filename=r"F:\data\recording.bin",
    results_dir=r"F:\data\kilosort_result",
    data_dtype="int16",
)
```

This API returns Kilosort's nine-value tuple and applies source-guarded optimizations without editing the installed Kilosort package. Use `kilosort.run_kilosort` directly for an unmodified baseline. The CLI equivalent is `terasort sort`; it remains distinct from `terasort session-sort`. Repeating `--filename` (or passing a Python filename list) provides a virtual multi-file session without creating a concatenated binary.

The Kilosort route offers local SSD read-ahead for network sources, with local sources bypassing that cache. Its cache, full-staging, and combined LFP options are **not used by the bounded dashboard route**, which has its own streaming reader. A separate LFP export is available:

```powershell
.\.venv\Scripts\terasort.exe lfp --filename F:\data\recording.bin --output F:\data\recording_lfp.i16 --sample-rate 30000 --n-channels 384 --scale-uv-per-count 0.05 --passband-hz 500
```

This produces 1,250 Hz INT16 LFP with anti-alias filtering and a metadata sidecar. It reads the raw source separately, takes additional computation/I/O, and supports `--resume`.

## Current benchmark and limitations

Recorded complete-pipeline development benchmark (2026-10-01), extended with original Kilosort1 on the same recording: 600 seconds, 384 channels, 32 kHz INT16, 250 ground-truth neurons, RTX 5060 Ti 16 GB. New experimental sessions include bounded raw-peak alignment for enrollment and frozen-learning spatial/temporal recovery. TeraSort learned new seeds from a bounded three-window 60-second KS4 preview, sorted every core, and exported Kilosort/Phy arrays. All three outputs were scored without manual curation, including all clusters and resolved TeraSort template links.

| Metric | Bounded TeraSort, recorded run | Original Kilosort1 | Kilosort4 4.1.7 reference |
| --- | ---: | ---: | ---: |
| Spike precision | 83.52% | **92.92%** | 70.14% |
| Spike recall | 92.42% | 91.89% | **98.22%** |
| Spike F1 | 87.74% | **92.40%** | 81.84% |
| Units recovered at IoU >= 0.8 | 200 / 250 | 218 / 250 | **231 / 250** |
| Dense sorting time | 184.4 s | Final matching: 76.6 s | Not separately matched |
| Full processing time, including learning/export | **280.4 s** | 475.7 s | 742.4 s |
| Observed process-tree RAM, complete run | **3.27 GB** | 10.14 GB | 3.46 GB |
| Observed total device VRAM, external sampler | **4,081 MiB** | 6,618 MiB | 6,467 MiB |

TeraSort's recorded full-route wall time is **1.70x faster than KS1** and **2.65x faster than KS4**. Original KS1 uses the official repository configuration with 768 templates and six learning passes; it has higher precision/F1 and strict-unit recovery than TeraSort on this example. KS4 has the highest recall and strict-unit recovery. SpikeInterface 0.103.0 independently reproduces all three scores.

KS1 is a new run; TeraSort and KS4 retain their prior same-input/geometry/GPU measurements. The TeraSort timing predates automatic measured per-spike PCA export; a separate 32-second test measured about 7% dense-pass overhead without changing sorting arrays. This is not a fresh three-way release timing or Ethernet benchmark. KS1 final matching excludes 241.8 seconds of whole-recording template learning; it is not a matched standalone speed comparison with TeraSort dense sorting. The table uses the common external VRAM sampler; TeraSort's internal telemetry recorded a slightly higher 4,138.6 MiB peak. See the [KS1 configuration, timing breakdown, validation and reproduction](docs/evidence/ks1_comparison_20261001.md).

Peaks are observed values, not universal limits. The exact-bank control disables only peak alignment: the change adds 2,113 true spikes and one strict unit, raises recall from 92.21% to 92.42% and F1 from 87.66% to 87.74%, and reduces precision from 83.53% to 83.52%. SpikeInterface 0.103.0 independently reproduces both paired counts. One individual unit loses more than five precision points; none loses more than five recall or F1 points. Fresh calibration varies, so previous runs with different seeds are not causal comparisons. No cold-cache or Ethernet result is claimed.

Smoothed detections can differ from the raw peak by one or two samples. Enrollment now re-extracts only the selected bounded snippet at the raw peak, retaining core ownership, collision guards, original SNR, and raw holdout validation. It adds no source read or full-core FP32 conversion. Existing resumes retain their recorded rule. The earlier CUDA scorer, host optimizations, INT16 arrays and response-plane cap remain active. See the [alignment quality, memory and restart evidence](docs/evidence/raw_peak_enrollment_20261001.md), [spatial/temporal recovery evidence](docs/evidence/integrated_spatial_recovery_20260930.md), and [CUDA/host optimization evidence](docs/evidence/cuda_host_recovery_20260930.md).

Higher F1 does not mean equal neuron recovery: TeraSort still has lower recall and recovers 31 fewer strict units here. The public example has been used in development. Additional drift/collision and rare-unit checks are synthetic. Probe/chunk metadata for two independent public drift recordings are downloaded, but paired sorting comparisons remain pending. Completed long network recordings, 1x/10x/100x duration tests and full storage accounting also remain required. Neither arbitrary-file-size RAM/VRAM guarantees nor a hundred-terabyte capability claim is established.

New runs use template-conditioned contact scoring: weak peripheral contacts are excluded from the shape cosine, while full-contact amplitude fitting, residual gain and subtraction are retained. With identical retained seeds, this raised public250 recall from 87.53% to 89.83% and F1 from 86.05% to 86.87% at similar dense-pass time. A fresh drift/collision test also gained recall and F1, with lower precision. See the [contact-scoring and complete-pipeline evidence](docs/evidence/template_contact_scoring_20260930.md). The profile also includes [targeted collision recovery](docs/evidence/fast_recall_recovery_20260929.md); resumes preserve their recorded fitting policy.

The selected proposal policy admits unknowns down to the existing 4.5-SNR detector threshold, preserves strong sparse proposals for the existing 30-minute horizon, and validates PCA templates against unprojected waveform holdouts. It removes false noise enrollment in a synthetic sparse-unit diagnostic while retaining the fixed bank budgets and CUDA/INT16 dense path. Retrospective fitting now recovers all 28 rare-unit spikes in that synthetic diagnostic, including the 24 learning observations, without false positives or an additional raw-source read. It retains selected INT16 snippets in the fixed-size proposal bank and publishes recovery rows alongside enrollment, preserving earlier immutable shards. The default native exporter includes these events; old checkpoints retain their old policy. See the [retrospective recovery evidence](docs/evidence/retrospective_recovery_20260930.md). Matched-seed public recall has a small tradeoff, detailed alongside the fresh complete-pipeline comparison in the [sparse retention and raw validation evidence](docs/evidence/sparse_raw_holdout_20260930.md).

Linked templates now share refractory histories on CPU and GPU. The separate
CUDA joint-pair fitting helper also supports direct scaled INT16 storage. Automatic
joint recovery remains experimental because its small recall gains did not justify
the extra runtime and false assignments on the independent drift test. A fresh full
run retained 91.48% recall, 87.73% F1 and 197/250 strict units at 2.91x the KS4
reference speed. See the [joint fitting and consistency evidence](docs/evidence/joint_fit_refinement_20260930.md).

A supplementary shared-memory CUDA detector is being tested without changing
the selected profile. Its corrected full public result reaches 92.50% recall,
88.06% F1 and 202/250 strict units, with more dense computation. Drift regressions
keep it experimental. The [temporal detection study](docs/evidence/temporal_candidates_20260930.md)
includes the controlled timers, fixed memory budgets, crash/restart checks and
a new complete comparison with KS4 on generated drift.


PCA enrollment previously improved recall and F1 on both the complete public example and a separate drift/collision recording. The same-seed public comparison recovered 10,112 additional correctly assigned spikes; a fresh complete run enrolled seven proposals. New runs use this fixed policy, while resumes retain their recorded enrollment and feature policy. See the [PCA enrollment evidence](docs/evidence/pca_enrollment_20260930.md), [current session results](docs/session_sort.md), [earlier full comparison and scoring protocol](docs/evidence/public250_full_int16_vs_ks4_20260928.md), and [candidate coverage/fitting diagnostic](docs/evidence/public250_filter_scaling_and_fit_tuning_20260928.md). Candidate coverage is not interchangeable with final spike recall.

## Update and troubleshoot

```powershell
git pull --ff-only
.\install_and_start.bat
```

Update between runs when practical. New workers use current installed code, but an already-running worker does not acquire algorithm changes. Refresh the browser for HTML changes. Python server changes require restarting the dashboard; `start_server.bat` reuses an already-running server rather than replacing it. Active independent workers can be rediscovered after a server restart; keep their job-state directory intact.

- **Missing drive or access denied:** verify permissions and drive mappings under the server's Windows account; try the UNC path.
- **Resume rejected:** retain the original source, settings, output, and sibling seed directory. Do not delete checkpoints to force a resume.
- **No GPU readings:** check `nvidia-smi`. Device-wide utilization includes other programs.
- **Unexpected memory growth:** confirm the task says bounded mode. Kilosort `sort` and the compatibility API still have full-recording allocations.
- **Progress text stops changing:** check the job's state and source counter; the seed-learning log can remain unchanged during dense sorting.

## Development and references

Install test dependencies and run the Windows test wrapper:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
powershell -File scripts/test.ps1
node --test tests/web_picker.test.cjs
```

The wrapper uses a project-local temporary directory. CUDA tests require the GPU environment and a writable CuPy compiler cache. Python and browser tests verify contracts and recovery behavior; they do not replace ground-truth sorting benchmarks.

- [Session sorter, manifests, outputs, and research details](docs/session_sort.md)
- [Candidate-bank format](docs/spike_candidate_bank.md) and [cached waveforms](docs/spike_candidate_waveforms.md)
- [Earlier experimental streaming sorter](docs/streaming_sort.md)
- [Kilosort INT16 reader](docs/kilosort_int16.md) and [native cuBLAS adapter](docs/kilosort_cublas.md)

TeraSort is distributed under **GPL-3.0-only**; see [LICENSE](LICENSE). Its Kilosort adapters include transformations of Kilosort 4.1.7 code. Cite the upstream [Kilosort project](https://github.com/MouseLand/Kilosort) when using its components, and identify the TeraSort commit and processing settings in reported experiments.
