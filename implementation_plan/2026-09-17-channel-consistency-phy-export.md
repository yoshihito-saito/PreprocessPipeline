# Channel consistency and Phy export

Date: 2026-09-17. Branch: `bug/channel-consistency-phy-export`.

## Goal and observed failures

Preserve binary column identity, XML anatomy, bad-channel exclusions and sample
time across single-session preprocessing, sorting, standalone postprocessing,
notebook/API entrypoints and Phy. Support both 20 kHz and 30 kHz without resampling.

The read-only investigation reproduced a standalone GUI selecting a different
same-named binary than the selected sorting, yielding all-zero mean waveforms
with unchanged spike intervals. Other confirmed failures include inconsistent
map precedence, bad IDs dropped after partial-map attachment, all-bad fallback
to every channel, compact KS2.5 channel indices redirected to a full binary,
KS4 shank grouping loss, and anatomical CSV ordering different from XML.
Real SI/Phylib round trips also exposed asymmetric template versus centered raw
waveform windows, microvolt templates versus ADC raw waveforms, and broken
paths/maps for copied compact binary exports.

## Required behavior

- Keep the original full binary column order; existing preprocessing may zero bad
  columns. Standalone analysis may select good channels while retaining their
  original IDs. Do not require writing another preprocessed binary.
- Accept external sorting plus its corresponding raw or unfiltered concatenated
  recording. Matching means the same sample timeline and channel correspondence,
  not byte equality between raw and preprocessed signals. Make explicit input
  selection possible; never silently substitute an unrelated same-named file.
- Resolve XML, chanMap, manual exclusions and recording layout consistently.
  Validate invalid/duplicate/misaligned map entries rather than losing masks.
- Process good channels only for reference/filtering, once when requested.
  Preserve existing scientific filter parameters and curation/metric semantics.
- Skip deliberately all-bad probes normally, with a recorded reason. If every
  probe is bad, sorting/postprocess finish as skipped and must not reuse stale
  outputs. Invalid metadata remains an error.
- Bind Phy path, dtype, binary width, sampling rate, offset and channel map to
  the binary actually referenced. Compact copied binaries use compact indices;
  full external binaries use original column IDs.
- Preserve analysis windows (1 ms before, 2 ms after), microvolt metrics and
  original spike samples. Reconcile Phy display template origin/units separately
  using a centered display window and binary-compatible units; do not shift
  spike timestamps or merely pad missing waveform samples.
- Native KS2.5 needs explicit compact-to-recording-to-original column mapping
  and compatible signal/whitening treatment. KS4 receives distinct probe/shank
  identities without changing artifact-reference grouping elsewhere.
- Anatomical CSV follows XML electrode-group/channel ordering, including bad
  channel positions. Repeated shank numbers across probes must remain distinct;
  unsupported nonempty CSV cells must not be silently discarded or overwritten.

## Files and API scope

Primary modules: `src/postprocess/pipeline.py`, `metafile.py`, preprocessing
`recording.py`, `sorter_runner.py`, `sorting_stage.py`, GUI `config_model.py`,
`preflight.py`, `app.py`, `anatomical_map.py`, and `src/execution/worker.py` plus
direct resume/validation consumers as needed. Use small shared metadata helpers
only where multiple entrypoints actually need the same operation.

Keep public defaults and established parameter meanings. Add optional explicit
postprocess input fields where needed and document supplied-recording ownership.
Update notebook-facing helpers/docs rather than duplicating logic in notebooks.
Related prior plan: [partition exclusions](2026-09-10-kilosort-partition-channels.md).

## Final implementation decisions

- Replace the GUI's hardcoded eight-group default with an empty automatic
  assignment that derives all XML groups. Previously saved explicit assignments
  retain their meaning. Loaded map exclusions survive geometry edits, while
  anatomical labels are bound to a session/XML identity.
- Full preprocessing rejects partial maps that would shrink saved binary columns.
  Analysis/sorting views may use partial maps and retain original column IDs.
  Missing explicit maps and invalid requested IDs fail before sorting.
- Preserve SI analysis windows and features. For Phy display only, let the
  exported template have N samples, B=floor(N/2) samples before each unchanged
  spike, and N-B samples after. Average the same selected spikes in native ADC
  units, including Phylib-compatible clipped/zero-padded boundary windows.
  SI amplitudes remain signed physical amplitudes; they are not converted into
  Kilosort fitting coefficients.
- Restore native KS2.5 W=Wrot/scaleproc and input-binary channel indices. If the
  fit is xW≈aT, the displayed unwhitened template is TW⁻¹, preserving a and spike
  times. Native drift-corrected templates remain at their reference position.
  A marker in the bundled exporter prevents silently redirecting incompatible
  custom exports. The skip-preprocessing branch uses compact active-channel
  data so its Nchan read stride is correct. Native params retain binary dtype,
  including byte order.
- Cache reuse requires provenance for the lazy recording graph, properties,
  source binary path/size/mtime and source sorting/label hashes. Older or
  mismatched explicit caches fail before labels are modified; they need a rebuild.
- An existing manifest never authorizes fallback to unrelated older sorting
  folders. Valid all-bad manifests are normal no-work results through execution
  and resume. Low-level sorter execution still requires a nonempty channel set;
  session/stage orchestration and the direct CLI implement normal skip behavior.
- Automatic multiple targets share a recording timeline. Independent external
  recordings require separate calls; no sample shift is inferred or repaired.

## Follow-up: direct CLI all-bad skip

The user confirmed that heterogeneous recording inputs are outside this workflow
and requested only CLI all-bad handling. Keep that additional input policy out of
scope and continue on the same bug branch, preserving already staged edits.

- Resolve the channel selection before MATLAB/sorter dependency setup. Signal a
  valid empty selection with the existing `NoActiveChannels` ValueError subclass.
- The CLI catches only that condition, prints `no_active_channels`, and returns
  normally (exit status 0). It does not initialize a sorter, create a completed
  output, remove existing output, or overwrite a session partition manifest.
- Keep invalid maps/channel IDs/counts/rates and unrelated execution failures as
  errors. Keep ordinary CLI argument forwarding and the low-level Path return
  contract unchanged.
- Verify map-only exclusions, manual exclusions, their union, and a fully excluded
  active-channel subset for KS1/KS2.5/KS4 at 20/30 kHz in phy2. Use real CLI
  subprocesses for successful exit checks, spies for dependency isolation, and
  negative/partial-selection regressions. No MATLAB/GPU sorting is required to
  verify a no-work result.

## Implementation sequence

1. Shared input/channel validation and GUI/XML/anatomy alignment.
2. Consistent bad exclusions, normal empty-probe handling and skip propagation.
3. Optional preprocessing and notebook/API consistency; common Phy export fixes.
4. Sorter-specific native export and grouping corrections.
5. Independent numerical review, regression verification, documentation/change log.

## Verification

Use `/local/workdir/ys2375/miniforge3/envs/phy2/bin/python` and temporary synthetic
data only. Do not touch active jobs, live session data or existing sorter outputs.

- Regression tests for wrong binary resolution, explicit versus adjacent maps,
  nonmonotonic/partial channel IDs, XML skips, full versus partition exclusions,
  single/all probes bad and stale-output avoidance.
- Real SI/Phylib 20/30 kHz round trips with interleaved bad IDs, nonmonotonic maps,
  full and compact binaries, relative paths, nonunit gain and optional filtering.
  Assert exact channel/sample correspondence, unchanged source bytes/spike times,
  correct display origin and physically justified floating-point tolerances.
- Check original analysis window and metric units remain unchanged after export.
- Native sorter adapter/export tests without expensive sorting, with explicit
  limits if MATLAB/GPU execution is unavailable.
- Anatomical CSV round trip against nonmonotonic XML and repeated shank IDs.
- Relevant existing preprocessing/postprocess/GUI/execution tests and final diff.

Known baseline: selected suite previously had 101 passes and five unrelated
failures (three KS4 configuration/import expectations and two CMR preflight
expectations). Whole postprocess test collection imports a missing
`src.postprocess.ks4_diagnostics` module. Record baseline limitations accurately;
do not weaken or skip assertions just to claim a clean suite.

## Non-goals

No resampling, spike-time adjustment, curation threshold changes, mandatory
processed-binary storage, unrelated cleanup, live sorting execution or rewriting
existing scientific outputs. Multi-day behavior must not regress, but the reported
failure and primary verification target single-day sessions.
