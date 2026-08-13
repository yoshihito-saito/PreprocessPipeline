# Standalone multi-day channel MAD analysis

- Date: 2026-08-12
- Commit: uncommitted
- Implementation plan: [2026-08-12-standalone-multiday-mad.md](../implementation_plan/2026-08-12-standalone-multiday-mad.md)
- Literature/design discussion: `paperflow/multiday-neuron-tracking-electrophysiology/proposal.md` (local ignored research artifact)

## What changed and why

- Added the standalone `src.stability` package.
- Added `analyze_multi_day_mad()` to read a staged
  `multi_day_selected_subepochs.csv`, calculate centered MAD-derived channel noise
  before and after a configurable bandpass, and save `multi_day_mad.csv` and
  `multi_day_mad.png` by default.
- Added lower-level computation and plotting APIs for notebooks and scripts.
- Kept the module separate from the preprocess pipeline, GUI, execution stages, and
  bad-channel metadata.
- Added source-provenance validation for Intan and Open Ephys input, including
  Open Ephys selection of ephys channels without ADC channels.
- Rejects inconsistent ordered physical channel-name maps across selected
  subepochs so heatmap rows are not silently misaligned.
- Stores source dtype, per-channel gain/offset, and scaling source in the output.
- Reports an explicit filter-length error for recordings/windows too short for the
  zero-phase bandpass.
- Used float32 bandpass output so integer rounding does not alter the post-filter MAD.
- Added focused synthetic tests and a README usage example.
- Added `plot_multi_day_mad_by_channel()` to overlay color-coded channel traces in
  separate pre-filter and post-bandpass panels with a channel-index colorbar.
- Added a standalone MAD-profile mapping check that ranks plausible transformations
  independently for each complete 64-channel Intan headstage block across adjacent
  selected subepochs. It records
  identity similarity, profile discontinuity, pre/post permutation similarities,
  conservative consensus gain, support count, and a priority rank that requires
  both positive gain and positive corrected similarity, without applying an
  automatic swap decision.
- Uninformative constant blocks are skipped locally, unsupported transitions are not
  marked as best candidates, and structural MAD-table indices require exact finite
  integers.
- Updated `.gitignore` narrowly so the new stability test is tracked while the rest
  of the repository's ignored local `tests/` content remains unchanged.

## Scientific semantics

The only channel-noise metric is

\[
\hat\sigma_{MAD}=1.482602218505602\,
\operatorname{median}(|x-\operatorname{median}(x)|).
\]

It is computed independently in evenly distributed windows and summarized by the
median across windows. Values are stored in µV as `pre_filter_mad_uv` and
`post_filter_mad_uv`. The post-filter value is unreferenced; CMR is deliberately not
part of this auxiliary first version.

## Verification

Commands run:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/stability/__init__.py src/stability/channel_mad.py src/stability/mapping_check.py tests/stability/test_channel_mad.py tests/stability/test_mapping_check.py
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/stability/test_channel_mad.py
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/stability
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_multiday.py tests/stability
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_multiday.py tests/stability/test_channel_mad.py
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/stability/test_channel_mad.py tests/preprocess/test_multiday.py tests/preprocess/test_openephys_stream_classification.py tests/preprocess/test_io_source_selection.py
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess tests/stability
```

Observed results:

- Focused stability tests: 19 passed after adding and reviewing the mapping-candidate
  analysis.
- Multi-day staging plus stability tests: 39 passed.
- Final targeted stability/multi-day/Open Ephys suite: 63 passed.
- Broader preprocess plus stability suite: 273 passed, 17 failed. All 17 failures
  are in pre-existing preprocess/artifact/sorter tests outside `src.stability`; the
  failures concern current code/test mismatches such as artifact backend/config
  fields and Kilosort helpers. The new stability tests passed in that run.

A read-only smoke analysis was also run on the first real staged Day14 row using one
0.1-second window. It loaded all 128 channels, produced finite pre/post MAD values,
and wrote/read the temporary CSV and PNG successfully. Temporary smoke outputs were
then removed; no source data were changed.

The mapping check was run on the real Day14–Day217 `multi_day_mad.csv`. It wrote
`multi_day_mapping_check.csv` and `multi_day_mapping_check.png`. After applying the
hardware constraint that a flip must affect a complete 64-channel headstage, no
candidate had positive priority. The output therefore reports no supported
headstage permutation rather than promoting the nonphysical 32-channel pattern.

An independent read-only scientific/numerical review checked MAD semantics,
SpikeInterface scaling, physical-channel alignment, output provenance, and short
recordings. The review confirmed the MAD/window semantics and µV scaling and its
three actionable findings were addressed with regression tests.

A second independent review checked the mapping Spearman calculation, permutation
maps, candidate semantics, validation, and writes. It confirmed the numerical core
and all candidate maps, and identified unsupported-best labeling, structural-index
validation, and partial constant-block handling; all three were addressed with
focused regression tests.

## Limitations and next steps

- Intan scaling defaults to `int16` and `0.195 µV/count`; callers must override these
  parameters for recordings with different acquisition scaling.
- `pre_filter` includes the acquisition system's analog response and in-vivo neural
  signals and should not be interpreted as a pure electronics-noise bench test.
- The output reports measurements only. Bad-channel thresholds, classification,
  automatic rejection, CMR comparison, and GUI integration are deferred.
- Mapping ranking only tests the explicit blockwise candidate library and cannot
  discover an arbitrary adapter-specific wiring permutation. An exact hardware
  permutation can be added later if its connector map is known.
- After applying the hardware constraint that a flip affects a complete 64-channel
  headstage, the real Day14–Day217 MAD table contains no positively supported
  headstage-flip candidate. The earlier 32-channel Day172→Day193 pattern is therefore
  not treated as a physically plausible mapping event.
