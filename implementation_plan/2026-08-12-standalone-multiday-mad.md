# Standalone multi-day channel MAD analysis

## Goal and motivation

Add an auxiliary `src.stability` analysis module that accepts a staged
`multi_day_selected_subepochs.csv`, reads each selected source recording, computes a
simple per-channel MAD-derived noise estimate before and after spike-band filtering,
and writes `multi_day_mad.csv` plus a diagnostic heatmap.

This is an upstream recording-quality check for later multi-day unit-stability work.
It must remain opt-in and separate from the preprocess pipeline and GUI.

Literature and design discussion are recorded under
`paperflow/multiday-neuron-tracking-electrophysiology/`, especially `proposal.md`.

## Current problem

The staged multi-day CSV preserves source `.dat` paths, source type, channel counts,
sample counts, sampling rates, and subepoch order, but there is no independent API
that uses this provenance to compare channel noise across subepochs. Existing
postprocess `noise_levels` are unit-analysis support values and do not provide this
source-subepoch channel view.

## Affected files

- New package: `src/stability/`
- New focused tests: `tests/stability/`
- `README.md`: standalone usage example
- `implementation_plan/README.md`
- `change_log/` after implementation and verification

No preprocess pipeline, GUI, configuration model, execution stage, or output contract
will call this module.

## API and output

Provide:

```python
analyze_multi_day_mad(
    selected_subepochs_csv,
    *,
    output_csv_path=None,
    output_figure_path=None,
    bandpass_min_hz=500.0,
    bandpass_max_hz=8000.0,
    window_duration_s=1.0,
    num_windows=20,
    dtype="int16",
    gain_to_uV=0.195,
    offset_to_uV=0.0,
    overwrite=False,
)
```

Default outputs are siblings of the input CSV:

- `multi_day_mad.csv`
- `multi_day_mad.png`

The CSV has one row per `(staged subepoch, ephys channel)` and includes source
provenance, channel identity, sampling metadata, `pre_filter_mad_uv`, and
`post_filter_mad_uv`. It also stores the source dtype, per-channel gain/offset, and
scaling source needed to reproduce the µV conversion. Window count and sampled
duration are provenance fields, not additional scientific metrics.

Expose lower-level compute and plotting functions so notebooks or scripts can reuse
the analysis without writing files.

In addition to the two-stage heatmap, provide a two-panel channel-wise line plot.
Overlay all channels in each panel, encode the zero-based channel index with a
continuous colormap and shared colorbar, and show pre-filter and post-bandpass MAD
in the upper and lower panels respectively.

Provide a second standalone API that consumes the saved MAD table without rereading
raw recordings:

```python
analyze_multi_day_mapping_check("multi_day_mad.csv")
```

It writes `multi_day_mapping_check.csv` and `multi_day_mapping_check.png`. The table
has one row per adjacent-subepoch, channel-block, and plausible permutation, plus a
global rank and a marker for the best candidate at each transition. It remains a
candidate-ranking aid and does not classify, remap, or remove channels.

Candidate permutations are restricted to each aligned 64-channel Intan headstage
block: full block reversal, swapping its 32-channel halves, and reversing within
each 32-channel half. A mapping event cannot be proposed for only 32 channels, since
the hardware constraint is that a flipped headstage affects its complete 64-channel
block. This bounded library avoids fitting an arbitrary permutation to two noisy
scalar profiles. If no 64-channel candidate has positive priority, the plot reports
that no supported headstage permutation was found.

## Algorithm semantics

For each source subepoch:

1. Load only electrophysiology channels in physical/source order.
   - Intan binary rows use the CSV channel count plus configurable dtype/gain/offset.
   - Open Ephys rows resolve the stream from `source_subepoch_path` and select the
     ephys channel indices described by `structure.oebin`.
   - Require the ordered physical channel-name map to agree across selected
     subepochs; otherwise a channel-by-subepoch comparison would be invalid.
2. Choose up to `num_windows` non-overlapping, evenly distributed windows. A
   subepoch shorter than one requested window uses its full duration.
3. Read each window in µV before filtering.
4. Apply the requested SpikeInterface bandpass lazily with float32 filter output,
   then read the same windows in µV. Float output avoids adding integer rounding to
   the post-filter estimate.
5. For each window and channel compute the Gaussian-sigma-equivalent centered MAD:

   \[
   \hat\sigma_{MAD} = 1.482602218505602\,
   \operatorname{median}(|x-\operatorname{median}(x)|).
   \]

6. Store the median of the window-level estimates for each channel and stage.

`pre_filter` means the source acquisition trace before this analysis' digital
bandpass. It can still contain acquisition-system analog filtering and biological
signals. `post_filter` means bandpass-filtered but unreferenced; common median
reference is deliberately excluded from this first auxiliary check.

### Mapping candidate scores

For adjacent selected subepochs and each candidate block, calculate Spearman
similarities of pre- and post-filter channel MAD profiles under the identity mapping
and candidate permutation. Report:

\[
S_{identity}=\min(r_{pre,id}, r_{post,id})
\]

and

\[
G_{consensus}(P)=\min(r_{pre,P}-r_{pre,id},
                       r_{post,P}-r_{post,id}).
\]

The minimum makes the screen conservative: positive gain requires both pre- and
post-filter profiles to support the same candidate. Also retain all component
similarities, component gains, `support_count`, and
`profile_discontinuity = 1 - S_identity` so a reviewer can distinguish a
permutation-like rescue from an unexplained recording-quality change. Rank
candidates by

\[
Q(P)=\max(0,G_{consensus}(P))\max(0,S_{permuted}(P)),
\]

where `S_permuted` is the smaller pre/post permuted similarity. This prevents a
candidate that only improves a poor match to another poor match from outranking a
smaller but credible rescue. Preserve the uncombined similarities and gains for
interpretation, and do not apply an automatic bad/good threshold.
Only mark a row `best_for_transition` when its priority score is strictly positive;
transitions with no supported permutation remain unflagged. A constant candidate
block is uninformative and is skipped locally rather than suppressing valid blocks;
reject the analysis only when every candidate is uninformative.

## Validation and failure behavior

- Reject missing/empty CSV files and missing required columns.
- Reject missing source files, non-positive sampling/channel/sample values, invalid
  bandpass limits, unsupported source types, and source metadata mismatches.
- Reject inconsistent physical channel maps across subepochs.
- Require µV-scalable traces; do not silently label native ADC counts as µV.
- Reject recordings/windows too short for zero-phase filtering with a specific
  filter-length error.
- Refuse to replace existing output files unless `overwrite=True`.
- Write CSV and figure through same-directory temporary files before replacement.
- Preserve CSV `staged_order` rather than sorting paths or names independently.

## Verification strategy

- Unit-test the centered MAD equation against exact arrays and axis behavior.
- Test deterministic window placement, short recordings, and invalid parameters.
- Test CSV parsing/provenance and one-row-per-channel output with synthetic
  SpikeInterface recordings.
- Verify that a low-frequency synthetic component is reduced by the bandpass result.
- Verify CSV/PNG creation, overwrite protection, and plotting from a saved metrics
  CSV.
- Verify that the channel-wise plot has pre/post panels, one distinctly colored line
  per channel, and the expected metric values.
- Verify exact identity/permuted Spearman scores, conservative consensus gain,
  candidate enumeration and ranking, incomplete/constant-profile validation, and
  mapping-check CSV/PNG output.
- Run the focused stability test suite and the relevant preprocessing tests.

## Non-goals

- No bad-channel classification or threshold.
- No automatic channel-swap classification or remapping.
- No unconstrained data-fitted permutation; only explicit 64-channel headstage
  candidates.
- No automatic channel removal or chanMap modification.
- No common-median referencing in the first version.
- No GUI controls, preprocess-stage invocation, Slurm stage, or notebook requirement.
- No unit-level stability or cross-day neuron identity analysis.
