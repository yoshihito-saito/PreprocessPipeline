# Memory-efficient state scoring

Reduce candidate-selection memory without changing scientific outputs. Slurm job
424 exhausted its 768 GiB allocation on a 141,134,479,360-byte, 128-channel LFP.
The current code retains full-rate int16 and float64 arrays for all candidates
and evaluates up to 128 channels simultaneously.

Read candidates from a read-only memmap, slice every fifth sample before exact
int16-to-float64 conversion, and retain only active candidate inputs. Cap candidate
concurrency at four (respect smaller requested counts). Aggregate scores and plot
histograms promptly in original channel order; preserve tie-breaking. Copy only
the selected full-rate SW/theta channels for outputs. Also slice selected inputs
before conversion in final scoring. Preserve EMG, frequency grids, windows,
smoothing, thresholds, global normalization, output shapes/dtypes, and MAT v7.3.

Verify against an archived pre-edit module on the complete test_rec_260509 LFP,
using identical saved settings and separate output directories. Compare all MAT
scientific values exactly (including NaNs), excluding detection date and output
directory provenance; compare decoded figures and measure peak RSS. Run focused
state-scoring and MAT tests, inspect the diff, and obtain independent review.
Real multi-day execution is not required; explicitly report extrapolation limits.

Git branch requested: feature/memory-efficient-state-scoring. Creation currently
blocked by the environment's read-only .git mount; source work remains uncommitted.
