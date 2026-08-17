# CellExplorer spikes v7.3 saving

## Date and state

- Date: 2026-08-17
- Implementation commit: `4fcd9a0620a7acb2f6eead21fbe1d79e2edf7b0a`
- Recovery verification update: uncommitted
- Plan: [2026-08-17 CellExplorer spikes v7.3 saving](../implementation_plan/2026-08-17-cellexplorer-spikes-v73.md)

## What changed

- Saved both `.unsorted.spikes.cellinfo.mat` and `.spikes.cellinfo.mat` with
  MATLAB `-v7.3` from the PreprocessPipeline CellExplorer wrapper.
- This prevents structures larger than the MATLAB v7 limit from producing a
  header-only MAT file while leaving the CellExplorer version and processing
  workflow unchanged.

## Verification

Ran:

```text
/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "msgs=checkcode('external/matlab/run_cell_explorer_processing.m','-id'); if ~isempty(msgs), disp(struct2table(msgs)); error('checkcode reported messages'); end; spikes=struct('times',{{[0.1 0.2]}},'UID',1); f=[tempname,'.mat']; cleanup=onCleanup(@() delete(f)); save(f,'spikes','-v7.3'); vars=whos('-file',f); assert(any(strcmp({vars.name},'spikes'))); loaded=load(f,'spikes'); assert(isequal(loaded.spikes,spikes)); disp('checkcode: ok'); disp('v7.3 save/load: ok');"
```

Result:

```text
checkcode: ok
v7.3 save/load: ok
```

The affected multi-day session was then recovered from the non-drift sorter
`Kilosort4_2026-08-11_203942` using a temporary Python launcher and the
vendored MATLAB `loadSpikes` implementation. Waveform extraction completed in
29 minutes for 111 units. The installed MAT file is 889,650,843 bytes, and
MATLAB reports one `spikes` struct with an uncompressed size of approximately
2.3874 GB.

## Limitations

- The full CellExplorer metric pipeline was not rerun because the existing
  `cell_metrics` output already contains the same 111 units.
