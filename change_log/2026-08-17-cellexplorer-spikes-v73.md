# CellExplorer spikes v7.3 saving

## Date and state

- Date: 2026-08-17
- Base commit: `029e9236eac06aa414c7964872083c63dafcebb3`
- State: uncommitted
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

## Limitations

- The existing empty session output was not regenerated; rerunning the
  CellExplorer postprocess is required to replace it.
