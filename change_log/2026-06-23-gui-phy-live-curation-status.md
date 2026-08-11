# GUI Phy Live Curation Status

## Date and git state

- Date: 2026-06-23
- Status: uncommitted

## What changed

- Changed GUI `Run phy` from detached launch to a managed Phy `QProcess`.
- Added live polling of `<params.py parent>/cluster_info.tsv` every 30 seconds.
- Added curation count summaries with total, good, mua, noise, unclassified, and
  other counts.
- Added final curation summary when the Phy process exits.
- Reformatted `neuro_py`/`nelpy` duration text such as `1:22:51:784 hours` into
  `1 h 22 min 51.784 s`.
- Prevented closing the GUI while Phy is still active, so the final summary can
  be written after Phy closes.

## Why

The raw Phy log duration format was hard to read, and detached launch made it
impossible for the GUI to know when Phy was closed. Managing Phy as a separate
process allows live progress logging and final curation counts.

## Verification

Compile check:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/gui/app.py
```

Result: command completed successfully.

Cluster count and duration formatting smoke check:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "from pathlib import Path; from src.preprocess.gui.app import MainWindow; p=Path('/local/workdir/ys2375/PreprocessPipeline/sorting_temp/RM018_day39_260620/Kilosort_2026-06-22_165139_spi'); counts=MainWindow._read_phy_cluster_counts(object(), p); print(counts); print(MainWindow._format_phy_cluster_counts(object(), counts)); print(MainWindow._format_phy_duration(object(), '1:22:51:784 hours'))"
```

Result:

```text
{'total': 1230, 'good': 150, 'mua': 0, 'noise': 1080, 'unclassified': 0, 'other': 0}
total=1230, good=150, mua=0, noise=1080, unclassified=0, other=0
1 h 22 min 51.784 s
```

## Known limitations and next steps

- The GUI polls `cluster_info.tsv`; Phy must write/save updates to that file for
  counts to change.
- The GUI logs only when counts change, plus a final forced summary on close.
