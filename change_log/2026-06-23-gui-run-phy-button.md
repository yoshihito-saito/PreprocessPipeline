# GUI Run Phy Button

## Date and git state

- Date: 2026-06-23
- Status: uncommitted

## What changed

- Added a `Manual Curation` group below `Ephys run` in the Ephys GUI.
- Added a `Run phy` button.
- Added Phy launch handling in `src/preprocess/gui/app.py`.
- The launch target resolves through the existing postprocess sorting-folder
  settings: explicit `sorting folder` first, otherwise the newest local sorting
  output.
- If a matching `<run>_spi/params.py` exists, it is preferred over the original
  Kilosort folder so manual curation opens the postprocessed SpikeInterface/Phy
  output.
- The Phy process is launched detached as `phy template-gui params.py`.
- The working directory is set to the folder containing `params.py`, preserving
  relative paths such as `../<basename>.dat`.
- The GUI first looks for `phy` next to the Python executable running the GUI,
  then falls back to `PATH`.

## Why

Manual curation currently requires opening a terminal and manually selecting the
right local Kilosort/Phy folder. A GUI button reduces path mistakes and keeps
the interactive ephys workflow in the GUI.

## Verification

Command run:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/gui/app.py
```

Result: command completed successfully.

Target resolution check:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "from pathlib import Path; from tempfile import TemporaryDirectory; from src.preprocess.gui.app import MainWindow; d=TemporaryDirectory(); root=Path(d.name); run=root/'Kilosort_2026'; spi=root/'Kilosort_2026_spi'; run.mkdir(); spi.mkdir(); (run/'params.py').write_text('base'); (spi/'params.py').write_text('spi'); print(MainWindow._resolve_phy_params_path(object(), run)); d.cleanup()"
```

Result: the resolved path ended in `Kilosort_2026_spi/params.py`.

Additional check:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "import sys; from pathlib import Path; print(Path(sys.executable).resolve().parent / 'phy')"
```

Resolved Phy executable:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/phy
```

## Observed behavior

The GUI code now exposes `Run phy` under `Manual Curation` and will launch Phy
against the resolved `params.py` folder when clicked.

## Known limitations and next steps

- The launched Phy process is detached; the GUI does not monitor or stop it.
- Full runtime validation requires pressing the button in an active GUI session
  with display access.
