# GUI Phy Log Summary

## Date and git state

- Date: 2026-06-23
- Status: uncommitted

## What changed

- Added delayed Phy log summarization after a successful GUI `Run phy` launch.
- The GUI reads `<params.py parent>/phy.log` after launch and calls
  `neuro_py.raw.spike_sorting.phy_log_to_epocharray`.
- The GUI log now reports the Phy log path, curation epoch count, and estimated
  total curation time.
- The helper sets writable `MPLCONFIGDIR` and `NUMBA_CACHE_DIR` defaults under
  `/tmp` before importing `neuro_py`, avoiding cache/import failures observed in
  this environment.
- Log summarization failures are warnings only and do not fail the Phy launch.

## Why

The manual curation workflow should surface Phy log information directly in the
GUI after launching Phy, using the existing `neuro_py` helper requested by the
user.

## Verification

Compile check:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/gui/app.py
```

Result: command completed successfully.

`neuro_py` smoke check:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-preprocess-gui NUMBA_CACHE_DIR=/tmp/numba-preprocess-gui /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "from neuro_py.raw.spike_sorting import phy_log_to_epocharray; e=phy_log_to_epocharray('/local/workdir/ys2375/PreprocessPipeline/sorting_temp/RM018_day39_260620/Kilosort_2026-06-22_165139_spi/phy.log'); print(getattr(e, 'n_epochs', None)); print(getattr(e, 'duration', None))"
```

Result:

```text
15
1:22:51:784 hours
```

## Known limitations and next steps

- The GUI summarizes once shortly after launch; it does not continuously monitor
  `phy.log`.
- If `phy.log` has not been created yet, the GUI emits a warning rather than
  retrying indefinitely.
