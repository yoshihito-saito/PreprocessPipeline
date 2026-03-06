# Kilosort2.5 Update Notes

Date: 2026-03-06

## Summary

`Kilosort-2.5` was added to this repository under `sorter/Kilosort-2.5`.

## Local Changes

- Added an optional merge shape gate in `postProcess/find_merges.m`.
- When enabled, merges must pass the existing refractory criteria and also show similar CCG/ACG shapes.
- Default behavior is unchanged because the new option is disabled unless set in `ops`.

## Optional Settings

```matlab
ops.mergeShapeEnable = false;
ops.mergeShapeMinCorr = 0.8;
ops.mergeShapeExcludeMs = 2;
ops.mergeShapeWindowMs = 50;
```

Commented examples were added to:

- `configFiles/StandardConfig_MOVEME.m`
- `configFiles/configFile384.m`

## Validation

- MATLAB `checkcode` passed for `postProcess/find_merges.m`
- a synthetic MATLAB run with `mergeShapeEnable = true` completed without errors
