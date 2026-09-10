# Kilosort4 partition channel exclusions

## Failure and intended behavior
Day10–Day245 probe1 failed with `Channel '48' was not in probe['chanMap']`. The recording was sliced to 32 channels, but the complement of its channels in the original 128-channel binary was also passed to Kilosort as bad_channels.

Only full-width recordings should receive these original binary channel exclusions. For sliced recordings the exclusion has already happened; do not pass the removed channels again. Preserve full-session preprocessed bad-channel handling, partition membership, geometry, and scientific parameters.

## Implementation and verification
Change the bad_channels assignment in src/preprocess/sorter_runner.py. Add a regression at execute_sorting_job using noncontiguous channels from a 128-channel input, both raw and preprocessed. Reproduce the failure before the fix; verify existing full-session and partition tests afterward. No full GPU run required.
