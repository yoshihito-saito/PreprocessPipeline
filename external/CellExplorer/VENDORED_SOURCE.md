# Vendored CellExplorer Source

This directory contains the ayalab1 CellExplorer working tree vendored for
PreprocessPipeline MATLAB integration.

- Source repository: https://github.com/ayalab1/CellExplorer
- Vendored commit: `31ef26f9818b260a27273860f529ae40efa5b803`
- Upstream summary: `Merge pull request #25 from ayalab1/fix-mergepoints-waveform-fallback`
- Vendored on: 2026-06-23
- Local integration branch: `feature/integrating-cellexplore`

The upstream `.git` directory is intentionally excluded. Runtime integration
code should add this directory to the MATLAB path explicitly so the pipeline
does not depend on a globally installed CellExplorer version.
