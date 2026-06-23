# Vendored CellExplorer Source

This directory contains the ayalab1 CellExplorer working tree vendored for
PreprocessPipeline MATLAB integration.

- Source repository: https://github.com/ayalab1/CellExplorer
- Vendored commit: `b7e9314bcb3b909fcfaf03e7d614a32cf1cacbba`
- Vendored on: 2026-06-23
- Local integration branch: `feature/integrating-cellexplore`

The upstream `.git` directory is intentionally excluded. Runtime integration
code should add this directory to the MATLAB path explicitly so the pipeline
does not depend on a globally installed CellExplorer version.
