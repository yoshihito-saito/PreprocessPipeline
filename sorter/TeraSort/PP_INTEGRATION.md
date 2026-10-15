# Bundled TeraSort source

This runtime/package source was copied from [ayalab1/TeraSort](https://github.com/ayalab1/TeraSort), branch `codex/preprocess-bounded-nt`, commit `734e145f654219199fa700238b75432f6ad40944`. The tracked `src/terasort` package is copied byte-for-byte. The upstream GPL-3.0-only license is retained in `LICENSE`.

PreprocessPipeline defaults **Sorter path** to this directory. The source does not contain a ready Python environment. On Linux, from the PreprocessPipeline checkout:

```bash
bash sorter/TeraSort/scripts/install-linux.sh
sorter/TeraSort/.venv/bin/python sorter/TeraSort/scripts/build_cublas_bridge_linux.py
```

The second command is needed only when selecting the `cublas` backend. The installer checks the NVIDIA driver, installs dependencies into this directory's `.venv`, and performs its own CUDA checks. PreprocessPipeline never runs that installer automatically. For a preexisting separate environment, set `PREPROCESS_TERASORT_PYTHON` before starting PreprocessPipeline.
