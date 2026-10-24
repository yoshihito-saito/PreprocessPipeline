"""Optional one-time local staging for recordings read repeatedly over a network."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


COPY_BYTES = 8 * 1024 * 1024


def stage_inputs(filenames, directory, progress=None):
    """Copy named inputs to a fresh directory without modifying the sources.

    The whole input must fit on the scratch volume. Incomplete copies keep a
    ``.partial`` suffix so a failed run can never mistake them for complete
    sources. The caller owns the staged copies after this function returns.
    """
    paths = [Path(name).expanduser().resolve(strict=True) for name in filenames]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("Staging requires one or more named source files")
    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Staging directory must be new: {destination}")
    if any(destination.is_relative_to(path.parent) for path in paths):
        raise ValueError("Staging directory must be outside the source folders")
    sizes = [path.stat().st_size for path in paths]
    if any(size <= 0 for size in sizes):
        raise ValueError("Cannot stage an empty source file")
    total_bytes = sum(sizes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needed = total_bytes + max(2 * 1024**3, total_bytes // 10)
    if shutil.disk_usage(destination.parent).free < needed:
        raise OSError("Insufficient scratch space for sources plus sorting output")
    destination.mkdir()
    records = []
    staged = []
    started = time.perf_counter()
    copied_bytes = 0
    last_percent = 0
    if progress is not None:
        progress(0)
    for index, (source, size) in enumerate(zip(paths, sizes)):
        before = source.stat()
        if before.st_size != size:
            raise RuntimeError(f"Source size changed before staging: {source}")
        target = destination / f"{index:04d}_{source.name}"
        partial = target.with_name(target.name + ".partial")
        copy_started = time.perf_counter()
        source_digest = hashlib.sha256()
        with source.open("rb", buffering=COPY_BYTES) as reader:
            with partial.open("xb", buffering=COPY_BYTES) as writer:
                while chunk := reader.read(COPY_BYTES):
                    source_digest.update(chunk)
                    writer.write(chunk)
                    copied_bytes += len(chunk)
                    percent = copied_bytes * 100 // total_bytes
                    if progress is not None and percent >= last_percent + 5:
                        progress(min(99, percent))
                        last_percent = percent
                writer.flush()
                os.fsync(writer.fileno())
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise RuntimeError(f"Source changed while staging: {source}")
        if partial.stat().st_size != size:
            raise RuntimeError(f"Incomplete staged copy: {partial}")
        staged_digest = hashlib.sha256()
        with partial.open("rb", buffering=COPY_BYTES) as reader:
            while chunk := reader.read(COPY_BYTES):
                staged_digest.update(chunk)
        if staged_digest.digest() != source_digest.digest():
            raise RuntimeError(f"Staged copy checksum mismatch: {partial}")
        partial.rename(target)
        staged.append(target)
        records.append({"source": str(source), "staged": str(target), "bytes": size,
                        "source_mtime_ns": before.st_mtime_ns,
                        "sha256": source_digest.hexdigest(),
                        "copy_seconds": round(time.perf_counter() - copy_started, 3)})
    manifest = {"schema_version": 1, "kind": "terasort.local_staging",
                "stage_dir": str(destination), "total_bytes": total_bytes,
                "total_seconds": round(time.perf_counter() - started, 3),
                "sources": records,
                "note": "Scratch copies are retained after sorting; source files were opened read-only."}
    (destination / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if progress is not None:
        progress(100)
    return staged, manifest


def write_staging_record(results_dir, manifest):
    path = Path(results_dir) / "input_staging.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
