"""Streaming export of session shards to common Kilosort/Phy arrays.

This exports sorting results, not Kilosort's native rez/ops training state.
Each probe/day has its own dataset; gaps are removed only from the raw-file
concatenation clock, with original coordinates retained in a separate array.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
import shutil
from pathlib import Path
import sqlite3
import tempfile

import h5py
import numpy as np

BLOCK = 65536
MAX_TEMPLATES = 4096


def ensure_session_export(source_root, *, progress=None, flat_output=None):
    """Publish the default export, preserving completed exports and curation."""
    source = Path(source_root).expanduser().resolve(strict=True)
    flat = flat_output is not None
    output = Path(flat_output).expanduser().resolve() if flat else source / "phy"
    if output == source or source.is_relative_to(output):
        raise ValueError("Export must not replace or contain the source session")
    digest = hashlib.sha256((source / "run.json").read_bytes()).hexdigest()
    if progress is not None:
        progress({"stage": "export_start"})
    if output.exists():
        report = json.loads((output / "export.json").read_text(encoding="utf-8"))
        if (report.get("format") != "TeraSort Kilosort/Phy export" or
                report.get("version") != 1 or report.get("source_run_sha256") != digest
                or report.get("layout", "probe-day") != ("flat-single-group" if flat else "probe-day")):
            raise ValueError("Existing phy export belongs to a different session; preserve it and choose a new output root")
        for group in report["datasets"]:
            folder = output if flat else output / group["probe_id"] / group["day_id"]
            required = ["params.py", "spike_times.npy", "spike_clusters.npy",
                        "spike_templates.npy", "templates.npy", "amplitudes.npy"]
            if group.get("pc_features_exported"):
                required.extend(["pc_features.npy","pc_feature_ind.npy","pc_feature_basis.npy",
                                 "pc_feature_provenance.json"])
            for name in required:
                if not (folder / name).is_file():
                    raise ValueError(f"Existing phy export is missing {name}; preserve curation before repairing")
    else:
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            marker = partial / "export_owner.json"
            if (not marker.is_file() or json.loads(marker.read_text()).get("source_run_sha256") != digest):
                raise ValueError("Unrecognized phy.partial directory; move it aside before retrying")
            # Only discard this exporter's unpublished temporary output. Never
            # traverse a redirected path or remove completed/curated results.
            if partial.is_symlink() or partial.resolve() != partial:
                raise ValueError("Refusing to clean a redirected export directory")
            for path in partial.rglob("*"):
                if path.is_symlink() or path.resolve() != path:
                    raise ValueError("Refusing to clean redirected export contents")
            shutil.rmtree(partial)
        report = export_session(source, output, progress=progress,
                                **({'flat': True} if flat else {}))
    if progress is not None:
        progress({"stage": "export_complete"})
    return {"output_root": str(output), "datasets": report["datasets"]}


class NpyStream:
    """Append rows without mapping an entire long recording into memory."""
    def __init__(self, path, dtype, shape):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        self.rows = 0
        self.handle = open(path, "xb")
        np.lib.format.write_array_header_2_0(self.handle, {
            "descr": np.lib.format.dtype_to_descr(self.dtype),
            "fortran_order": False, "shape": self.shape})

    def write(self, values):
        values = np.asarray(values, dtype=self.dtype, order="C")
        if values.shape[1:] != self.shape[1:]:
            raise ValueError("Incorrect array row shape")
        self.handle.write(values.tobytes())
        self.rows += len(values)

    def close(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()
        if exc_type is None and self.rows != self.shape[0]:
            raise ValueError(f"Incomplete NPY array: {self.rows}/{self.shape[0]}")


def _rows(h, *, indexed=False):
    data = h["spikes"]
    for start in range(0, len(data), BLOCK):
        rows = data[start:start + BLOCK]
        if indexed:
            width = h["spike_features"].shape[2]
            packed = np.empty(len(rows), np.dtype(rows.dtype.descr+[("_feature_row","<i8"),("_features","<f4",(3,width))]))
            for name in rows.dtype.names:packed[name] = rows[name]
            packed["_feature_row"] = np.arange(start,start+len(rows),dtype=np.int64)
            packed["_features"] = h["spike_features"][start:start+len(rows)]
            rows = packed
        if "spike_export_mask" in h:
            mask = h["spike_export_mask"][start:start + BLOCK]
            if mask.shape != (len(rows),) or np.any((mask != 0) & (mask != 1)):
                raise ValueError("Invalid spike export mask")
            rows = rows[mask.astype(bool)]
        yield rows


def _sorted_rows(h, *, indexed=False):
    """Sort one shard; exceptionally large shards spill to bounded SQLite."""
    if len(h["spikes"]) <= BLOCK:
        for rows in _rows(h, indexed=indexed):
            yield rows[np.argsort(rows["sample_index"], kind="stable")]
        return
    dtype = h["spikes"].dtype
    if indexed:dtype = np.dtype(dtype.descr+[("_feature_row","<i8"),("_features","<f4",(3,h["spike_features"].shape[2]))])
    with tempfile.TemporaryDirectory(prefix="terasort-export-") as temp:
        connection = sqlite3.connect(str(Path(temp) / "sort.sqlite"))
        try:
            connection.execute("PRAGMA cache_size=-16384")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE spikes(t INTEGER, ordinal INTEGER PRIMARY KEY, payload BLOB)")
            ordinal = 0
            for rows in _rows(h, indexed=indexed):
                connection.executemany("INSERT INTO spikes VALUES(?,?,?)", (
                    (int(row["sample_index"]), ordinal + i, row.tobytes())
                    for i, row in enumerate(rows)))
                ordinal += len(rows)
            connection.commit()
            cursor = connection.execute("SELECT payload FROM spikes ORDER BY t, ordinal")
            while batch := cursor.fetchmany(BLOCK):
                yield np.frombuffer(b"".join(item[0] for item in batch), dtype=dtype)
        finally:
            connection.close()


def _identity(path, count):
    writer = NpyStream(path, "<f4", (count, count))
    try:
        for start in range(0, count, 64):
            stop = min(count, start + 64)
            rows = np.zeros((stop - start, count), np.float32)
            rows[np.arange(stop - start), np.arange(start, stop)] = 1
            writer.write(rows)
    finally:
        writer.close()


def _check(h, path, run):
    if (not h.attrs.get("complete", False)
            or h.attrs.get("format") != "TeraSortSessionShard"
            or h.attrs.get("version") != 1
            or h.attrs.get("config_sha256") != run["config_sha256"]):
        raise ValueError(f"Incomplete or mismatched shard: {path}")
    if "model_after/waveforms" not in h or "spikes" not in h:
        raise ValueError(f"Missing export data: {path}")


def export_session(source_root, output_root, *, progress=None, flat=False):
    """Export completed contiguous shard coverage, without reading raw voltage.

    Outputs are built in a new sibling .partial directory and published only
    on success. Existing exports and curation are never overwritten.
    """
    source = Path(source_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Export output or partial directory already exists: {output}")
    if output == source or source.is_relative_to(output):
        raise ValueError("Export must not replace or contain the source session")
    raw_run = (source / "run.json").read_bytes()
    run = json.loads(raw_run)
    groups = []
    for probe in run["source"]["probes"]:
        probe_id = probe["probe_id"]
        if Path(probe_id).name != probe_id or probe_id in (".", ".."):
            raise ValueError("Unsafe probe identifier")
        segments = probe["segments"]
        paths = sorted((source / probe_id).glob("*.h5"))
        by_day = {}
        for path_index, path in enumerate(paths):
            with h5py.File(path, "r") as h:
                _check(h, path, run)
                day = str(h.attrs["day_id"])
                if Path(day).name != day or day in (".", ".."):
                    raise ValueError("Unsafe day identifier")
                start, stop = int(h.attrs["start_sample"]), int(h.attrs["stop_sample"])
                stat = path.stat()
                by_day.setdefault(day, []).append({"path": path, "start": start,
                    "stop": stop, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            if progress is not None and path_index % 100 == 0:
                progress({"stage": "export_scan", "completed_shards": path_index + 1,
                          "total_shards": len(paths)})
        for items in by_day.values():
            items.sort(key=lambda item: item["start"])
        first = run["config"].get("start_sample", 0)
        last = run["config"].get("stop_sample")
        for segment in segments:
            begin = max(first, segment["start_sample"])
            end = segment["start_sample"] + segment["n_samples"]
            if last is not None:
                end = min(end, last)
            if begin >= end:
                continue
            relevant = [x for x in by_day.get(segment["day_id"], [])
                        if x["stop"] > begin and x["start"] < end]
            cursor = begin
            for item in relevant:
                if item["start"] != cursor or item["stop"] > end:
                    raise ValueError("Shard coverage has a gap, overlap, or invalid boundary")
                cursor = item["stop"]
            if cursor != end:
                raise ValueError("Session is incomplete; finish or resume sorting before export")
        for day, items in by_day.items():
            day_segments = [s for s in segments if s["day_id"] == day]
            if not day_segments:
                raise ValueError("Shard day is absent from source manifest")
            groups.append((probe, day, items, day_segments))
    if not groups:
        raise ValueError("No completed session shards")
    if flat and len(groups) != 1:
        raise ValueError("A flat Phy export requires exactly one probe/day")
    partial.mkdir(parents=True)
    (partial / "export_owner.json").write_text(json.dumps({
        "source_run_sha256": hashlib.sha256(raw_run).hexdigest()}), encoding="utf-8")
    reports = []
    for probe, day, items, segments in groups:
        folder = partial if flat else partial / probe["probe_id"] / day
        if not flat:
            folder.mkdir(parents=True)
        reports.append(_export_group(folder, probe, day, items, segments, run, progress))
    report = {"format": "TeraSort Kilosort/Phy export", "version": 1,
              "source_root": str(source), "source_run_sha256": hashlib.sha256(raw_run).hexdigest(),
              "datasets": reports, "raw_voltage_bytes_read": 0,
              "pc_features_exported": all(r["pc_features_exported"] for r in reports),
              "native_kilosort_training_state": False}
    if flat:
        report["layout"] = "flat-single-group"
    (partial / "export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(partial, output)
    return report


def _export_group(folder, probe, day, items, segments, run, progress):
    from .session_recovery import read_recovered, merge_recovered, MAX_PER_DAY
    nt = run["config"].get("nt", 61)
    recovery_batches, recovery_templates = [], {}
    recovery_feature_batches = []
    feature_basis = feature_width = None
    feature_presence = []
    count = 0
    channels = mapping = None
    n_templates = 0
    # Small header/count pass, followed by a single data pass.
    for item_index, item in enumerate(items):
        with h5py.File(item["path"], "r") as h:
            shape = h["model_after/waveforms"].shape
            n = shape[0]
            if len(shape) != 3 or shape[1] != nt or shape[2] > 64:
                raise ValueError("Invalid or oversized template shape")
            if h["model_after/channels"].shape != (n, shape[2]):
                raise ValueError("Invalid template channel shape")
            if n > MAX_TEMPLATES:
                raise ValueError(f"More than {MAX_TEMPLATES} templates in a day; split export by epoch")
            if ("template_linking/template_to_unit" in h and
                    h["template_linking/template_to_unit"].shape != (n,)):
                raise ValueError("Invalid template identity map shape")
            current_channels = h["model_after/channels"][:]
            current_map = (h["template_linking/template_to_unit"][:] if
                           "template_linking/template_to_unit" in h else np.arange(n))
            if channels is not None and (not np.array_equal(current_channels[:len(channels)], channels)
                    or not np.array_equal(current_map[:len(mapping)], mapping)):
                raise ValueError("Template channels or identity links changed within an export day")
            channels, mapping, n_templates = current_channels, current_map, n
            recovered = read_recovered(h)
            present = "spike_features" in h
            feature_presence.append(present)
            if present:
                from .session_phy_capture import validate_basis
                basis = h["phy_feature_basis"][:]
                validate_basis(basis)
                shape_pc = h["spike_features"].shape
                if (len(shape_pc)!=3 or shape_pc[:2]!=(len(h["spikes"]),3)
                        or not 1<=shape_pc[2]<=16 or h["spike_features"].dtype!=np.float32):
                    raise ValueError("Invalid measured Phy feature payload")
                if feature_basis is not None and (not np.array_equal(feature_basis,basis)
                        or feature_width!=shape_pc[2]):
                    raise ValueError("Phy feature basis changed within an export day")
                feature_basis,feature_width = basis,shape_pc[2]
                extra_features = (h["novelty_recovered_features"][:] if len(recovered)
                    else np.empty((0,3,feature_width),np.float32))
                if extra_features.shape!=(len(recovered),3,feature_width):
                    raise ValueError("Invalid recovery Phy feature payload")
            if len(recovered):
                if 'spike_export_filter_json' in h.attrs:
                    minimum = json.loads(h.attrs['spike_export_filter_json'])['minimum']
                    keep_recovered = recovered['fitted_amplitude'] >= minimum
                    recovered = recovered[keep_recovered]
                    if present:extra_features = extra_features[keep_recovered]
                for unit in np.unique(recovered['unit_id']):
                    if int(unit) in recovery_templates:
                        raise ValueError('Novelty unit recovered in multiple enrollment shards')
                    recovery_templates[int(unit)] = (h['model_after/waveforms'][unit],
                        int(np.count_nonzero(recovered['unit_id'] == unit)))
                recovery_batches.append(recovered)
                if present:recovery_feature_batches.append(extra_features)
                if sum(len(batch) for batch in recovery_batches) > MAX_PER_DAY:
                    raise ValueError('Novelty recovery exceeds the bounded day payload')
                count += len(recovered)
            if "spike_export_mask" in h:
                mask = h["spike_export_mask"]
                if mask.shape != (len(h["spikes"]),):
                    raise ValueError("Invalid spike export mask shape")
                for start in range(0, len(mask), BLOCK):
                    values = mask[start:start + BLOCK]
                    if np.any((values != 0) & (values != 1)):
                        raise ValueError("Invalid spike export mask")
                    count += int(np.count_nonzero(values))
            else:
                count += len(h["spikes"])
        if progress is not None and item_index % 100 == 0:
            progress({"stage": "export_count", "probe_id": probe["probe_id"], "day_id": day,
                      "completed_shards": item_index + 1, "total_shards": len(items)})
    if any(feature_presence) and not all(feature_presence):
        raise ValueError("Mixed Phy feature coverage within an export day")
    have_features = all(feature_presence)
    if run["config"].get("phy_features") and not have_features:
        raise ValueError("Measured Phy features are missing from completed shards")
    if n_templates == 0:
        raise ValueError("No templates available for Phy export")
    nchan = len(probe["geometry"])
    if (channels.ndim != 2 or channels.shape[0] != n_templates
            or channels.shape[1] > 64 or np.any(channels < -1) or np.any(channels >= nchan)
            or mapping.shape != (n_templates,) or np.any(mapping < 0) or np.any(mapping >= n_templates)):
        raise ValueError("Invalid template channel/identity map")
    if any(len(np.unique(row[row >= 0])) != np.count_nonzero(row >= 0) for row in channels):
        raise ValueError("Duplicate template channels")
    group_ids, cluster_map = np.unique(mapping, return_inverse=True)
    sums = np.zeros((n_templates, nt, channels.shape[1]), np.float64)
    weights = np.zeros(n_templates, np.int64)
    from .session_store import SPIKE_DTYPE
    recovered = np.concatenate(recovery_batches) if recovery_batches else np.empty(0, SPIKE_DTYPE)
    recovery_order = np.argsort(recovered['sample_index'], kind='stable')
    recovered = recovered[recovery_order]
    recovered_features = (np.concatenate(recovery_feature_batches)[recovery_order]
        if recovery_feature_batches else np.empty((0,3,feature_width or 1),np.float32))
    if have_features:
        packed = np.empty(len(recovered),np.dtype(SPIKE_DTYPE.descr+[("_feature_row","<i8"),("_features","<f4",(3,feature_width))]))
        for name in SPIKE_DTYPE.names:packed[name] = recovered[name]
        packed["_feature_row"] = -1-np.arange(len(recovered),dtype=np.int64)
        packed["_features"] = recovered_features
        recovered = packed
    for unit, (waveform, weight) in recovery_templates.items():
        sums[unit] += waveform * weight
        weights[unit] += weight
    coverage = np.zeros(len(recovered), bool)
    for item in items:
        coverage |= (recovered['sample_index'] >= item['start']) & (recovered['sample_index'] < item['stop'])
    if not coverage.all():
        raise ValueError('Novelty recovery lies outside completed source coverage')
    last_waveforms = None
    bases = np.cumsum([0] + [s["n_samples"] for s in segments[:-1]], dtype=np.int64)
    geometry = np.asarray(probe["geometry"], np.float32)
    specs = {"spike_times": ("<i8", (count,)), "spike_times_session": ("<i8", (count,)),
             "spike_templates": ("<i4", (count,)), "spike_clusters": ("<i4", (count,)),
             "amplitudes": ("<f4", (count,)), "spike_positions": ("<f4", (count, 2))}
    if have_features:
        specs["pc_features"] = ("<f4",(count,3,feature_width))
    gain_for_features = float(probe["gain_uv_per_count"])
    if not np.isfinite(gain_for_features) or gain_for_features<=0:raise ValueError("Invalid source gain")
    previous = -1
    with ExitStack() as stack:
        writers = {}
        for name, (dtype, shape) in specs.items():
            writers[name] = stack.enter_context(NpyStream(folder / (name + ".npy"), dtype, shape))
        for index, item in enumerate(items):
            stat = item["path"].stat()
            if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("Source shard changed during export")
            with h5py.File(item["path"], "r") as h:
                _check(h, item["path"], run)
                last_waveforms = h["model_after/waveforms"][:]
                if not np.isfinite(last_waveforms).all():
                    raise ValueError("Nonfinite template waveform")
                shard_counts = np.zeros(n_templates, np.int64)
                left, right = np.searchsorted(recovered['sample_index'], [item['start'], item['stop']])
                extra = recovered[left:right]
                def native():
                    for rows in _sorted_rows(h,indexed=have_features):
                        if np.any(rows['unit_id'] < 0) or np.any(rows['unit_id'] >= len(last_waveforms)):
                            raise ValueError('Invalid spike coordinates, IDs, or amplitudes: native template ID')
                        yield rows
                for rows in merge_recovered(native(), extra, BLOCK):
                    if not len(rows):
                        continue
                    times, units = rows["sample_index"], rows["unit_id"]
                    if (np.any(units < 0) or np.any(units >= n_templates)
                            or times[0] < item["start"] or times[-1] >= item["stop"]
                            or times[0] < previous or not np.isfinite(rows["fitted_amplitude"]).all()):
                        raise ValueError("Invalid spike coordinates, IDs, or amplitudes")
                    previous = int(times[-1])
                    local = np.full(len(rows), -1, np.int64)
                    for base, segment in zip(bases, segments):
                        start = segment["start_sample"]
                        keep = (times >= start) & (times < start + segment["n_samples"])
                        local[keep] = times[keep] - start + base
                    if np.any(local < 0):
                        raise ValueError("Spike lies outside the raw-file source clock")
                    contacts = rows["channel_index"]
                    if np.any(contacts >= nchan):
                        raise ValueError("Invalid spike channel")
                    for name, values in {"spike_times": local, "spike_times_session": times,
                            "spike_templates": units, "spike_clusters": cluster_map[units],
                            "amplitudes": rows["fitted_amplitude"], "spike_positions": geometry[contacts]}.items():
                        writers[name].write(values)
                    if have_features:
                        # Carry measured PCs through the same bounded sort.
                        # Avoid very large, scattered HDF5 fancy-index reads.
                        pc = rows["_features"]
                        if not np.isfinite(pc).all():raise ValueError("Nonfinite measured Phy features")
                        writers["pc_features"].write(pc/gain_for_features)
                    shard_counts += np.bincount(units, minlength=n_templates)
                shard_counts -= np.bincount(extra['unit_id'], minlength=n_templates)
                n = len(last_waveforms)
                sums[:n] += last_waveforms * shard_counts[:n, None, None]
                weights += shard_counts
            if progress is not None:
                progress({"stage": "export", "probe_id": probe["probe_id"], "day_id": day,
                          "completed_shards": index + 1, "total_shards": len(items)})
    templates = np.divide(sums, np.maximum(weights, 1)[:, None, None]).astype(np.float32)
    templates[weights == 0] = last_waveforms[weights == 0]
    gain = float(probe["gain_uv_per_count"])
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Invalid source gain")
    _template_files(folder, templates / gain, channels, nchan)
    if have_features:
        np.save(folder/"pc_feature_ind.npy",channels[:,:feature_width].astype(np.int32))
        np.save(folder/"pc_feature_basis.npy",feature_basis)
        (folder/"pc_feature_provenance.json").write_text(json.dumps(dict(
            format="TeraSort measured Phy PCA features",version=1,spikes=count,pcs=3,
            local_channels=feature_width,raw_voltage_bytes_read=0,
            source="resident preprocessed voltage and retained raw novelty observations",
            basis="fixed three temporal PCs learned from bounded seeds",
            units="ADC counts",scope="curation features only; sorting assignments unchanged"),indent=2),encoding="utf-8")
    np.save(folder / "channel_map.npy", np.asarray(probe.get("channel_map", np.arange(nchan)), dtype=np.int32))
    np.save(folder / "channel_positions.npy", geometry)
    np.save(folder / "channel_shanks.npy", np.asarray(probe["shank"], np.int32))
    _identity(folder / "whitening_mat.npy", nchan)
    _identity(folder / "whitening_mat_inv.npy", nchan)
    params = {"dat_path": [Path(s["path"]).as_posix() for s in segments],
              "n_channels_dat": probe.get("n_channels_dat", nchan), "dtype": "int16", "offset": 0,
              "sample_rate": probe["sample_rate_hz"], "hp_filtered": probe.get("input_is_preprocessed", False)}
    (folder / "params.py").write_text("".join(f"{k} = {v!r}\n" for k, v in params.items()), encoding="utf-8")
    for filename, field in (("cluster_group.tsv", "group"), ("cluster_KSLabel.tsv", "KSLabel")):
        (folder / filename).write_text(f"cluster_id\t{field}\n" +
            "".join(f"{i}\tunsorted\n" for i in range(len(group_ids))), encoding="utf-8")
    (folder / "cluster_groups.csv").write_text("cluster_id,group\n" +
        "".join(f"{i},3\n" for i in range(len(group_ids))), encoding="utf-8")
    (folder / "unit_mapping.tsv").write_text("template_id\tcluster_id\tsource_template_id\tsource_linked_unit_id\n" +
        "".join(f"{i}\t{cluster_map[i]}\t{i}\t{mapping[i]}\n" for i in range(n_templates)), encoding="utf-8")
    (folder / "source_segments.json").write_text(json.dumps([
        {**s, "export_start_sample": int(base)} for s, base in zip(segments, bases)], indent=2), encoding="utf-8")
    (folder / "EXPORT_NOTES.txt").write_text(
        "TeraSort assignments in common Kilosort1/4-Phy array format, not native KS training state.\n"
        "Templates: spike-count-weighted model_after snapshots, in ADC counts; identity spatial whitening.\n"
        "Amplitudes: fitted dimensionless scales against the evolving sorting template.\n"
        + ("Measured per-spike PCA features from the dense pass. " if have_features else
         "Legacy session: run terasort phy-features to add measured PCA features. ") +
        "No fabricated contamination/good-unit labels. All clusters are unsorted.\n" +
        "spike_times.npy uses concatenated source-file samples; spike_times_session.npy preserves the session clock.\n"
        "Novelty recovery rows are merged into source order; prior assignment shards remain immutable.\n"
        "No cross-day identity merges. Raw voltage files remain at their original paths.\n", encoding="utf-8")
    return {"probe_id": probe["probe_id"], "day_id": day, "spikes": count,
            "templates": n_templates, "clusters": len(group_ids), "shards": len(items),
            "pc_features_exported": have_features}


def _template_files(folder, templates, channels, nchan):
    n = len(templates)
    templates = np.where((channels >= 0)[:, None, :], templates, 0)
    wave = NpyStream(folder / "templates.npy", "<f4", (n, templates.shape[1], nchan))
    ind = NpyStream(folder / "templates_ind.npy", "<i4", (n, nchan))
    similarity = NpyStream(folder / "similar_templates.npy", "<f4", (n, n))
    norms = np.sqrt(np.sum(templates.astype(np.float64) ** 2, axis=(1, 2)))
    try:
        for i in range(n):
            valid = channels[i] >= 0
            dense = np.zeros((templates.shape[1], nchan), np.float32)
            dense[:, channels[i, valid]] = templates[i][:, valid]
            wave.write(dense[None])
            ind.write(np.arange(nchan, dtype=np.int32)[None])
            scores = np.zeros(n, np.float32)
            for j in range(n):
                common, a, b = np.intersect1d(channels[i, valid], channels[j], return_indices=True)
                if len(common) and norms[i] > 0 and norms[j] > 0:
                    scores[j] = np.sum(templates[i][:, valid][:, a].astype(np.float64)
                                       * templates[j][:, b]) / (norms[i] * norms[j])
            similarity.write(np.clip(scores, -1, 1)[None])
    finally:
        wave.close()
        ind.close()
        similarity.close()
