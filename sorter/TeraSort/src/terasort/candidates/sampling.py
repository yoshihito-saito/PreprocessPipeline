"""Apply the bounded calibration reservoir to an immutable SCB candidate bank.

SCB events may contain multiple channel detections of one action potential.
This sampler preserves their original row IDs; it does not spatially deduplicate
or assert that every selected row represents a distinct neuron spike.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .sampling_core import select_stratified, select_uniform


class BankMetadata:
    """Shared chunk cache prevents rereading each HDF5 event field separately."""

    def __init__(self, events, channel_positions, origin, lateral_axis, depth_axis):
        self.events = events
        self.channel_positions = channel_positions
        self.origin = origin
        self.lateral_axis = lateral_axis
        self.depth_axis = depth_axis
        self.cache_key = None
        self.cache_rows = None

    def rows(self, index):
        key = ((index.start, index.stop, index.step) if isinstance(index, slice)
               else id(index))
        if self.cache_key != key:
            self.cache_rows = self.events[index]
            self.cache_key = key
        return self.cache_rows

    def field(self, kind):
        return BankField(self, kind)


class BankField:
    def __init__(self, parent, kind):
        self.parent = parent
        self.kind = kind

    def __len__(self):
        return len(self.parent.events)

    def __getitem__(self, index):
        rows = self.parent.rows(index)
        if self.kind == "times":
            return rows["sample_index"] - self.parent.origin
        if self.kind == "positions":
            coordinates = self.parent.channel_positions[rows["channel_index"]]
            return coordinates[:, [self.parent.lateral_axis, self.parent.depth_axis]]
        if self.kind == "amplitudes":
            return rows["snr"]
        raise ValueError(self.kind)


def infer_depth_axis(channel_positions):
    """Prefer the axis with the most distinct contact positions, then span."""
    scores = [(len(np.unique(channel_positions[:, axis])),
               float(np.ptp(channel_positions[:, axis]))) for axis in range(3)]
    return max(range(3), key=lambda axis: scores[axis])


def shank_centers(channel_positions, probe_path, lateral_axis, depth_axis):
    if probe_path is None:
        if np.ptp(channel_positions[:, lateral_axis]) > 100:
            raise ValueError("Wide lateral geometry requires --probe-json for shank grouping")
        return [1], [float(channel_positions[:, lateral_axis].mean())]
    probe = json.loads(probe_path.read_text())
    shanks = np.asarray(probe["kcoords"]).astype(int)
    if len(shanks) != len(channel_positions):
        raise ValueError("Probe and bank channel counts disagree")
    coordinates = np.stack([probe["xc"], probe["yc"]], axis=1)
    bank_coordinates = channel_positions[:, [lateral_axis, depth_axis]]
    if not np.allclose(coordinates, bank_coordinates, atol=.01):
        raise ValueError("Probe and bank geometry differ")
    labels = np.unique(shanks)
    return labels.tolist(), [float(channel_positions[shanks == label, lateral_axis].mean())
                             for label in labels]


def sample_bank(bank_path, output, *, probe_path=None, budget=30000,
                seed=0, time_bins=30, depth_bin_um=80.,
                minimum_per_stratum=4, chunk_size=250000,
                depth_axis=None, lateral_axis=0):
    bank_path = Path(bank_path).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(bank_path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("Incomplete candidate bank")
        manifest = json.loads(handle.attrs["manifest_json"])
        if manifest.get("format") != "SCB":
            raise ValueError("Not an SCB candidate bank")
        events = handle["events"]
        blocks = handle["blocks"]
        if not len(events) or not len(blocks):
            raise ValueError("Candidate bank has no events or block bounds")
        origin = int(blocks[0]["start_sample"])
        stop = int(blocks[-1]["stop_sample"])
        if stop <= origin:
            raise ValueError("Invalid source sample bounds")
        positions = handle["channel_positions_um"][:]
        if not np.isfinite(positions).all():
            raise ValueError("Bank needs finite channel coordinates")
        if depth_axis is None:
            depth_axis = infer_depth_axis(positions)
        if depth_axis == lateral_axis or depth_axis not in (0, 1, 2) or lateral_axis not in (0, 1, 2):
            raise ValueError("Depth and lateral axes must be distinct 0, 1 or 2")
        labels, centers = shank_centers(positions, probe_path, lateral_axis, depth_axis)
        metadata = BankMetadata(events, positions, origin, lateral_axis, depth_axis)
        times = metadata.field("times")
        locations = metadata.field("positions")
        amplitudes = metadata.field("amplitudes")
        stride = max(1, len(events) // 100000)
        edges = np.quantile(amplitudes[::stride][:100001], (.5, .9))
        selected, weights, counts, quotas = select_stratified(
            times, locations, amplitudes, source_samples=stop - origin,
            shank_centers=np.asarray(centers), depth_min=float(positions[:, depth_axis].min()),
            depth_max=float(positions[:, depth_axis].max()), budget=budget, seed=seed,
            time_bins=time_bins, depth_bin_um=depth_bin_um, amp_edges=edges,
            min_per_nonempty=minimum_per_stratum, chunk_size=chunk_size)
        uniform = select_uniform(len(events), budget, seed, chunk_size)
        output.mkdir(parents=True)
        np.save(output / "stratified_row_indices.npy", selected)
        np.save(output / "stratified_inverse_inclusion_weight.npy", weights)
        np.save(output / "uniform_row_indices.npy", uniform)
        np.save(output / "stratum_counts.npy", counts)
        np.save(output / "stratum_quotas.npy", quotas)
        report = {
            "format": "terasort.scb_calibration_sample_v1",
            "bank_path": str(bank_path),
            "bank_bytes": bank_path.stat().st_size,
            "bank_mtime_ns": bank_path.stat().st_mtime_ns,
            "bank_version": manifest.get("version"),
            "bank_event_definition": manifest.get("event_definition"),
            "row_count": len(events), "selected_rows": len(selected),
            "sample_rate_hz": manifest.get("sample_rate_hz"),
            "source_start_sample": origin, "source_stop_sample": stop,
            "budget": budget, "seed": seed,
            "time_bins": time_bins, "depth_bin_um": depth_bin_um,
            "lateral_axis": lateral_axis, "depth_axis": depth_axis,
            "minimum_per_nonempty_stratum": minimum_per_stratum,
            "shank_labels": labels, "shank_center_x_um": centers,
            "snr_edges": edges.tolist(),
            "nonempty_strata": int(np.count_nonzero(counts)),
            "selected_nonempty_strata": int(np.count_nonzero(quotas)),
            "note": "Selected row IDs are calibration candidates, not unique spike identities. "
                    "No raw events were removed from the source bank.",
        }
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-bins", type=int, default=30)
    parser.add_argument("--depth-bin-um", type=float, default=80.)
    parser.add_argument("--depth-axis", type=int, choices=(0, 1, 2),
                        help="SCB coordinate axis for probe depth; inferred if omitted")
    parser.add_argument("--lateral-axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--minimum-per-stratum", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sample_bank(args.bank, args.output, probe_path=args.probe_json,
                         budget=args.budget, seed=args.seed, time_bins=args.time_bins,
                         depth_bin_um=args.depth_bin_um,
                         minimum_per_stratum=args.minimum_per_stratum,
                         chunk_size=args.chunk_size, depth_axis=args.depth_axis,
                         lateral_axis=args.lateral_axis)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
