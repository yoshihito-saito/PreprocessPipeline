"""Bounded, deterministic candidate subsampling across time and probe location.

Only candidate metadata is scanned. No unit labels enter selection. The result
is a set of indices into the immutable candidate ledger plus inverse-inclusion
weights; dense spike assignment must still process every candidate or raw
sample. Selection uses two metadata passes and O(budget + chunk + strata)
memory, independent of recording duration.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def splitmix64(indices, seed):
    x = np.asarray(indices, dtype=np.uint64) + np.uint64(seed)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def geometry(sort):
    positions = np.load(sort / "channel_positions.npy")
    shanks = np.load(sort / "channel_shanks.npy").astype(int)
    labels = np.unique(shanks)
    centers = np.asarray([positions[shanks == label, 0].mean() for label in labels])
    return labels, centers, float(positions[:, 1].min()), float(positions[:, 1].max())


def stratum_ids(times, positions, amplitudes, *, source_samples, time_bins,
                shank_centers, depth_min, depth_bin_um, depth_bins, amp_edges):
    if (np.any(times < 0) or np.any(times >= source_samples)
            or not np.isfinite(positions).all() or not np.isfinite(amplitudes).all()):
        raise ValueError("Invalid candidate metadata")
    tb = np.minimum((times.astype(np.int64) * time_bins) // source_samples,
                    time_bins - 1)
    shank = np.abs(positions[:, 0, None] - shank_centers[None, :]).argmin(axis=1)
    depth = np.clip(np.floor((positions[:, 1] - depth_min) / depth_bin_um).astype(int),
                    0, depth_bins - 1)
    amp = np.searchsorted(amp_edges, amplitudes)
    return (((tb * len(shank_centers) + shank) * depth_bins + depth)
            * (len(amp_edges) + 1) + amp).astype(np.int32)


def allocate_quotas(counts, budget, min_per_nonempty):
    counts = np.asarray(counts, dtype=np.int64)
    target = min(int(budget), int(counts.sum()))
    if target < 0 or min_per_nonempty < 0:
        raise ValueError("Budget and minimum quota must be nonnegative")
    nonempty = int(np.count_nonzero(counts))
    floor = min_per_nonempty if nonempty * min_per_nonempty <= target else 0
    quota = np.minimum(counts, floor)
    remaining = target - int(quota.sum())
    while remaining:
        capacity = counts - quota
        active = np.flatnonzero(capacity > 0)
        weights = np.sqrt(counts[active].astype(float))
        ideal = remaining * weights / weights.sum()
        addition = np.minimum(capacity[active], np.floor(ideal).astype(np.int64))
        taken = int(addition.sum())
        if taken:
            quota[active] += addition
            remaining -= taken
        else:
            order = np.lexsort((active, -ideal))
            quota[active[order[:remaining]]] += 1
            remaining = 0
    assert int(quota.sum()) == target
    return quota


def select_stratified(times, positions, amplitudes, *, source_samples,
                      shank_centers, depth_min, depth_max, budget,
                      seed=0, time_bins=30, depth_bin_um=80.,
                      amp_edges=(), min_per_nonempty=4, chunk_size=250000):
    if budget <= 0 or chunk_size <= 0 or time_bins <= 0 or depth_bin_um <= 0:
        raise ValueError("Positive budget, chunk size, time bins and depth size required")
    n = len(times)
    if len(positions) != n or len(amplitudes) != n:
        raise ValueError("Candidate arrays differ in length")
    depth_bins = max(1, int(np.ceil((depth_max - depth_min) / depth_bin_um)))
    number_strata = time_bins * len(shank_centers) * depth_bins * (len(amp_edges) + 1)
    options = dict(source_samples=source_samples, time_bins=time_bins,
                   shank_centers=np.asarray(shank_centers), depth_min=depth_min,
                   depth_bin_um=depth_bin_um, depth_bins=depth_bins,
                   amp_edges=np.asarray(amp_edges))
    counts = np.zeros(number_strata, dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        ids = stratum_ids(times[start:stop], positions[start:stop],
                          amplitudes[start:stop], **options)
        counts += np.bincount(ids, minlength=number_strata)
    quotas = allocate_quotas(counts, budget, min_per_nonempty)
    best = {}  # stratum -> (hash priority, ledger index), at most its quota
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        strata = stratum_ids(times[start:stop], positions[start:stop],
                             amplitudes[start:stop], **options)
        indices = np.arange(start, stop, dtype=np.int64)
        keys = splitmix64(indices, seed)
        order = np.argsort(strata, kind="stable")
        sorted_strata = strata[order]
        borders = np.r_[0, np.flatnonzero(np.diff(sorted_strata)) + 1, len(order)]
        for left, right in zip(borders[:-1], borders[1:]):
            stratum = int(sorted_strata[left])
            quota = int(quotas[stratum])
            if quota == 0:
                continue
            group = order[left:right]
            if len(group) > quota:
                group = group[np.argpartition(keys[group], quota - 1)[:quota]]
            selected_keys, selected_indices = keys[group], indices[group]
            if stratum in best:
                old_keys, old_indices = best[stratum]
                selected_keys = np.r_[old_keys, selected_keys]
                selected_indices = np.r_[old_indices, selected_indices]
            if len(selected_keys) > quota:
                keep = np.argpartition(selected_keys, quota - 1)[:quota]
                selected_keys, selected_indices = selected_keys[keep], selected_indices[keep]
            best[stratum] = selected_keys, selected_indices
    selected = np.concatenate([indices for _, indices in best.values()])
    selected.sort()
    selected_strata = stratum_ids(times[selected], positions[selected],
                                  amplitudes[selected], **options)
    if len(selected) != int(quotas.sum()) or len(np.unique(selected)) != len(selected):
        raise AssertionError("Reservoir size or uniqueness failed")
    weights = counts[selected_strata] / quotas[selected_strata]
    return selected, weights.astype(np.float32), counts, quotas


def select_uniform(n, budget, seed=0, chunk_size=250000):
    """Same stable hash priority as a baseline, with bounded memory."""
    target = min(n, budget)
    keys_best = np.empty(0, dtype=np.uint64)
    index_best = np.empty(0, dtype=np.int64)
    for start in range(0, n, chunk_size):
        indices = np.arange(start, min(start + chunk_size, n), dtype=np.int64)
        keys = splitmix64(indices, seed)
        if len(keys) > target:
            keep = np.argpartition(keys, target - 1)[:target]
            keys, indices = keys[keep], indices[keep]
        keys_best = np.r_[keys_best, keys]
        index_best = np.r_[index_best, indices]
        if len(keys_best) > target:
            keep = np.argpartition(keys_best, target - 1)[:target]
            keys_best, index_best = keys_best[keep], index_best[keep]
    index_best.sort()
    return index_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", type=Path, required=True)
    parser.add_argument("--source-samples", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-bins", type=int, default=30)
    parser.add_argument("--depth-bin-um", type=float, default=80.)
    parser.add_argument("--minimum-per-stratum", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    sort = args.sort.resolve()
    times = np.load(sort / "spike_times.npy", mmap_mode="r")
    positions = np.load(sort / "spike_positions.npy", mmap_mode="r")
    amplitudes = np.load(sort / "amplitudes.npy", mmap_mode="r")
    labels, centers, depth_min, depth_max = geometry(sort)
    stride = max(1, len(amplitudes) // 100000)
    amp_edges = np.quantile(amplitudes[::stride][:100001], (.5, .9))
    selected, weights, counts, quotas = select_stratified(
        times, positions, amplitudes, source_samples=args.source_samples,
        shank_centers=centers, depth_min=depth_min, depth_max=depth_max,
        budget=args.budget, seed=args.seed, time_bins=args.time_bins,
        depth_bin_um=args.depth_bin_um, amp_edges=amp_edges,
        min_per_nonempty=args.minimum_per_stratum, chunk_size=args.chunk_size)
    uniform = select_uniform(len(times), args.budget, args.seed, args.chunk_size)
    output.mkdir(parents=True)
    np.save(output / "stratified_indices.npy", selected)
    np.save(output / "stratified_inverse_inclusion_weight.npy", weights)
    np.save(output / "uniform_indices.npy", uniform)
    np.save(output / "stratum_counts.npy", counts)
    np.save(output / "stratum_quotas.npy", quotas)
    report = {"method": "deterministic two-pass balanced reservoir over time, shank, depth and amplitude",
              "sort": str(sort), "candidate_count": len(times), "budget": args.budget,
              "selected_count": len(selected), "seed": args.seed,
              "time_bins": args.time_bins, "depth_bin_um": args.depth_bin_um,
              "minimum_per_nonempty_stratum": args.minimum_per_stratum,
              "shank_labels": labels.tolist(), "shank_center_x_um": centers.tolist(),
              "amplitude_edges": amp_edges.tolist(), "nonempty_strata": int(np.count_nonzero(counts)),
              "inverse_inclusion_weight_range": [float(weights.min()), float(weights.max())],
              "note": "Indices select calibration candidates only; dense assignment must still inspect all data."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
