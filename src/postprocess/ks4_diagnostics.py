from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KS4ClusterDiagnostic:
    cluster_id: int
    n_spikes: int
    amplitude_min: float
    amplitude_median: float
    amplitude_max: float
    amplitude_centers: tuple[float, float] | None
    amplitude_balance: float | None
    amplitude_separation: float | None
    amplitude_explained_fraction: float | None
    split_amplitude_threshold: float | None
    best_matching_cluster_id: int | None
    best_template_correlation: float | None
    peak_channel: int | None


def _load_array(results_dir: Path, name: str) -> np.ndarray:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required KS4 result file not found: {path}")
    return np.load(path)


def _cluster_ids(spike_clusters: np.ndarray) -> np.ndarray:
    return np.unique(spike_clusters.astype(int))


def _cluster_mask(spike_clusters: np.ndarray, cluster_id: int) -> np.ndarray:
    return spike_clusters.astype(int) == int(cluster_id)


def _safe_log_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    amp = np.asarray(amplitudes, dtype=float)
    amp = amp[np.isfinite(amp) & (amp > 0)]
    if amp.size == 0:
        return np.zeros(0, dtype=float)
    return np.log10(amp)


def _kmeans_1d_two_clusters(values: np.ndarray, *, max_iter: int = 64) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Need at least two 1D values for 2-means.")

    centers = np.quantile(x, [0.25, 0.75]).astype(float)
    if np.isclose(centers[0], centers[1]):
        centers = np.array([x.min(), x.max()], dtype=float)
    if np.isclose(centers[0], centers[1]):
        labels = np.zeros(x.size, dtype=int)
        return centers, labels

    labels = np.zeros(x.size, dtype=int)
    for _ in range(max_iter):
        distances = np.abs(x[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in (0, 1):
            members = x[labels == idx]
            if members.size:
                centers[idx] = float(members.mean())

    order = np.argsort(centers)
    centers = centers[order]
    relabel = np.zeros(2, dtype=int)
    relabel[order] = np.arange(2, dtype=int)
    labels = relabel[labels]
    return centers, labels


def _amplitude_split_metrics(amplitudes: np.ndarray) -> dict[str, float] | None:
    log_amp = _safe_log_amplitudes(amplitudes)
    if log_amp.size < 20:
        return None
    if np.allclose(log_amp, log_amp[0]):
        return None

    centers, labels = _kmeans_1d_two_clusters(log_amp)
    counts = np.bincount(labels, minlength=2).astype(float)
    if np.any(counts == 0):
        return None

    global_center = float(log_amp.mean())
    sse_single = float(np.square(log_amp - global_center).sum())
    sse_split = 0.0
    variances: list[float] = []
    for idx in (0, 1):
        members = log_amp[labels == idx]
        center = float(centers[idx])
        sse_split += float(np.square(members - center).sum())
        variances.append(float(members.var()))

    pooled_var = max(float(np.mean(variances)), 1e-12)
    return {
        "center_low": float(10 ** centers[0]),
        "center_high": float(10 ** centers[1]),
        "balance": float(np.min(counts) / counts.sum()),
        "separation": float(abs(centers[1] - centers[0]) / np.sqrt(pooled_var)),
        "explained_fraction": float(0.0 if sse_single <= 0 else 1.0 - (sse_split / sse_single)),
        "threshold": float(10 ** ((centers[0] + centers[1]) / 2.0)),
    }


def _peak_channels(templates: np.ndarray) -> np.ndarray:
    abs_templates = np.abs(np.asarray(templates, dtype=float))
    if abs_templates.ndim != 3:
        raise ValueError("templates.npy must have shape (n_clusters, n_time, n_channels)")
    max_per_channel = abs_templates.max(axis=1)
    return np.argmax(max_per_channel, axis=1).astype(int)


def _template_similarity(results_dir: Path, templates: np.ndarray) -> np.ndarray:
    path = results_dir / "similar_templates.npy"
    if path.exists():
        sim = np.asarray(np.load(path), dtype=float)
        if sim.shape[0] == sim.shape[1]:
            return sim

    flat = np.asarray(templates, dtype=float).reshape(templates.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = flat / norms
    return normalized @ normalized.T


def diagnose_ks4_cluster(results_dir: str | Path, cluster_id: int) -> KS4ClusterDiagnostic:
    results_path = Path(results_dir).resolve()
    spike_clusters = _load_array(results_path, "spike_clusters.npy").astype(int)
    amplitudes = _load_array(results_path, "amplitudes.npy").astype(float)
    templates = _load_array(results_path, "templates.npy")

    cluster_ids = _cluster_ids(spike_clusters)
    if int(cluster_id) not in set(cluster_ids.tolist()):
        raise ValueError(f"Cluster {cluster_id} not found in {results_path}")

    similarities = _template_similarity(results_path, templates)
    peaks = _peak_channels(templates)
    mask = _cluster_mask(spike_clusters, cluster_id)
    amp = amplitudes[mask]
    split = _amplitude_split_metrics(amp)

    row = similarities[int(cluster_id)].copy()
    row[int(cluster_id)] = -np.inf
    best_idx = int(np.argmax(row)) if row.size else None
    best_corr = float(row[best_idx]) if best_idx is not None and np.isfinite(row[best_idx]) else None

    return KS4ClusterDiagnostic(
        cluster_id=int(cluster_id),
        n_spikes=int(mask.sum()),
        amplitude_min=float(np.min(amp)) if amp.size else float("nan"),
        amplitude_median=float(np.median(amp)) if amp.size else float("nan"),
        amplitude_max=float(np.max(amp)) if amp.size else float("nan"),
        amplitude_centers=None if split is None else (split["center_low"], split["center_high"]),
        amplitude_balance=None if split is None else split["balance"],
        amplitude_separation=None if split is None else split["separation"],
        amplitude_explained_fraction=None if split is None else split["explained_fraction"],
        split_amplitude_threshold=None if split is None else split["threshold"],
        best_matching_cluster_id=best_idx,
        best_template_correlation=best_corr,
        peak_channel=int(peaks[int(cluster_id)]) if int(cluster_id) < len(peaks) else None,
    )


def rank_ks4_amplitude_split_candidates(
    results_dir: str | Path,
    *,
    min_spikes: int = 250,
    min_balance: float = 0.1,
    top_k: int = 10,
) -> list[KS4ClusterDiagnostic]:
    results_path = Path(results_dir).resolve()
    spike_clusters = _load_array(results_path, "spike_clusters.npy").astype(int)

    diagnostics: list[KS4ClusterDiagnostic] = []
    for cluster_id in _cluster_ids(spike_clusters):
        diag = diagnose_ks4_cluster(results_path, int(cluster_id))
        if diag.n_spikes < int(min_spikes):
            continue
        if diag.amplitude_balance is None or diag.amplitude_balance < float(min_balance):
            continue
        diagnostics.append(diag)

    diagnostics.sort(
        key=lambda d: (
            -1.0 if d.amplitude_explained_fraction is None else -d.amplitude_explained_fraction,
            -1.0 if d.amplitude_separation is None else -d.amplitude_separation,
            1.0 if d.best_template_correlation is None else -d.best_template_correlation,
        )
    )
    return diagnostics[: int(top_k)]


def _format_diagnostic(diag: KS4ClusterDiagnostic) -> str:
    centers = (
        "-"
        if diag.amplitude_centers is None
        else f"{diag.amplitude_centers[0]:.2f}/{diag.amplitude_centers[1]:.2f}"
    )
    balance = "-" if diag.amplitude_balance is None else f"{diag.amplitude_balance:.2f}"
    separation = "-" if diag.amplitude_separation is None else f"{diag.amplitude_separation:.2f}"
    explained = (
        "-"
        if diag.amplitude_explained_fraction is None
        else f"{diag.amplitude_explained_fraction:.2f}"
    )
    best = (
        "-"
        if diag.best_matching_cluster_id is None or diag.best_template_correlation is None
        else f"{diag.best_matching_cluster_id} ({diag.best_template_correlation:.3f})"
    )
    return (
        f"cluster={diag.cluster_id} n_spikes={diag.n_spikes} peak_ch={diag.peak_channel} "
        f"amp_med={diag.amplitude_median:.2f} centers={centers} balance={balance} "
        f"sep={separation} explained={explained} best_match={best}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose KS4 amplitude-driven merge candidates.")
    parser.add_argument("results_dir", type=Path, help="Kilosort4 results folder")
    parser.add_argument("--cluster-id", type=int, default=None, help="Inspect a single cluster")
    parser.add_argument("--min-spikes", type=int, default=250, help="Minimum spikes for candidate scan")
    parser.add_argument("--min-balance", type=float, default=0.1, help="Minimum minor-fraction for 2-way split")
    parser.add_argument("--top-k", type=int, default=10, help="How many clusters to print in scan mode")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)

    if args.cluster_id is not None:
        diag = diagnose_ks4_cluster(args.results_dir, args.cluster_id)
        if args.json:
            print(json.dumps(asdict(diag), indent=2, sort_keys=True))
        else:
            print(_format_diagnostic(diag))
        return 0

    diagnostics = rank_ks4_amplitude_split_candidates(
        args.results_dir,
        min_spikes=args.min_spikes,
        min_balance=args.min_balance,
        top_k=args.top_k,
    )
    payload: Any = [asdict(diag) for diag in diagnostics]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for diag in diagnostics:
            print(_format_diagnostic(diag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
