import pandas as pd
import numpy as np
import shutil
from pathlib import Path

def create_cluster_info_tsv(
    phy_dir: str | Path,
    metrics_df: pd.DataFrame,
    backup: bool = True,
) -> pd.DataFrame:
    """
    Create a unified `cluster_info.tsv` file containing all metrics in one place.

    Parameters
    ----------
    phy_dir : str | Path
        Phy output folder containing `cluster_si_unit_ids.tsv`.
    metrics_df : pd.DataFrame
        DataFrame indexed by SI unit_id (or with a 'cluster_id'/'si_unit_id' column) that contains
        all quality metrics (isi_violations_ratio, isi_violations_count, presence_ratio, snr,
        amplitude_median, etc.).
    backup : bool
        If True, make a timestamped backup of the original TSV before overwriting.

    Returns
    -------
    pd.DataFrame
        The created `cluster_info.tsv` contents as a DataFrame sorted by cluster_id.
    """
    phy_dir = Path(phy_dir)

    # --- 1) Map cluster_id (Phy) <-> si_unit_id (SpikeInterface)
    map_path = phy_dir / "cluster_si_unit_ids.tsv"
    if not map_path.exists():
        raise FileNotFoundError(f"Mapping not found: {map_path}")
    mapping = pd.read_csv(map_path, sep="\t")
    mapping["cluster_id"] = pd.to_numeric(mapping["cluster_id"], errors="coerce").astype("Int64")
    mapping["si_unit_id"] = pd.to_numeric(mapping["si_unit_id"], errors="coerce")

    # --- 2) Normalize metrics_df to have a 'si_unit_id' column
    df = metrics_df.copy()
    if "si_unit_id" not in df.columns and "cluster_id" not in df.columns:
        df.index.name = "si_unit_id"
        df = df.reset_index()
    if "cluster_id" in df.columns and "si_unit_id" not in df.columns:
        df = df.merge(mapping, on="cluster_id", how="inner")
    else:
        df = df.merge(mapping, on="si_unit_id", how="inner")
    
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cluster_id"]).set_index("cluster_id").sort_index()

    # --- 3) Include group column if cluster_group.tsv exists
    cg_path = phy_dir / "cluster_group.tsv"
    if cg_path.exists():
        cg = pd.read_csv(cg_path, sep="\t")
        cg["cluster_id"] = pd.to_numeric(cg["cluster_id"], errors="coerce").astype("Int64")
        cg = cg.set_index("cluster_id")
        if "group" in cg.columns:
            df = df.join(cg[["group"]], how="left")
            df["group"] = df["group"].fillna("unsorted")
    else:
        df["group"] = "unsorted"

    # --- 4) Save unified cluster_info.tsv
    info_path = phy_dir / "cluster_info.tsv"
    out_df = df.reset_index().sort_values("cluster_id")
    
    if backup and info_path.exists():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = info_path.with_stem(f"{info_path.stem}_{timestamp}_bak")
        shutil.copyfile(info_path, bak)
    
    out_df.to_csv(info_path, sep="\t", index=False)
    
    return out_df


def mark_noise_clusters_from_metrics(
    phy_dir: str | Path,
    metrics_df: pd.DataFrame,
    thresholds: dict,
    backup: bool = True,
    reset_to_unsorted: bool = True,
    update_cluster_info: bool = True,
) -> pd.DataFrame:
    """
    Update Phy's `cluster_group.tsv` to mark `group = "noise"` based on quality metrics.
    Optionally also update `cluster_info.tsv` with all metrics and group labels.

    Parameters
    ----------
    phy_dir : str | Path
        Phy output folder containing `cluster_group.tsv` and `cluster_si_unit_ids.tsv`.
    metrics_df : pd.DataFrame
        DataFrame indexed by SI unit_id (or with a 'cluster_id'/'si_unit_id' column) that contains:
        - isi_violations_ratio
        - isi_violations_count
        - presence_ratio
        - snr
        - amplitude_median
    thresholds : dict
        Rules to mark noise. Supported keys (use any subset):
        - "isi_violations_ratio_gt": float  # mark noise if ratio > value
        - "isi_violations_count_gt": int    # mark noise if count > value
        - "presence_ratio_lt": float        # mark noise if presence < value
        - "snr_lt": float                   # mark noise if snr < value
        - "amplitude_median_lt": float    # mark noise if abs(amplitude_median) < value
    backup : bool
        If True, make a timestamped backup of the original TSV before overwriting.
    reset_to_unsorted : bool
        If True, reset all existing labels to "unsorted" before applying new noise labels.
    update_cluster_info : bool
        If True, also create/update a unified `cluster_info.tsv` with all metrics and group labels.

    Returns
    -------
    pd.DataFrame
        The updated `cluster_group.tsv` contents as a DataFrame sorted by cluster_id.
    """
    phy_dir = Path(phy_dir)

    # --- 1) Map cluster_id (Phy) <-> si_unit_id (SpikeInterface)
    map_path = phy_dir / "cluster_si_unit_ids.tsv"
    if not map_path.exists():
        raise FileNotFoundError(f"Mapping not found: {map_path}")
    mapping = pd.read_csv(map_path, sep="\t")  # columns: cluster_id, si_unit_id
    mapping["cluster_id"] = pd.to_numeric(mapping["cluster_id"], errors="coerce").astype("Int64")
    mapping["si_unit_id"] = pd.to_numeric(mapping["si_unit_id"], errors="coerce")

    # --- 2) Normalize metrics_df to have a 'si_unit_id' column, then join mapping to get cluster_id
    df = metrics_df.copy()
    if "si_unit_id" not in df.columns and "cluster_id" not in df.columns:
        # assume index is SI unit_id
        df.index.name = "si_unit_id"
        df = df.reset_index()
    if "cluster_id" in df.columns and "si_unit_id" not in df.columns:
        # rare case: metrics already in cluster_id space
        df = df.merge(mapping, on="cluster_id", how="inner")
    else:
        df = df.merge(mapping, on="si_unit_id", how="inner")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cluster_id"]).set_index("cluster_id").sort_index()

    # --- 3) Build boolean masks for each rule (missing columns are treated as NaN -> False)
    # Normalize amplitude to absolute if requested
    df["_amp_abs_"] = np.abs(df["amplitude_median"]) if "amplitude_median" in df.columns else np.nan

    conds = []
    # ISI violations: both ratio AND count must exceed thresholds (AND condition)
    if ("isi_violations_ratio_gt" in thresholds and "isi_violations_count_gt" in thresholds and
        "isi_violations_ratio" in df.columns and "isi_violations_count" in df.columns):
        isi_cond = ((df["isi_violations_ratio"] > thresholds["isi_violations_ratio_gt"]) &
                   (df["isi_violations_count"] > thresholds["isi_violations_count_gt"]))
        conds.append(isi_cond)
    elif "isi_violations_ratio_gt" in thresholds and "isi_violations_ratio" in df.columns:
        conds.append(df["isi_violations_ratio"] > thresholds["isi_violations_ratio_gt"])
    elif "isi_violations_count_gt" in thresholds and "isi_violations_count" in df.columns:
        conds.append(df["isi_violations_count"] > thresholds["isi_violations_count_gt"])
    
    if "presence_ratio_lt" in thresholds and "presence_ratio" in df.columns:
        conds.append(df["presence_ratio"] < thresholds["presence_ratio_lt"])
    if "snr_lt" in thresholds and "snr" in df.columns:
        conds.append(df["snr"] < thresholds["snr_lt"])
    if "amplitude_median_lt" in thresholds and "_amp_abs_" in df.columns:
        conds.append(df["_amp_abs_"] < thresholds["amplitude_median_lt"])

    noise_mask = pd.concat(conds, axis=1).any(axis=1) if conds else pd.Series(False, index=df.index)

    # --- 4) Read or create cluster_group.tsv (Phy expects: cluster_id<TAB>group)
    cg_path = phy_dir / "cluster_group.tsv"
    if cg_path.exists():
        cg = pd.read_csv(cg_path, sep="\t")
        if not {"cluster_id", "group"}.issubset(set(cg.columns)):
            raise ValueError("cluster_group.tsv must have columns: 'cluster_id', 'group'")
    else:
        # initialize as 'unsorted' for all clusters listed in mapping
        cg = pd.DataFrame({"cluster_id": mapping["cluster_id"], "group": "unsorted"})

    cg["cluster_id"] = pd.to_numeric(cg["cluster_id"], errors="coerce").astype("Int64")
    cg = cg.set_index("cluster_id")

    # --- 5) Reset all to unsorted if requested, then apply noise labels
    if reset_to_unsorted:
        cg["group"] = "unsorted"
    else:
        cg.loc[df.index, "group"] = cg.loc[df.index, "group"].fillna("unsorted")
    
    cg.loc[noise_mask.index[noise_mask], "group"] = "noise"

    # --- 6) Save (with backup)
    out_df = cg.reset_index().sort_values("cluster_id")
    if backup and cg_path.exists():
        bak = cg_path.with_suffix(".bak.tsv")
        shutil.copyfile(cg_path, bak)
    out_df.to_csv(cg_path, sep="\t", index=False)

    # --- 7) Update cluster_info.tsv if requested
    if update_cluster_info:
        # Merge all metrics with updated group labels
        df_with_group = df.drop(columns=["_amp_abs_"], errors="ignore")
        df_with_group = df_with_group.join(cg[["group"]], how="left")
        df_with_group["group"] = df_with_group["group"].fillna("unsorted")
        
        info_path = phy_dir / "cluster_info.tsv"
        info_df = df_with_group.reset_index().sort_values("cluster_id")
        
        if backup and info_path.exists():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = info_path.with_stem(f"{info_path.stem}_{timestamp}_bak")
            shutil.copyfile(info_path, bak)
        
        info_df.to_csv(info_path, sep="\t", index=False)

    return out_df


def merge_individual_metric_tsvs(
    phy_dir: str | Path,
    backup: bool = True,
) -> pd.DataFrame:
    """
    Merge individual metric TSV files (cluster_snr.tsv, cluster_isi_violations_ratio.tsv, etc.)
    into a single unified `cluster_info.tsv` file.

    This is useful when you have Phy output folders with individual metric files generated by
    SpikeInterface exporters and want to consolidate them.

    Parameters
    ----------
    phy_dir : str | Path
        Phy output folder containing individual cluster_*.tsv metric files.
    backup : bool
        If True, make a timestamped backup of the original cluster_info.tsv before overwriting.

    Returns
    -------
    pd.DataFrame
        The merged cluster_info.tsv contents as a DataFrame sorted by cluster_id.
    """
    phy_dir = Path(phy_dir)
    
    # Find all cluster_*.tsv files except special ones
    exclude_files = {"cluster_group.tsv", "cluster_si_unit_ids.tsv", "cluster_info.tsv"}
    metric_files = [f for f in phy_dir.glob("cluster_*.tsv") if f.name not in exclude_files]
    
    if not metric_files:
        raise FileNotFoundError(f"No individual metric TSV files found in {phy_dir}")
    
    # Start with cluster_si_unit_ids if available, otherwise use first metric file
    map_path = phy_dir / "cluster_si_unit_ids.tsv"
    if map_path.exists():
        df = pd.read_csv(map_path, sep="\t")
    else:
        df = pd.read_csv(metric_files[0], sep="\t")
        metric_files = metric_files[1:]
    
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    
    # Merge all metric files
    for metric_file in metric_files:
        metric_name = metric_file.stem.replace("cluster_", "")
        metric_df = pd.read_csv(metric_file, sep="\t")
        metric_df["cluster_id"] = pd.to_numeric(metric_df["cluster_id"], errors="coerce").astype("Int64")
        df = df.merge(metric_df, on="cluster_id", how="outer")
    
    # Add group column from cluster_group.tsv if it exists
    cg_path = phy_dir / "cluster_group.tsv"
    if cg_path.exists():
        cg = pd.read_csv(cg_path, sep="\t")
        cg["cluster_id"] = pd.to_numeric(cg["cluster_id"], errors="coerce").astype("Int64")
        df = df.merge(cg, on="cluster_id", how="left")
        if "group" in df.columns:
            df["group"] = df["group"].fillna("unsorted")
    else:
        df["group"] = "unsorted"
    
    # Sort and save
    df = df.sort_values("cluster_id")
    
    info_path = phy_dir / "cluster_info.tsv"
    if backup and info_path.exists():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = info_path.with_stem(f"{info_path.stem}_{timestamp}_bak")
        shutil.copyfile(info_path, bak)
    
    df.to_csv(info_path, sep="\t", index=False)
    
    return df
