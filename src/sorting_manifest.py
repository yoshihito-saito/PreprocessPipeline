"""Interpret explicit no-work sorter manifests without discovering old runs."""
from pathlib import Path
import json


def all_partitions_skipped(path: Path | None) -> bool:
    if path is None or not Path(path).is_file():
        return False
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    partitions = payload.get("partitions")
    if isinstance(partitions, list):
        for item in partitions:
            if isinstance(item, dict) and item.get("status") == "skipped" and (
                item.get("skip_reason") != "no_active_channels"
                or item.get("channels_0based") != [] or item.get("output_folder")
            ):
                raise ValueError(f"Malformed skipped sorter partition in {path}")
    return bool(isinstance(partitions, list) and partitions and all(
        isinstance(item, dict) and item.get("status") == "skipped"
        and item.get("skip_reason") == "no_active_channels"
        and item.get("channels_0based") == [] and not item.get("output_folder")
        for item in partitions
    ))
