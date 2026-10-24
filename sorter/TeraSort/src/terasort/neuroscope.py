"""Read Neuroscope acquisition metadata without changing recording folders."""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def read_xml(path, gain_uv_per_count=None):
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as handle:
        payload = handle.read(4*1024**2+1)
    if len(payload) > 4*1024**2:
        raise ValueError("XML metadata exceeds 4 MiB")
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("XML document types and entities are not supported")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid Neuroscope XML: {exc}") from exc
    if root.tag != "parameters":
        raise ValueError("Expected Neuroscope parameters XML, not acquisition settings.xml")
    try:
        count = int(root.findtext("acquisitionSystem/nChannels"))
        rate = float(root.findtext("acquisitionSystem/samplingRate"))
        bits = int(root.findtext("acquisitionSystem/nBits"))
        offset = float(root.findtext("acquisitionSystem/offset", "0"))
    except (TypeError, ValueError) as exc:
        raise ValueError("XML requires nChannels, samplingRate and nBits") from exc
    if not 1 <= count <= 1000000 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid XML channel count or sample rate")
    if bits != 16 or offset != 0:
        raise ValueError("This pathway requires signed INT16 recordings with zero offset")

    def groups_at(location):
        result, seen, skipped = [], set(), set()
        for group in root.findall(location):
            channels = []
            for item in group.findall("channel") + group.findall("channels/channel"):
                try:
                    channel = int(item.text)
                except (TypeError, ValueError) as exc:
                    raise ValueError("XML channels must be zero-based integers") from exc
                if not 0 <= channel < count or channel in seen:
                    raise ValueError("XML channels are out of range or duplicated across groups")
                skip = item.get("skip", "0").lower()
                if skip not in ("0", "1", "false", "true"):
                    raise ValueError("Invalid XML channel skip flag")
                seen.add(channel)
                channels.append(channel)
                if skip in ("1", "true"):
                    skipped.add(channel)
            if not channels:
                raise ValueError("XML contains an empty channel group")
            result.append(channels)
        return result, seen, skipped

    anatomy, anatomical_channels, bad = groups_at("anatomicalDescription/channelGroups/group")
    sorting, sorting_channels, sorting_bad = groups_at("spikeDetection/channelGroups/group")
    groups = sorting or anatomy
    active = sorting_channels if sorting else anatomical_channels
    if not groups:
        raise ValueError("XML requires anatomical or spike-detection channel groups")
    excluded = bad | sorting_bad | (set(range(count)) - active)
    if len(excluded) == count:
        raise ValueError("XML excludes every recording channel")
    settings = dict(n_chan_bin=count, fs=rate)
    warnings = ["Automatic probe uses the MATLAB wrapper's staggered layout convention; XML supplies channel order, not measured contact spacing."]
    if gain_uv_per_count is not None:
        gain = float(gain_uv_per_count)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("Gain must be positive microvolts per ADC count")
        settings["scale"] = gain
    else:
        warnings.append("Gain is unspecified: retain ADC units. XML voltageRange/amplification are not assumed to describe the ADC conversion.")
    candidates = []
    for name in (path.with_suffix(".dat"), path.parent / "amplifier.dat"):
        if name.is_file() and str(name) not in candidates:
            candidates.append(str(name))
    return dict(xml_path=str(path), xml_sha256=hashlib.sha256(payload).hexdigest(),
                settings=settings, channel_groups=groups, anatomical_groups=anatomy,
                bad_channels=sorted(excluded), recording_candidates=candidates,
                lfp_sampling_rate_hz=root.findtext("fieldPotentials/lfpSamplingRate"),
                warnings=warnings)


def configure_probe(metadata, probe):
    """Preserve supplied coordinates while applying XML groups and exclusions."""
    import numpy as np
    if not isinstance(probe, dict):
        raise ValueError("Probe JSON must contain a probe dictionary")
    channels = np.asarray(probe.get("chanMap", []))
    x, y = np.asarray(probe.get("xc", []), dtype=float), np.asarray(probe.get("yc", []), dtype=float)
    n = metadata["settings"]["n_chan_bin"]
    if (channels.ndim != 1 or not len(channels) or channels.dtype.kind not in "iu"
            or len(np.unique(channels)) != len(channels) or np.any(channels < 0)
            or np.any(channels >= n) or x.shape != channels.shape or y.shape != channels.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Probe JSON requires unique acquisition-order chanMap and finite xc/yc coordinates")
    groups = {ch:i for i, group in enumerate(metadata["channel_groups"]) for ch in group}
    keep = np.array([int(ch) not in metadata["bad_channels"] for ch in channels])
    if not keep.any():
        raise ValueError("No usable probe contacts remain after XML exclusions")
    return dict(chanMap=channels[keep].tolist(), xc=x[keep].tolist(), yc=y[keep].tolist(),
                kcoords=[groups[int(ch)] for ch in channels[keep]], n_chan=int(keep.sum()))


def generate_probe(metadata):
    """Generate the wrapper's default staggered map from XML anatomical order.

    Coordinates follow createChannelMapFile_KSW's staggered convention. Keep
    excluded slots in the layout before filtering, and preserve ADC indices.
    """
    groups = metadata['anatomical_groups'] or metadata['channel_groups']
    positions = {}
    for group_id, group in enumerate(groups):
        for slot, channel in enumerate(group):
            positions[channel] = (200*(group_id+1) + (-20 if slot % 2 == 0 else 20),
                                  -20*(slot+1), group_id)
    channels = sorted(set(range(metadata['settings']['n_chan_bin'])) - set(metadata['bad_channels']))
    if any(ch not in positions for ch in channels):
        raise ValueError('Sorting channels are missing from XML anatomical groups')
    return dict(chanMap=channels, xc=[positions[ch][0] for ch in channels],
                yc=[positions[ch][1] for ch in channels],
                kcoords=[positions[ch][2] for ch in channels], n_chan=len(channels))
