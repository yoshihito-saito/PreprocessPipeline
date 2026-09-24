from pathlib import Path
import struct

import pytest

from src.preprocess.intan_rhd import read_intan_rhd_header
from src.preprocess.io import build_acquisition_catalog


def _qstring(value: str) -> bytes:
    data = value.encode("utf-16-le")
    return struct.pack("<I", len(data)) + data


def _recording(folder: Path, *, enabled=(True, True, True, True), samples=8) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<Ihhf", 0xC6912702, 3, 4, 20000.0) + bytes(36)
    header += _qstring("") * 3 + struct.pack("<hh", 0, 0) + _qstring("")
    header += struct.pack("<h", 1)
    header += _qstring("Port D") + _qstring("D") + struct.pack("<hhh", 1, len(enabled), len(enabled))
    for channel, active in enumerate(enabled):
        header += _qstring(f"D-{channel:03}") * 2
        header += struct.pack("<hhhhhh", channel, channel, 0, int(active), channel, 0) + bytes(16)
    (folder / "info.rhd").write_bytes(header)
    amplifier = folder / "amplifier.dat"
    amplifier.write_bytes(bytes(samples * sum(enabled) * 2))
    (folder / "time.dat").write_bytes(struct.pack(f"<{samples}i", *range(samples)))
    return amplifier


def _catalog(*paths, header=None):
    return build_acquisition_catalog(list(paths), n_amplifier_channels=4, dtype="int16", intan_header=header)


def test_valid_recordings_and_packed_digital_lines(tmp_path):
    first = _recording(tmp_path / "first")
    second = _recording(tmp_path / "second", samples=16)
    for path, samples in ((first, 8), (second, 16)):
        (path.parent / "digitalin.dat").write_bytes(b"\xff\xff" * samples)
        (path.parent / "digitalout.dat").write_bytes(b"\x01\x00" * samples)
    assert _catalog(first, second).sample_counts == [8, 16]


def test_disabled_channel_rejected_even_when_size_divides_xml_frame(tmp_path):
    good = _recording(tmp_path / "first")
    bad = _recording(tmp_path / "second", enabled=(True, False, True, True))
    assert bad.stat().st_size % (4 * 2) == 0
    # A valid root header must not hide a later epoch's disabled channel.
    root_header = read_intan_rhd_header(good.parent / "info.rhd")
    with pytest.raises(ValueError, match=r"second: XML expects 4.*enables 3.*D-001"):
        _catalog(good, bad, header=root_header)


def test_header_channel_count_checked_without_timestamps(tmp_path):
    path = _recording(tmp_path, enabled=(True, True, False, True))
    (tmp_path / "time.dat").unlink()
    with pytest.raises(ValueError, match="channel mismatch"):
        _catalog(path)


@pytest.mark.parametrize("sidecar,width", [("time.dat", 4), ("digitalin.dat", 2), ("digitalout.dat", 2)])
@pytest.mark.parametrize("size_delta", [-2, 2, 1])
def test_every_recording_sidecar_length_is_checked(tmp_path, sidecar, width, size_delta):
    good = _recording(tmp_path / "first")
    bad = _recording(tmp_path / "second")
    (bad.parent / sidecar).write_bytes(bytes(8 * width + size_delta))
    with pytest.raises(ValueError, match=rf"stream length mismatch.*second:.*{sidecar}"):
        _catalog(good, bad)


def test_short_amplifier_with_complete_frames_is_rejected(tmp_path):
    path = _recording(tmp_path)
    path.write_bytes(bytes(7 * 4 * 2))
    with pytest.raises(ValueError, match=r"amplifier.dat has 7 samples.*time.dat has 8"):
        _catalog(path)


def test_missing_header_warns_but_still_checks_timestamps(tmp_path):
    path = _recording(tmp_path)
    (tmp_path / "info.rhd").unlink()
    with pytest.warns(RuntimeWarning, match="cannot independently verify"):
        assert _catalog(path).sample_counts == [8]
    path.write_bytes(bytes(7 * 4 * 2))
    with pytest.warns(RuntimeWarning, match="No local RHD"):
        with pytest.raises(ValueError, match="stream length mismatch"):
            _catalog(path)


def test_unreadable_local_header_stops_processing(tmp_path):
    path = _recording(tmp_path)
    (tmp_path / "info.rhd").write_bytes(b"broken")
    with pytest.raises(ValueError, match="unreadable local RHD"):
        _catalog(path)


def test_sole_named_header_is_used_and_multiple_headers_are_rejected(tmp_path):
    path = _recording(tmp_path)
    original = tmp_path / "info.rhd"
    original.rename(tmp_path / "recording.rhd")
    assert _catalog(path).sample_counts == [8]
    (tmp_path / "another.rhd").write_bytes((tmp_path / "recording.rhd").read_bytes())
    with pytest.raises(ValueError, match="Ambiguous local Intan RHD"):
        _catalog(path)


@pytest.mark.parametrize("size", [0, 63])
def test_empty_or_incomplete_amplifier_frames_are_rejected(tmp_path, size):
    path = _recording(tmp_path)
    path.write_bytes(bytes(size))
    with pytest.raises(ValueError, match="Empty Intan|not divisible"):
        _catalog(path)


def test_failed_validation_does_not_modify_inputs(tmp_path):
    path = _recording(tmp_path, enabled=(True, False, True, True))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="channel mismatch"):
        _catalog(path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("failure", ["channels", "length"])
def test_pipeline_stops_before_any_event_or_binary_export(tmp_path, monkeypatch, failure):
    from unittest.mock import Mock
    from src.preprocess import pipeline
    from src.preprocess.metafile import PreprocessConfig

    session = tmp_path / "session"
    enabled = (True, False, True, True) if failure == "channels" else (True,) * 4
    path = _recording(session / "epoch", enabled=enabled)
    if failure == "length":
        path.write_bytes(bytes(7 * 4 * 2))
    (session / "session.xml").write_text(
        "<parameters><acquisitionSystem><nChannels>4</nChannels>"
        "<samplingRate>20000</samplingRate></acquisitionSystem></parameters>"
    )
    exports = []
    for name in ("materialize_intermediate_dat", "export_analog_digital_events", "load_subsession_recordings"):
        operation = Mock(side_effect=AssertionError("Export reached before validation"))
        monkeypatch.setattr(pipeline, name, operation)
        exports.append(operation)
    with pytest.raises(ValueError, match="channel mismatch|stream length mismatch"):
        pipeline.run_preprocess_session(PreprocessConfig(basepath=session, localpath=tmp_path / "outputs"))
    for operation in exports:
        operation.assert_not_called()
