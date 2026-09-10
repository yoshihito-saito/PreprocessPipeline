from pathlib import Path

import pytest

from src.preprocess import sorter_runner as sr


@pytest.mark.parametrize("preprocessed", [False, True])
def test_partition_does_not_exclude_original_binary_channels_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preprocessed: bool,
) -> None:
    config = tmp_path / "sorter.yaml"
    config.write_text("n_jobs: 1\nauto_geom_from_probe: false\n")
    dat = tmp_path / "input.dat"
    dat.write_bytes(bytes(128 * 2 * 10))
    active = list(range(0, 128, 4))
    full_recording, selected_recording = object(), object()
    monkeypatch.setattr(sr, "_get_kilosort4_allowed_param_keys", lambda: {"n_jobs"})
    monkeypatch.setattr(sr.se, "read_binary", lambda *a, **k: full_recording)

    def select(recording, channels):
        assert recording is full_recording
        assert channels == active
        return selected_recording

    monkeypatch.setattr(sr, "select_recording_channels", select)
    called = []

    def run_sorter(**kwargs):
        assert kwargs["recording"] is selected_recording
        # SpikeInterface exports a compact chanMap for this recording. Original
        # binary exclusions must not be applied to that compact channel space.
        assert not kwargs.get("bad_channels")
        called.append(True)
        return object()

    monkeypatch.setattr(sr.ss, "run_sorter", run_sorter)
    sr.execute_sorting_job(
        sorter="kilosort4", dat_path=dat, xml_path=None,
        config_path=config, output_folder=tmp_path / "out",
        sampling_frequency=20000.0, num_channels=128,
        active_channels_0based=active, input_is_preprocessed=preprocessed,
        preprocess_for_sorting=False,
    )
    assert called == [True]
