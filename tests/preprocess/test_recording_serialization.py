"""Exercise custom recording reconstruction used by process workers."""

import pickle

import numpy as np
from scipy.io import savemat
from spikeinterface.core import BaseRecording, NumpyRecording

from src.preprocess.artifact_removal import detect_high_amplitude_artifacts
from src.preprocess.recording import _LocalMedianReference3D, attach_probe_from_chanmap


def _referenced_recording(tmp_path):
    traces = np.random.default_rng(7).normal(size=(1000, 6)).astype("float32")
    traces[350, :3] += 1000
    parent = NumpyRecording(traces, sampling_frequency=1000.0)
    parent.set_channel_gains(np.ones(6))
    parent.set_channel_offsets(np.zeros(6))
    chanmap = tmp_path / "chanMap.mat"
    savemat(chanmap, {
        "chanMap": np.arange(1, 7), "chanMap0ind": np.arange(6),
        "xcoords": [0, 0, 20, 200, 200, 220],
        "ycoords": [0, 0, 20, 0, 0, 20],
        "zcoords": [0, 25, 0, 0, 25, 0],
        "kcoords": [1, 1, 1, 2, 2, 2],
        "connected": np.ones(6, dtype=bool),
    })
    attached = attach_probe_from_chanmap(parent, chanmap)
    attached.set_property("group", [1, 1, 1, 2, 2, 2])
    return _LocalMedianReference3D(attached, (0, 80))


def test_custom_3d_reference_dictionary_and_pickle_roundtrip(tmp_path):
    recording = _referenced_recording(tmp_path)
    restored = [
        BaseRecording.from_dict(recording.to_dict(recursive=True)),
        pickle.loads(pickle.dumps(recording)),
    ]
    for copy in restored:
        np.testing.assert_array_equal(copy.get_traces(), recording.get_traces())
        np.testing.assert_array_equal(copy.channel_ids, recording.channel_ids)
        np.testing.assert_array_equal(copy.get_channel_locations(axes="xyz"),
                                      recording.get_channel_locations(axes="xyz"))
        np.testing.assert_array_equal(copy.get_property("group"), recording.get_property("group"))


def test_highamp_loky_workers_match_serial_with_3d_reference(tmp_path):
    recording = _referenced_recording(tmp_path)
    kwargs = dict(by_group=True, estimate_windows=3, estimate_window_s=0.1,
                  threshold_sigma=10, seed=2, chunk_s=0.2, dead_time_ms=1)
    serial = detect_high_amplitude_artifacts(recording, n_jobs=1, **kwargs)
    parallel = detect_high_amplitude_artifacts(recording, n_jobs=2, **kwargs)
    assert serial.keys() == parallel.keys()
    assert any(350 in frames for frames in serial.values())
    for group in serial:
        np.testing.assert_array_equal(parallel[group], serial[group])
