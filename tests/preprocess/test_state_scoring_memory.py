from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.preprocess import state_scoring as scoring


@pytest.mark.parametrize("parallel_jobs", [1, 2, 128])
def test_candidate_streaming_preserves_samples_order_and_histograms(tmp_path, monkeypatch, parallel_jobs):
    # Non-divisible length and int16 extremes also exercise the retained last sample.
    data = np.random.default_rng(42).integers(-32768, 32768, (103, 6), dtype=np.int16)
    path = tmp_path / "test.lfp"
    data.tofile(path)
    candidates = np.array([6, 2, 4, 1, 3])
    old_subset = data[:, np.unique(candidates) - 1]
    expected = scoring._matlab_downsample(old_subset.astype(np.float64), 5)
    expected_by_channel = dict(zip(np.unique(candidates), expected.T))
    values = np.array([np.nan, -1.0, 0.0, 0.5, 1.0, 2.0, np.inf])
    captured = {}
    worker_counts = []

    def executor(*, max_workers):
        worker_counts.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    def check_samples(trace):
        matches = [ch for ch, ref in expected_by_channel.items() if np.array_equal(trace, ref)]
        assert len(matches) == 1
        assert trace.dtype == np.float64
        assert trace.shape == (21,)
        return matches[0]

    def sw_eval(trace, *args, **kwargs):
        check_samples(trace)
        return 1.0, values.copy()  # Equal scores must retain user candidate order.

    def th_eval(trace, *args, **kwargs):
        ch = check_samples(trace)
        return 1.0, values.copy(), np.array([ch, ch + 1.0]), np.array([2.0, 3.0])

    def plot_payload(**kwargs):
        check_samples(kwargs["lfp_ch"])
        return {}

    def save_figure(**kwargs):
        captured.update(kwargs)
        return tmp_path / "figure.jpg"

    def forbid_bulk_load(*args, **kwargs):
        pytest.fail("Candidate evaluation must not load all channels together")

    monkeypatch.setattr(scoring, "ThreadPoolExecutor", executor)
    monkeypatch.setattr(scoring, "_load_binary_lfp_channels", forbid_bulk_load)
    monkeypatch.setattr(scoring, "_score_sw_candidate", sw_eval)
    monkeypatch.setattr(scoring, "_score_theta_candidate", th_eval)
    monkeypatch.setattr(scoring, "_build_sw_plot_payload", plot_payload)
    monkeypatch.setattr(scoring, "_build_th_plot_payload", plot_payload)
    monkeypatch.setattr(scoring, "_save_swth_figure", save_figure)
    result, _, _ = scoring._compute_sleepscore_lfp(
        basepath=tmp_path, basename="test", lfp_path=path,
        session_struct={"extracellular": {"nChannels": 6, "srLfp": 1250}},
        reject_channels_1based=np.array([], dtype=int),
        sw_channels_1based=candidates, th_channels_1based=candidates,
        ignoretime=np.empty((0, 2)), window_sec=10.0, smoothfact=10.0,
        overwrite=True, save_files=False, parallel_jobs=parallel_jobs,
    )
    assert int(result["SWchanID"].item()) == 6
    assert int(result["THchanID"].item()) == 6
    for key in ("swLFP", "thLFP"):
        np.testing.assert_array_equal(result[key], data[:, 5:6])
        assert result[key].dtype == np.int16
        assert result[key].base.nbytes == data.shape[0] * data.dtype.itemsize
    np.testing.assert_array_equal(result["t"], (np.arange(103) / 1250).reshape(1, -1))
    counts, _ = np.histogram(np.clip(values[np.isfinite(values)], 0, 1), bins=np.linspace(0, 1, 22))
    np.testing.assert_array_equal(captured["swhists"], np.tile(counts[:, None], (1, 5)))
    np.testing.assert_array_equal(captured["th_meanspec"], [candidates, candidates + 1])
    np.testing.assert_array_equal(captured["sw_order"], np.argsort(np.ones(5)))
    assert worker_counts == ([] if parallel_jobs == 1 else [min(parallel_jobs, 4)] * 2)
