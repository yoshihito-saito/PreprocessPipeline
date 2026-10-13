from pathlib import Path
from types import SimpleNamespace
import errno

import pytest

from src.preprocess import disk_space as ds
from src.preprocess.metafile import PreprocessConfig


def usage(monkeypatch, free):
    monkeypatch.setattr(ds.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))


def catalog(samples=1000):
    return SimpleNamespace(
        sample_counts=[samples], analog_sample_counts_by_subsession=[],
        board_adc_channels=2, board_digital_word_channels=1,
        auxiliary_paths=[], supply_paths=[], time_paths=[],
    )


def estimate(tmp_path, config, source=None, sorter=None):
    return ds.preprocess_space_components(
        config, source or catalog(), output_dir=tmp_path, basename="session",
        n_channels=4, sampling_frequency=20000, analog_sampling_frequency=20000,
        sorter=sorter,
    )


def test_rejects_shortfall_with_destination_and_action(tmp_path, monkeypatch):
    usage(monkeypatch, ds.GIB)
    target = tmp_path / "new" / "session"
    with pytest.raises(OSError) as caught:
        ds.require_disk_space(target, {"temporary data": 10 * ds.GIB}, stage="Sorting")
    assert caught.value.errno == errno.ENOSPC
    message = str(caught.value)
    assert str(target) in message
    assert "15.00 GiB" in message
    assert "available 1.00 GiB" in message
    assert "Free at least 14.00 GiB more" in message
    assert not target.exists()


def test_checks_destination_ancestor_and_allows_exact_boundary(tmp_path, monkeypatch, capsys):
    seen = []
    def disk_usage(path):
        seen.append(path)
        return SimpleNamespace(free=15 * ds.GIB)
    monkeypatch.setattr(ds.shutil, "disk_usage", disk_usage)
    ds.require_disk_space(tmp_path / "new", {"recording": 10 * ds.GIB}, stage="Preprocessing")
    assert seen == [tmp_path.resolve()]
    assert "Preprocessing disk-space check" in capsys.readouterr().out


def test_large_recording_has_proportional_headroom(tmp_path, monkeypatch):
    usage(monkeypatch, 119 * ds.GIB)
    with pytest.raises(OSError, match="120.00 GiB"):
        ds.require_disk_space(tmp_path, {"data": 100 * ds.GIB}, stage="Sorting")


def test_empty_work_needs_no_headroom(tmp_path, monkeypatch):
    usage(monkeypatch, 0)
    ds.require_disk_space(tmp_path, {}, stage="Preprocessing")


def test_usage_errors_are_not_silently_ignored(tmp_path, monkeypatch):
    def denied(path):
        raise PermissionError("cannot inspect destination")
    monkeypatch.setattr(ds.shutil, "disk_usage", denied)
    with pytest.raises(PermissionError, match="cannot inspect"):
        ds.require_disk_space(tmp_path, {"data": 10}, stage="Sorting")


def test_accounts_for_enabled_outputs_and_sorter(tmp_path):
    config = PreprocessConfig(tmp_path, save_raw=True, state_score=True)
    result = estimate(tmp_path, config, sorter="Kilosort")
    assert result["processed recording"] == 8000
    assert result["raw recording"] == 8000
    assert result["LFP"] == 63 * 4 * 2
    assert result["sleep scoring"] == 4 * 63 * 4 * 2
    assert result["analog sidecar"] == 4000
    assert result["digital sidecar"] == 2000
    assert result["sorter recording and temporary data"] == 16000


def test_resume_skips_existing_outputs_but_overwrite_budgets_new_copies(tmp_path):
    config = PreprocessConfig(tmp_path, export_intermediate_dat=False)
    for filename in ("session.dat", "session.lfp"):
        (tmp_path / filename).write_bytes(b"existing")
    assert estimate(tmp_path, config) == {}
    # Current free space already accounts for these files. Their bytes must
    # not be subtracted from the new atomic copies required by overwrite.
    config.overwrite = True
    assert estimate(tmp_path, config)["processed recording"] == 8000
    assert estimate(tmp_path, config)["LFP"] == 504


def test_selected_samples_and_dtype_determine_estimate(tmp_path):
    config = PreprocessConfig(tmp_path, dtype="float32", make_lfp=False, export_intermediate_dat=False)
    source = catalog()
    source.sample_counts = [10, 20]
    assert estimate(tmp_path, config, source) == {"processed recording": 30 * 4 * 4}


def test_sorter_rejects_before_matlab_or_output_creation(tmp_path, monkeypatch):
    from src.preprocess import sorter_runner as sr
    usage(monkeypatch, 0)
    dat = tmp_path / "session.dat"
    dat.write_bytes(bytes(4 * 2 * 10))
    def unexpected(*args, **kwargs):
        pytest.fail("MATLAB must not be resolved on a full destination volume")
    monkeypatch.setattr(sr, "_resolve_matlab_cmd", unexpected)
    with pytest.raises(OSError, match="Insufficient disk space"):
        sr.execute_sorting_job(
            sorter="kilosort", dat_path=dat, xml_path=None,
            output_folder=tmp_path / "sorter", sampling_frequency=20000,
            num_channels=4,
        )
    assert not (tmp_path / "sorter").exists()


def test_persistent_preprocessing_keeps_downstream_sorter_in_estimate(tmp_path, monkeypatch):
    from src.execution import worker
    from src.preprocess import runtime_prep
    from src.execution import session
    import src.preprocess as preprocess
    config = PreprocessConfig(tmp_path, sorter="Kilosort")
    settings = SimpleNamespace(
        preprocess=SimpleNamespace(overwrite=False), to_preprocess_config=lambda: config,
        resolved_chanmap_path=lambda: None, chanmap_path="",
    )
    monkeypatch.setattr(worker, "_settings_for_run", lambda store: settings)
    monkeypatch.setattr(session, "prepare_preprocess_output_contract", lambda *a, **k: tmp_path / "contract")
    monkeypatch.setattr(runtime_prep, "prepare_preprocess_settings", lambda settings: {})
    seen = []
    class StopHere(Exception):
        pass
    def run(config, *, sorter_for_disk_check):
        seen.append((config.sorter, sorter_for_disk_check))
        raise StopHere
    monkeypatch.setattr(preprocess, "run_preprocess_session", run)
    store = SimpleNamespace(load_run=lambda: {"enabled_stages": ["preprocess", "sorting", "postprocess"]})
    spec = SimpleNamespace(resources=SimpleNamespace(cpus=1))
    with pytest.raises(StopHere):
        worker._run_preprocess(store, spec)
    assert seen == [(None, "Kilosort")]


def test_preprocessing_rejects_before_mergepoints_or_recording_writes(tmp_path, monkeypatch):
    from src.preprocess import pipeline
    usage(monkeypatch, 0)
    source = catalog()
    source.amplifier_paths = [tmp_path / "amplifier.dat"]
    source.source_type = "intan"
    source.sampling_frequency = None
    monkeypatch.setattr(pipeline, "resolve_basepath_and_basename", lambda path: (tmp_path, "session"))
    monkeypatch.setattr(pipeline, "resolve_local_output_dir", lambda *a: tmp_path)
    monkeypatch.setattr(pipeline, "ensure_xml", lambda *a, **k: tmp_path / "session.xml")
    monkeypatch.setattr(pipeline, "ensure_rhd", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "load_xml_metadata", lambda *a: SimpleNamespace(n_channels=4, sr=20000))
    monkeypatch.setattr(pipeline, "load_session_xml_metadata", lambda *a: None)
    monkeypatch.setattr(pipeline, "discover_subsessions", lambda **k: source.amplifier_paths)
    monkeypatch.setattr(pipeline, "build_acquisition_catalog", lambda **k: source)
    monkeypatch.setattr(pipeline, "print_catalog_summary", lambda *a: None)
    monkeypatch.setattr(pipeline, "_catalog_openephys_ttl_inputs", lambda *a: (None, None))
    def unexpected(*a, **k):
        pytest.fail("low space must stop before bulk output work")
    monkeypatch.setattr(pipeline, "compute_mergepoints", unexpected)
    with pytest.raises(OSError, match="Insufficient disk space"):
        pipeline.run_preprocess_session(PreprocessConfig(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_partition_plan_rejects_before_first_sorter(tmp_path, monkeypatch):
    from src.preprocess import sorting_stage
    usage(monkeypatch, 0)
    dat = tmp_path / "session.dat"
    dat.write_bytes(bytes(4 * 2 * 10))
    result = SimpleNamespace(dat_path=dat, local_output_dir=tmp_path, n_channels=4, bad_channels_0based=[])
    config = PreprocessConfig(tmp_path, sorter="Kilosort", sorter_partition_mode="probe")
    monkeypatch.setattr(sorting_stage, "_resolved_xml_path", lambda *a: None)
    monkeypatch.setattr(sorting_stage, "_resolved_chanmap_path", lambda *a: None)
    monkeypatch.setattr(sorting_stage, "build_sorter_partitions", lambda **k: [SimpleNamespace(status="pending")])
    def unexpected(**kwargs):
        pytest.fail("partitioned run must check before its first sorter")
    with pytest.raises(OSError, match="Sorting partitions"):
        sorting_stage.run_sorting_stage(config, result, execute_job=unexpected)
    assert list(tmp_path.iterdir()) == [dat]
