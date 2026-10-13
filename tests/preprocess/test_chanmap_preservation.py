from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from src.execution.controller import create_run
from src.execution.models import BackendName, ExecutionConfig, RequestedBackend, ResourceSpec, StageName
from src.execution.session import PRODUCER_SCHEMAS, assert_preprocess_geometry, stage_fingerprints
from src.execution.store import RunStore
from src.execution.worker import _load_upstream_result, _verify_analysis_artifacts
from src.preprocess.channel_layout import channel_map_geometry_sha256
from src.preprocess.gui.config_model import PipelineGuiSettings
from src.preprocess.pipeline import _prepare_effective_chanmap_for_final_channel_space
from src.preprocess.runtime_prep import prepare_preprocess_settings


@pytest.fixture
def custom_session(tmp_path):
    raw = tmp_path / "session"
    raw.mkdir()
    xml = raw / "session.xml"
    xml.write_text(
        '<session><generalInfo><description/></generalInfo>'
        '<acquisitionSystem><nChannels>8</nChannels><samplingRate>20000</samplingRate></acquisitionSystem>'
        '<anatomicalDescription><channelGroups><group>'
        + ''.join(f'<channel skip="{int(ch == 1)}">{ch}</channel>' for ch in range(8))
        + '</group></channelGroups></anatomicalDescription></session>', encoding="utf-8",
    )
    ids = np.array([2, 0, 3, 1, 7, 5, 6, 4])
    payload = {
        "chanMap": (ids + 1).reshape(1, -1),
        "chanMap0ind": ids.reshape(1, -1),
        "xcoords": np.array([-18.5, 0, 18.5, 0, 50, 50, 68.5, 50]).reshape(1, -1),
        "ycoords": np.array([20, 0, 20, 1000, 20, 0, 20, 0]).reshape(-1, 1),
        "zcoords": np.array([7.5, 7.5, 7.5, 7.5, -15, 15, 15, -15]).reshape(1, -1),
        "kcoords": np.array([1] * 4 + [2] * 4).reshape(-1, 1),
        "probe_ids": np.array([1] * 4 + [2] * 4).reshape(1, -1),
        "sidecoords": np.array([1] * 4 + [0, 1, 1, 0]).reshape(-1, 1),
        "connected": (ids != 2).astype(float).reshape(1, -1),
        "geometry_name": "custom H20 and double-sided probe",
        "custom_contact_metadata": np.arange(8).reshape(1, -1),
    }
    source = raw / "chanMap.mat"
    savemat(source, payload)
    settings = PipelineGuiSettings(
        basepath=str(raw), xml_path=str(xml), local_root=str(tmp_path / "local"),
        chanmap_path=str(source),
    )
    settings.preprocess.run_sorter = False
    settings.preprocess.reject_channels = [5]
    return settings, source, payload


@pytest.mark.parametrize("explicit", [True, False])
def test_runtime_preserves_physical_map_and_only_adds_exclusions(custom_session, explicit):
    settings, source, payload = custom_session
    original = source.read_bytes()
    if explicit:
        # A stale template must never override a selected physical map.
        settings.preprocess.probe_assignments = [{"type": "poly2", "groups": [0], "x_offset": 0}]
    else:
        settings.chanmap_path = ""
    prepare_preprocess_settings(settings)
    result = loadmat(settings.chanmap_path)
    for key, value in payload.items():
        if key not in ("connected", "geometry_name"):
            np.testing.assert_array_equal(result[key], value)
    assert str(result["geometry_name"][0]) == payload["geometry_name"]
    ids = result["chanMap0ind"].reshape(-1)
    assert set(ids[result["connected"].reshape(-1) == 0]) == {1, 2, 5}
    assert source.read_bytes() == original
    assert "probe_assignments_json" not in result
    # Both references to the same x/y point survive as distinct 3D contacts.
    assert len(np.unique(result["zcoords"])) == 3
    prepare_preprocess_settings(settings)  # overwrite=False, same geometry/exclusions


def test_missing_selected_map_fails_without_generating_a_substitute(custom_session):
    settings, source, _ = custom_session
    settings.chanmap_path = str(source.with_name("missing.mat"))
    with pytest.raises(FileNotFoundError, match="Selected chanMap"):
        prepare_preprocess_settings(settings)
    assert not settings.local_output_dir.exists()


def test_explicit_geometry_edits_still_generate_templates(custom_session):
    settings, source, _ = custom_session
    original = source.read_bytes()
    settings.chanmap_path = ""
    settings.preprocess.probe_assignments = [{"type": "poly3", "groups": [0], "x_offset": 0}]
    prepare_preprocess_settings(settings)
    result = loadmat(settings.chanmap_path)
    assert len(np.unique(result["xcoords"])) == 3
    assert "poly3" in str(result["probe_assignments_json"])
    assert source.read_bytes() == original


def test_corrupted_existing_map_is_rejected_before_any_repair(custom_session):
    settings, source, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    wrong = dict(payload, xcoords=np.zeros((8, 1)))
    canonical = out / "chanMap.mat"
    savemat(canonical, wrong)
    dat = out / "session.dat"
    dat.write_bytes(b"already processed recording")
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match="replacing only chanMap.mat is insufficient"):
        prepare_preprocess_settings(settings)
    assert canonical.read_bytes() == before
    assert dat.read_bytes() == b"already processed recording"
    with pytest.raises(ValueError, match="geometry differs"):
        assert_preprocess_geometry(settings, out)


@pytest.mark.parametrize("field", ["zcoords", "sidecoords", "probe_ids", "kcoords"])
def test_cache_geometry_check_detects_lost_or_changed_identity(custom_session, field):
    settings, _, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    wrong = dict(payload)
    if field == "kcoords":
        wrong[field] = np.arange(1, 9).reshape(-1, 1)
    else:
        wrong.pop(field)
    savemat(out / "chanMap.mat", wrong)
    with pytest.raises(ValueError, match=field):
        assert_preprocess_geometry(settings, out)


def test_direct_pipeline_does_not_overwrite_wrong_cached_geometry(custom_session):
    settings, source, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    canonical = out / "chanMap.mat"
    savemat(canonical, dict(payload, xcoords=np.zeros((8, 1))))
    original = canonical.read_bytes()
    with pytest.raises(ValueError, match="geometry differs"):
        _prepare_effective_chanmap_for_final_channel_space(
            chanmap_mat_path=source, output_dir=out, final_num_channels=8,
            ephys_channel_indices_by_subsession=None, overwrite=False,
        )
    assert canonical.read_bytes() == original


def test_direct_pipeline_refuses_stale_exclusions(custom_session):
    settings, source, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    savemat(out / "chanMap.mat", dict(payload, connected=np.ones((8, 1))))
    with pytest.raises(ValueError, match="exclusions changed"):
        _prepare_effective_chanmap_for_final_channel_space(
            chanmap_mat_path=source, output_dir=out, final_num_channels=8,
            ephys_channel_indices_by_subsession=None, overwrite=False,
        )


def test_fresh_staged_map_can_remap_acquisition_ids_without_moving_contacts(custom_session):
    settings, _, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    source = out / "chanMap.mat"
    shifted = dict(payload, chanMap=payload["chanMap"] + 10, chanMap0ind=payload["chanMap0ind"] + 10)
    savemat(source, shifted)
    result, mapping = _prepare_effective_chanmap_for_final_channel_space(
        chanmap_mat_path=source, output_dir=out, final_num_channels=8,
        ephys_channel_indices_by_subsession=[list(range(10, 18))], overwrite=False,
    )
    assert mapping == {ch: ch - 10 for ch in range(10, 18)}
    assert channel_map_geometry_sha256(loadmat(result)) == channel_map_geometry_sha256(payload)
    coordinates = loadmat(out / "session.chanCoords.channelInfo.mat", simplify_cells=True)["chanCoords"]
    order = np.argsort(payload["chanMap0ind"].reshape(-1))
    for key in ("x", "y", "z"):
        np.testing.assert_array_equal(coordinates[key], payload[key + "coords"].reshape(-1)[order])


def test_sorting_rejects_a_cached_map_that_disagrees_with_selected_geometry(custom_session):
    from src.preprocess.sorting_stage import _resolved_chanmap_path

    settings, _, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    savemat(out / "chanMap.mat", dict(payload, xcoords=np.zeros((8, 1))))
    with pytest.raises(ValueError, match="Sorting input recording"):
        _resolved_chanmap_path(settings.to_preprocess_config(), SimpleNamespace(local_output_dir=out))


def test_postprocessing_rejects_wrong_geometry_for_a_processed_recording(custom_session):
    from src.postprocess.pipeline import _resolve_effective_chanmap_for_postprocess

    settings, source, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    savemat(out / "chanMap.mat", dict(payload, xcoords=np.zeros((8, 1))))
    (out / ".preprocess-output-contract.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="Postprocessing input recording"):
        _resolve_effective_chanmap_for_postprocess(source, dat_path=out / "session.dat")


def test_probe_attachment_failure_cannot_fall_back_to_dummy_geometry(custom_session, monkeypatch):
    from spikeinterface.core import NumpyRecording
    from src.preprocess.recording import attach_probe_from_chanmap

    _, source, _ = custom_session
    recording = NumpyRecording(np.zeros((100, 8), dtype="int16"), 20000)

    def fail(*_args, **_kwargs):
        raise RuntimeError("probe attachment rejected")

    monkeypatch.setattr(recording, "set_probegroup", fail)
    with pytest.raises(ValueError, match="Cannot retain selected chanMap geometry"):
        attach_probe_from_chanmap(recording, source)


def test_zero_based_probe_labels_retain_all_physical_contacts(custom_session):
    from spikeinterface.core import NumpyRecording
    from src.preprocess.recording import attach_probe_from_chanmap, analysis_channel_locations

    _, source, payload = custom_session
    adjusted = dict(payload, probe_ids=payload["probe_ids"] - 1)
    savemat(source, adjusted)
    recording = NumpyRecording(np.zeros((100, 8), dtype="int16"), 20000)
    attached = attach_probe_from_chanmap(recording, source)
    expected = np.column_stack([np.asarray(payload[key]).reshape(-1) for key in ("xcoords", "ycoords", "zcoords")])
    order = np.argsort(payload["chanMap0ind"].reshape(-1))
    assert attached.get_num_channels() == 8
    np.testing.assert_array_equal(analysis_channel_locations(attached), expected[order])


def test_geometry_hash_aligns_channels_and_ignores_only_exclusions(custom_session):
    _, _, payload = custom_session
    expected = channel_map_geometry_sha256(payload)
    reordered = {key: np.asarray(value).reshape(-1)[::-1].reshape(-1, 1)
                 if key != "geometry_name" else value for key, value in payload.items()}
    reordered["connected"] = np.ones((8, 1))
    assert channel_map_geometry_sha256(reordered) == expected
    reordered["zcoords"][0] += 1
    assert channel_map_geometry_sha256(reordered) != expected


def test_run_freezes_map_and_includes_it_in_preprocess_fingerprint(custom_session):
    settings, source, payload = custom_session
    original = source.read_bytes()
    execution = ExecutionConfig(
        requested_backend=RequestedBackend.LOCAL, resolved_backend=BackendName.LOCAL,
        workspace=settings.local_root,
        resources={stage.value: ResourceSpec(1, 1024, 10, gpu_count=int(stage == StageName.SORTING)) for stage in StageName},
    )
    # This verifies run creation without launching scientific processing.
    run_dir = create_run(settings=settings, execution=execution, mode="preprocess")
    store = RunStore(run_dir)
    analysis = store.load_analysis()
    snapshot = Path(analysis.settings["chanmap_path"])
    assert snapshot.parent == run_dir / "snapshots"
    assert snapshot.read_bytes() == original
    assert analysis.artifact_sha256["chanmap"] == hashlib.sha256(original).hexdigest()
    changed = dict(analysis.to_dict())
    changed["artifact_sha256"] = dict(analysis.artifact_sha256, chanmap="changed-geometry")
    assert stage_fingerprints(analysis)["preprocess"] != stage_fingerprints(changed)["preprocess"]
    savemat(source, dict(payload, xcoords=np.zeros((8, 1))))
    _verify_analysis_artifacts(store, StageName.PREPROCESS)  # frozen selected geometry survives
    snapshot.write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="chanmap checksum changed"):
        _verify_analysis_artifacts(store, StageName.PREPROCESS)


def test_immutable_map_snapshot_location_does_not_change_acquisition_provenance(custom_session, tmp_path):
    from src.execution.controller import _input_provenance
    from src.execution.session import _input_snapshots_compatible

    settings, source, _ = custom_session
    prior = _input_provenance(settings)
    frozen = tmp_path / "snapshots" / "chanMap.input.mat"
    frozen.parent.mkdir()
    frozen.write_bytes(source.read_bytes())
    settings.chanmap_path = str(frozen)
    current = _input_provenance(settings)
    assert _input_snapshots_compatible(prior, current, settings)


def test_worker_rejects_old_preprocessing_contract_before_sorting(custom_session):
    settings, source, _ = custom_session
    result = {
        "status": "completed", "analysis_sha256": "test", "validation": {"passed": True},
        "producer_schema": "preprocess-output-v2",
        "outputs": {"chanmap_path": str(source)},
    }
    store = SimpleNamespace(read_attempt_fact=lambda *_args: result)
    spec = SimpleNamespace(stage=StageName.SORTING, upstream_attempts={"preprocess": 1}, analysis_sha256="test")
    with pytest.raises(RuntimeError, match="Regenerate preprocessing"):
        _load_upstream_result(store, spec, StageName.PREPROCESS)


def test_worker_rejects_map_replacement_after_preprocessing(custom_session):
    _, source, payload = custom_session
    result = {
        "status": "completed", "analysis_sha256": "test", "validation": {"passed": True},
        "producer_schema": PRODUCER_SCHEMAS["preprocess"],
        "outputs": {"chanmap_path": str(source), "output_inventory": {"metadata": {
            "channel_layout_geometry_sha256": channel_map_geometry_sha256(payload),
            "channel_layout_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }}},
    }
    store = SimpleNamespace(read_attempt_fact=lambda *_args: result)
    spec = SimpleNamespace(stage=StageName.POSTPROCESS, upstream_attempts={"preprocess": 1}, analysis_sha256="test")
    assert _load_upstream_result(store, spec, StageName.PREPROCESS) == result
    savemat(source, dict(payload, xcoords=np.zeros((8, 1))))
    with pytest.raises(RuntimeError, match="geometry changed"):
        _load_upstream_result(store, spec, StageName.PREPROCESS)


@pytest.mark.parametrize("field", ["xcoords", "zcoords", "sidecoords"])
def test_invalid_geometry_fails_before_output_publication(custom_session, field):
    settings, source, payload = custom_session
    broken = dict(payload)
    broken[field] = np.array([[np.nan]])
    savemat(source, broken)
    with pytest.raises(ValueError, match=field):
        prepare_preprocess_settings(settings)
    assert not settings.local_output_dir.exists()


def test_existing_2d_map_retains_its_coordinates_and_optional_metadata(custom_session):
    settings, source, payload = custom_session
    two_dimensional = {key: value for key, value in payload.items() if key != "zcoords"}
    savemat(source, two_dimensional)
    prepare_preprocess_settings(settings)
    result = loadmat(settings.chanmap_path)
    assert "zcoords" not in result
    for key in ("xcoords", "ycoords", "kcoords", "probe_ids", "sidecoords"):
        np.testing.assert_array_equal(result[key], two_dimensional[key])


def test_session_metadata_uses_physical_xyz_instead_of_poly2_default(custom_session, monkeypatch):
    from src.preprocess.io import load_session_xml_metadata
    from src.preprocess.metafile import MergePointsData
    from src.preprocess.session import build_session_struct

    settings, _, payload = custom_session
    prepare_preprocess_settings(settings)
    out = settings.local_output_dir
    dat = out / "session.dat"
    np.zeros((100, 8), dtype="int16").tofile(dat)
    merge = MergePointsData(
        timestamps_sec=np.array([[0, .005]]), timestamps_samples=np.array([[0, 100]]),
        firstlasttimepoints_samples=np.array([[0, 99]]), foldernames=["epoch"],
    )
    metadata = load_session_xml_metadata(Path(settings.xml_path))
    metadata.anatomical_groups_0based = [list(range(4)), list(range(4, 8))]
    monkeypatch.setattr(
        "src.preprocess.session._extract_kilosort_metadata",
        lambda *_args: ({"layout": "poly2", "verticalSpacing": 10}, 1),
    )
    session = build_session_struct(
        source_basepath=settings.basepath_path, local_basepath=out, session_basepath_mode="local",
        basename="session", dat_path=dat, sr=20000, sr_lfp=None, n_channels=8,
        bad_channels_1based=[2, 3, 6], merge_data=merge,
        xml_meta=metadata,
    )
    coords = session["extracellular"]["chanCoords"]
    assert coords["layout"] == "custom"
    assert set(np.asarray(session["channelTags"]["Bad"]["channels"]).reshape(-1)) == {2, 3, 6}
    order = np.argsort(payload["chanMap0ind"].reshape(-1))
    for key in ("x", "y", "z"):
        np.testing.assert_array_equal(coords[key], payload[key + "coords"].reshape(-1)[order])


def test_multiday_staging_preserves_the_map_selected_before_path_rewrite(custom_session, tmp_path, monkeypatch):
    settings, source, payload = custom_session
    second = tmp_path / "day2"
    second.mkdir()
    combined = tmp_path / "combined"
    combined.mkdir()
    (combined / "combined.xml").write_bytes(Path(settings.xml_path).read_bytes())
    staged = SimpleNamespace(
        server_basepath=combined, local_basepath=combined, name="combined",
        manifest_path=combined / "manifest.json", selected_subepochs_csv_path=combined / "epochs.csv",
        subepochs=["epoch1", "epoch2"],
    )
    monkeypatch.setattr("src.preprocess.runtime_prep.prepare_multi_day_basepath", lambda **_kwargs: staged)
    settings.multi_day_enabled = True
    settings.multi_day_session_paths = [settings.basepath, str(second)]
    settings.multi_day_name = "combined"
    prepare_preprocess_settings(settings)
    result = loadmat(settings.chanmap_path)
    assert channel_map_geometry_sha256(result) == channel_map_geometry_sha256(payload)
    np.testing.assert_array_equal(loadmat(source)["xcoords"], payload["xcoords"])


def test_gui_uses_source_map_and_bad_channel_edits_keep_geometry(custom_session, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from src.preprocess.gui.app import MainWindow

    settings, source, payload = custom_session
    monkeypatch.setattr("src.preprocess.gui.app._load_default_settings", lambda: PipelineGuiSettings())
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window._apply_settings(settings)
        assert window._apply_chanmap_file_to_controls(source)
        window.reject_channels.setText("1, 2, 5, 6")
        collected = window._collect_settings()
        assert collected.chanmap_path == str(source)
        assert channel_map_geometry_sha256(window._current_chanmap_data(collected)) == channel_map_geometry_sha256(payload)
        # A geometry edit is an explicit request to generate a new map.
        row = window._probe_rows[0]
        row["type"].setCurrentText("poly3")
        row["groups"].setText("0")
        edited = window._collect_settings()
        assert edited.chanmap_path == ""
        assert edited.resolved_chanmap_path() is None
        assert "zcoords" not in window._current_chanmap_data(edited)
    finally:
        window.close()
        application.processEvents()


def test_gui_automatic_selection_prefers_source_over_cached_template(custom_session, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from src.preprocess.gui.app import MainWindow

    settings, source, payload = custom_session
    settings.chanmap_path = ""
    out = settings.local_output_dir
    out.mkdir(parents=True)
    savemat(out / "chanMap.mat", dict(payload, xcoords=np.zeros((8, 1))))
    assert settings.resolved_chanmap_path() == source
    monkeypatch.setattr("src.preprocess.gui.app._load_default_settings", lambda: PipelineGuiSettings())
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window._apply_settings(settings)
        assert window._candidate_existing_chanmaps(include_explicit=False)[0] == source
        assert channel_map_geometry_sha256(window._current_chanmap_data(settings)) == channel_map_geometry_sha256(payload)
    finally:
        window.close()
        application.processEvents()


def test_gui_cannot_generate_over_a_map_owned_by_an_active_operation(custom_session, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from src.preprocess.gui.app import MainWindow
    from src.execution.session import acquire_manual_session_claim, release_manual_session_claim

    settings, _, payload = custom_session
    out = settings.local_output_dir
    out.mkdir(parents=True)
    canonical = out / "chanMap.mat"
    savemat(canonical, payload)
    original = canonical.read_bytes()
    token = acquire_manual_session_claim(session_dir=out, owner="active operation")
    monkeypatch.setattr("src.preprocess.gui.app._load_default_settings", lambda: PipelineGuiSettings())
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, _title, message: messages.append(message))
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window._apply_settings(settings)
        window._generate_chanmap()
        assert any("still owns this session" in message for message in messages)
        assert canonical.read_bytes() == original
    finally:
        release_manual_session_claim(session_dir=out, token=token)
        window.close()
        application.processEvents()
