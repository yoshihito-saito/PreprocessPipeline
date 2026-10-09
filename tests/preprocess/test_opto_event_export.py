from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from src.preprocess.gui.config_model import PipelineGuiSettings
from src.preprocess.io import atomic_savemat
from src.preprocess.pipeline import _build_opto_manipulation_struct_from_ttl


def test_opto_export_uses_selected_ttl_channel_and_pairs_pulses(tmp_path) -> None:
    on = np.empty((1, 16), dtype=object)
    off = np.empty((1, 16), dtype=object)
    for channel in range(16):
        on[0, channel] = np.empty((0, 1))
        off[0, channel] = np.empty((0, 1))
    on[0, 2] = np.array([[1.0], [3.0]])
    off[0, 2] = np.array([[1.2], [3.4]])
    on[0, 1] = np.array([[8.0]])

    source = tmp_path / "digitalIn.events.mat"
    atomic_savemat(source, {"digitalIn": {"timestampsOn": on, "timestampsOff": off}}, required_key="digitalIn")
    digital_in = loadmat(source, simplify_cells=True)["digitalIn"]
    opto = _build_opto_manipulation_struct_from_ttl(digital_in, ttl_channel_0based=2)

    target = tmp_path / "session.opto.manipulation.mat"
    atomic_savemat(target, {"opto": opto}, required_key="opto")
    saved = loadmat(target, simplify_cells=True)["opto"]
    np.testing.assert_allclose(saved["timestamps"], [[1.0, 1.2], [3.0, 3.4]])
    np.testing.assert_allclose(saved["duration"], [0.2, 0.4])
    assert saved["detectorinfo"]["channel"] == 3.0


def test_opto_export_setting_reaches_preprocess_config(tmp_path) -> None:
    settings = PipelineGuiSettings(basepath=str(tmp_path / "session"))
    settings.preprocess.export_opto_events = True
    settings.preprocess.artifact_ttl_channel = 4

    loaded = PipelineGuiSettings.from_json(settings.to_json())
    config = loaded.to_preprocess_config()

    assert config.export_opto_events is True
    assert config.artifact_TTL_channel == 4
