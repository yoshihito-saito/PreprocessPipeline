from pathlib import Path
import os

from src.execution.controller import _input_provenance
from src.execution.session import _input_snapshots_compatible
from src.preprocess.gui.config_model import PipelineGuiSettings



def test_xml_content_identity_allows_touch_but_rejects_same_size_edit(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    xml = source / "amplifier.xml"
    xml.write_text("<old/>")
    settings = PipelineGuiSettings(basepath=str(source), local_root=str(tmp_path / "local"))
    prior = _input_provenance(settings)
    stat = xml.stat()
    os.utime(xml, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    current = _input_provenance(settings)
    assert _input_snapshots_compatible(prior, current, settings)
    assert _input_snapshots_compatible(prior, prior, settings)
    xml.write_text("<new/>")
    os.utime(xml, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert not _input_snapshots_compatible(prior, _input_provenance(settings), settings)
    assert not _input_snapshots_compatible(prior, prior, settings)


def test_multiday_authoritative_xml_ignores_unused_legacy_amplifier_xml(tmp_path: Path) -> None:
    source = tmp_path / "Day117"
    source.mkdir()
    xml = source / "amplifier.xml"
    xml.write_text("<old/>")
    authoritative = tmp_path / "selected.xml"
    authoritative.write_text("<selected/>")
    settings = PipelineGuiSettings(
        basepath=str(source), local_root=str(tmp_path / "local"),
        xml_path=str(authoritative), multi_day_enabled=True,
        multi_day_session_paths=[str(source)], multi_day_name="combined",
    )
    prior = _input_provenance(settings)
    for item in prior["inputs"]:
        item.pop("sha256", None)
    xml.write_text("<changed/>")
    assert _input_snapshots_compatible(prior, _input_provenance(settings), settings)
