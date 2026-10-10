"""
Explicit cache maintenance preserves user models and avoids active installations.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture
from support.qt_process import run_popup

from topdon_duo import desktop, onnx_models
from topdon_duo.desktop_app.session import DesktopSession
from topdon_duo.model_downloads import ModelDownloads

viewer = viewer_fixture


@pytest.mark.usefixtures("viewer")
@pytest.mark.parametrize("logging", [False, True])
def test_cache_command_reports_space_and_honors_logging_lock(logging: bool) -> None:
    """
    Dispatch maintenance without applying a pipeline or touching camera controls.
    """
    session = DesktopSession(desktop, [])
    session.view_panel = Mock()
    session.view_panel.poll.return_value = [{"action": "clear_model_cache"}]
    graphs = cast(Mock, session.graphs)
    graphs.logging = logging
    clear = Mock(return_value=(2, 3_000_000))
    session.api = SimpleNamespace(**{**vars(desktop), "MODEL_DOWNLOADS": Mock(clear_cache=clear)})
    pipeline = session.pipeline
    revision = session.pipeline_revision
    session.poll_camera_commands()
    assert clear.call_count == (0 if logging else 1)
    assert session.pipeline is pipeline and session.pipeline_revision == revision
    if logging:
        assert "Stop logging" in session.status_message
    else:
        assert session.status_message == "Model cache cleared: 2 files, 3.0 MB freed"
    hardware = cast(Mock, session.hardware)
    hardware.set.assert_not_called()


def test_clear_cache_removes_only_downloaded_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Remove catalog/custom downloads and checksums, preserving other cache contents.
    """
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    downloaded = onnx_models.model_path("espcn")
    downloaded.write_bytes(b"weights")
    checksum = downloaded.with_suffix(".sha256")
    checksum.write_bytes(b"checksum")
    custom = tmp_path / ("custom-" + "a" * 24 + "-" + "b" * 12 + ".onnx")
    custom.write_bytes(b"custom")
    registered = onnx_models.model_path("style-candy")
    registered.write_bytes(b"local choice")
    registry = tmp_path / "custom-models.json"
    registry.write_text(json.dumps([str(registered)]))
    export_name = next(name for name, spec in onnx_models.MODELS.items() if spec.get("local_export"))
    exported = onnx_models.model_path(export_name)
    exported.write_bytes(b"expensive local export")
    unknown = tmp_path / "my-model.onnx"
    unknown.write_bytes(b"personal model")
    outside = tmp_path.parent / "outside.onnx"
    outside.write_bytes(b"outside")
    link = tmp_path / ("custom-" + "c" * 24 + "-" + "d" * 12 + ".onnx")
    link.symlink_to(outside)
    manager = ModelDownloads()
    manager.jobs["old"] = {"state": "error", "message": "old failure"}
    assert manager.clear_cache() == (3, len(b"weightschecksumcustom"))
    assert not downloaded.exists() and not checksum.exists() and not custom.exists()
    assert all(path.exists() for path in (registered, registry, exported, unknown, link, outside))
    assert manager.status() == ""
    assert manager.clear_cache() == (0, 0)


def test_clear_cache_refuses_active_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    A pending job excludes clearing without losing its progress or cached files.
    """
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    path = onnx_models.model_path("espcn")
    path.write_bytes(b"keep")
    manager = ModelDownloads()
    manager.jobs["espcn"] = {"state": "pending", "message": "Downloading"}
    with pytest.raises(ValueError, match="finish"):
        manager.clear_cache()
    assert path.read_bytes() == b"keep"
    assert manager.status() == "Downloading"


def test_clear_cache_fails_safely_on_invalid_local_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Do not remove files when local-model protections cannot be read reliably.
    """
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path))
    path = onnx_models.model_path("espcn")
    path.write_bytes(b"keep")
    (tmp_path / "custom-models.json").write_text('{}')
    with pytest.raises(ValueError, match="local model index"):
        ModelDownloads().clear_cache()
    assert path.read_bytes() == b"keep"


def test_preset_selector_requests_cache_clear_without_pipeline_edits(tmp_path: Path) -> None:
    """
    Confirmation and locking control the action without mutating pipeline or presets.
    """
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication, QMessageBox
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
before = deepcopy(editor.document)
presets = deepcopy(editor.presets)
serial = editor.edit_serial
index = next(i for i in range(editor.preset_combo.count())
             if editor.preset_combo.itemData(i) == ("clear_cache", ""))
QMessageBox.question = lambda *args: QMessageBox.StandardButton.No
editor.select_preset(index)
assert messages == []
QMessageBox.question = lambda *args: QMessageBox.StandardButton.Yes
editor.select_preset(index)
assert messages == [{"action": "clear_model_cache"}]
assert editor.document == before and editor.presets == presets and editor.edit_serial == serial
editor.locked = True
editor.select_preset(index)
assert len(messages) == 1
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
