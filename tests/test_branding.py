"""
Application branding uses current interfaces while retaining historical pipeline imports.
"""

import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest
from support.qt_process import run_popup

from based_thermals import APP_NAME, onnx_models
from based_thermals.desktop_app.constants import WINDOW_NAME
from based_thermals.pipeline import default_pipeline
from based_thermals.pipeline_ui.transfer.documents import read_pipelines, write_pipelines
from based_thermals.web import LiveStream, create_app


@pytest.mark.parametrize("suffix", ["", "-desktop", "-models"])
def test_distribution_installs_only_current_commands(suffix: str) -> None:
    """
    Run current commands without opening a camera or downloading models.
    """
    installed = distribution("based-thermals")
    assert installed.metadata["Name"] == "based-thermals"
    entry_points = {entry.name: entry.value for entry in installed.entry_points}
    current, legacy = f"based-thermals{suffix}", f"topdon-duo{suffix}"
    assert legacy not in entry_points
    assert not (Path(sys.executable).parent / legacy).exists()
    result = subprocess.run(
        [str(Path(sys.executable).parent / current), "--help"],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
    if suffix != "-models":
        assert "based-thermals" in result.stdout


def test_web_title_is_application_brand_and_device_identity_remains_visible() -> None:
    """
    Distinguish the application title from its actual connected camera's metadata.
    """
    stream = LiveStream()
    response = create_app(stream).test_client().get("/")
    text = response.get_data(as_text=True)
    assert "<title>based-thermals</title>" in text
    assert "<h1>based-thermals</h1>" in text
    assert stream.profile.title in text
    assert APP_NAME == WINDOW_NAME == "based-thermals"


def test_model_directory_uses_current_override_and_ignores_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Use the current cache or explicit override without falling back to an obsolete variable.
    """
    monkeypatch.setenv("TOPDON_MODEL_DIR", str(tmp_path / "legacy"))
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path / "current"))
    assert onnx_models.model_path("espcn").parent == tmp_path / "current"
    monkeypatch.delenv("BASED_THERMALS_MODEL_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    assert onnx_models.model_path("espcn").parent == tmp_path / "based-thermals" / "models"


@pytest.mark.parametrize("format_name", ["based-thermals-pipelines", "topdon-duo-pipelines"])
def test_named_exports_use_current_brand_and_read_both_formats(
    format_name: str, tmp_path: Path,
) -> None:
    """
    Keep all saved settings importable while new portable bundles carry the project name.
    """
    path = tmp_path / "portable.json"
    document = default_pipeline()
    write_pipelines(path, {"Saved": document}, single=False)
    payload = json.loads(path.read_text())
    assert payload["format"] == "based-thermals-pipelines"
    payload["format"] = format_name
    path.write_text(json.dumps(payload))
    loaded = read_pipelines(path)
    assert len(loaded) == 1 and loaded[0].name == "Saved"
    assert loaded[0].document == document
    payload["version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported"):
        read_pipelines(path)


def test_qt_viewer_uses_generic_application_title(tmp_path: Path) -> None:
    """
    Check the actual Qt window separately from OpenCV's native binding.
    """
    result = run_popup(
        '''
from PySide6.QtWidgets import QApplication
from based_thermals.viewer_window import ViewerWindow
app = QApplication([])
window = ViewerWindow(lambda message: None)
assert window.windowTitle() == "based-thermals"
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
