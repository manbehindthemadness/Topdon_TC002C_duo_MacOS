"""
Fresh desktop startup selects the published CPU preset while restored work remains authoritative.
"""

from pathlib import Path
from typing import Any

import pytest
from support.desktop_recording import viewer_fixture
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.desktop_app.session import DesktopSession
from based_thermals.pipeline import default_pipeline, node
from based_thermals.pipeline_presets import predefined_presets
from based_thermals.pipeline_ui.presets import pipeline_content
from based_thermals.settings_preferences import load_settings, save_settings


@pytest.fixture
def fresh_viewer(viewer: Any, tmp_path: Path) -> Any:
    """
    Keep the fake desktop boundaries but remove the interaction fixture's saved baseline.
    """
    (tmp_path / "config/based-thermals/settings.json").unlink()
    return viewer


@pytest.mark.usefixtures("fresh_viewer")
def test_fresh_duo_session_uses_published_cpu_preset() -> None:
    """
    Start from the exact shipped settings without saved viewer preferences or hardware access.
    """
    assert not load_settings()
    session = DesktopSession(desktop, [])
    assert session.pipeline == predefined_presets()["Detail Enhanced CPU Upscale"]


@pytest.mark.usefixtures("fresh_viewer")
def test_saved_working_pipeline_wins_over_first_install_default() -> None:
    """
    Preserve existing work including an edited document that differs from every preset.
    """
    document = default_pipeline()
    document["software"].insert(1, node("software", "gamma", amount=2.7))
    save_settings({"pipeline": document})
    session = DesktopSession(desktop, [])
    assert session.pipeline == document


@pytest.mark.usefixtures("fresh_viewer")
def test_legacy_preferences_still_migrate_instead_of_loading_cpu_preset() -> None:
    """
    Preserve older saved settings that predate working-pipeline persistence.
    """
    save_settings({"display": {"gamma": 2.1}})
    expected = desktop.migrate_pipeline(load_settings())
    session = DesktopSession(desktop, [])
    assert pipeline_content(session.pipeline) == pipeline_content(expected)


@pytest.mark.usefixtures("fresh_viewer")
def test_cli_source_override_still_applies_to_fresh_preset() -> None:
    """
    Honor explicit source selection after choosing the fresh-install pipeline.
    """
    session = DesktopSession(desktop, ["--image-source", "raw"])
    expected = predefined_presets()["Detail Enhanced CPU Upscale"]
    expected["software"][0]["params"]["source"] = "raw"
    assert session.pipeline == expected


@pytest.mark.usefixtures("fresh_viewer")
def test_virtual_profile_does_not_inherit_duo_preset_controls() -> None:
    """
    Keep a new synthetic camera on its own pipeline and capability defaults.
    """
    session = DesktopSession(desktop, ["--camera", "virtual-dual"])
    assert session.pipeline["hardware"] == []


def test_first_viewer_state_selects_cpu_preset_in_dropdown(tmp_path: Path) -> None:
    """
    Show the default preset's name when the fresh desktop sends its initial pipeline to Qt.
    """
    script = """
from PySide6.QtWidgets import QApplication
from based_thermals.pipeline_presets import predefined_presets
from based_thermals.view_window import ViewWindow
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
editor.update_state({'pipeline': predefined_presets()['Detail Enhanced CPU Upscale']}, False)
assert editor.current_preset_name == 'Detail Enhanced CPU Upscale'
assert editor.preset_combo.currentText() == 'Detail Enhanced CPU Upscale'
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr


__all__ = ["viewer_fixture"]
