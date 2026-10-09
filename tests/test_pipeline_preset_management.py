"""
Creating blank pipelines and deleting saved user and bundled presets.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_capture_panel import popup_environment


def test_deleted_bundled_presets_stay_deleted_and_can_be_saved_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Persist bundled deletions without modifying shipped resources or other presets.
    """
    from topdon_duo.pipeline_presets import load_presets, predefined_presets, save_presets

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    presets = load_presets()
    name = "Yautja GPU"
    deleted = presets.pop(name)
    save_presets(presets)
    assert load_presets() == presets
    path = tmp_path / "topdon-duo" / "pipeline-presets.json"
    assert json.loads(path.read_text())[name] is None
    assert predefined_presets()[name] == deleted
    presets["User pipeline"] = deleted
    save_presets(presets)
    assert load_presets() == presets
    presets.pop("User pipeline")
    presets[name] = deleted
    save_presets(presets)
    assert load_presets() == presets
    assert json.loads(path.read_text()) == {}


def test_create_new_delete_confirmation_locks_and_write_failure(tmp_path: Path) -> None:
    """
    Start blank work and delete presets safely through the offscreen Qt controls.
    """
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox
import topdon_duo.pipeline_ui.editor as editor_module
from topdon_duo.pipeline import validate_pipeline
from topdon_duo.pipeline_presets import load_presets
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
assert editor.preset_combo.itemText(0) == "Create new"
assert not editor.preset_combo.model().item(3).isEnabled()
editor.select_preset(editor.preset_combo.findText("Yautja GPU"))
original = deepcopy(editor.document)
editor.select_tab(2)
editor.select_preset(0)
assert editor.tab == "A" and editor.tab_bar.currentIndex() == 0
assert editor.document["hardware"] == []
assert [n["type"] for n in editor.document["software"]] == ["source", "output"]
assert all([n["type"] for n in stack] == ["source"]
           for stack in editor.document["branches"].values())
assert validate_pipeline(editor.document) == editor.document
assert messages[-1]["document"] == editor.document
assert editor.current_preset_name is None
assert editor.save_preset_button.text() == "Save pipeline"
assert load_presets()["Yautja GPU"] == original
QInputDialog.getText = lambda *_: ("Blank pipeline", True)
editor.save_preset_button.click()
assert editor.current_preset_name == "Blank pipeline"
editor.select_preset(0)
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, False)
assert editor.current_preset_name is None
assert editor.save_preset_button.text() == "Save pipeline"
editor.select_preset(editor.preset_combo.findText("Blank pipeline"))
assert editor.preset_combo.model().item(3).isEnabled()
QMessageBox.question = lambda *_: QMessageBox.StandardButton.No
before = deepcopy(load_presets())
editor.select_preset(editor.preset_combo.findText("Delete current pipeline…"))
assert load_presets() == before
assert editor.preset_combo.currentText() == "Blank pipeline"
QMessageBox.question = lambda *_: QMessageBox.StandardButton.Yes
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, True)
assert not editor.preset_combo.model().item(3).isEnabled()
editor.select_preset(0)
editor.select_preset(editor.preset_combo.findText("Delete current pipeline…"))
assert load_presets() == before and editor.current_preset_name == "Blank pipeline"
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, False)
saved_document = deepcopy(editor.document)
editor.select_preset(editor.preset_combo.findText("Delete current pipeline…"))
assert "Blank pipeline" not in load_presets()
assert editor.document == saved_document
assert editor.preset_combo.currentText() == "Create new"
assert editor.save_preset_button.text() == "Save pipeline"
editor.select_preset(editor.preset_combo.findText("Yautja GPU"))
original_save = editor_module.save_presets
def failed_save(_candidate: dict) -> None:
    '''
    Simulate a persistence failure before the existing collection changes.
    '''
    raise OSError("disk full")
editor_module.save_presets = failed_save
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
before = deepcopy(load_presets())
editor.select_preset(editor.preset_combo.findText("Delete current pipeline…"))
assert warnings and load_presets() == before
assert editor.current_preset_name == "Yautja GPU"
editor_module.save_presets = original_save
editor.select_preset(editor.preset_combo.findText("Delete current pipeline…"))
assert "Yautja GPU" not in load_presets()
assert editor.document == original
reopened = ViewWindow(lambda _: None)
assert "Yautja GPU" not in reopened.pipeline_editor.presets
reopened.close()
window.close()
"""
    env = popup_environment()
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
