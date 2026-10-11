"""
Loaded preset identity survives UI/application restarts alongside the working pipeline.
"""

from pathlib import Path

import pytest
from support.qt_process import run_popup


def test_loaded_edited_preset_survives_a_fresh_process(tmp_path: Path) -> None:
    """
    Restore unsaved edits under the chosen name even when another preset originally matched.
    """
    setup = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication, QInputDialog
from topdon_duo.settings_preferences import save_settings
from topdon_duo.view_window import ViewWindow
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
editor.insert("software", "gamma", 1)
QInputDialog.getText = lambda *_: ("A copy", True)
editor.save_current_preset()
QInputDialog.getText = lambda *_: ("Z selected", True)
editor.save_current_preset(save_as=True)
editor.change(editor.document["software"][1], "amount", 2.7)
assert editor.current_preset_name == "Z selected"
assert editor.presets["Z selected"]["software"][1]["params"]["amount"] != 2.7
save_settings({"pipeline": deepcopy(editor.document)})
window.close()
"""
    result = run_popup(setup, tmp_path)
    assert result.returncode == 0, result.stderr
    reopen = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication
import topdon_duo.pipeline_ui.selection as selection
from topdon_duo.settings_preferences import load_settings
from topdon_duo.pipeline_presets import load_presets
from topdon_duo.view_window import ViewWindow
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
path = editor.preset_selection.path
before = path.read_bytes()
# Temporary startup widgets and states without a pipeline must not overwrite the association.
editor.update_state({}, False)
assert path.read_bytes() == before
document = load_settings()["pipeline"]
# Node identity and UI expansion changes do not break preset association.
for item in document["software"]:
    item["id"] += "restored"
    item["expanded"] = not item["expanded"]
editor.update_state({"pipeline": document}, False)
assert editor.current_preset_name == "Z selected"
assert editor.preset_combo.currentText() == "Z selected"
assert editor.save_preset_button.text() == "Update pipeline"
assert editor.document["software"][1]["params"]["amount"] == 2.7
assert load_presets()["Z selected"]["software"][1]["params"]["amount"] != 2.7
assert path.read_bytes() == before
calls = []
original_save = selection.save_json
selection.save_json = lambda *args, **kwargs: calls.append(args)
editor.update_state({"pipeline": deepcopy(document)}, False)
editor.update_preset_selection()
assert calls == []
selection.save_json = original_save
editor.save_current_preset()
assert load_presets()["Z selected"]["software"][1]["params"]["amount"] == 2.7
window.close()
"""
    result = run_popup(reopen, tmp_path)
    assert result.returncode == 0, result.stderr


def test_new_deleted_renamed_and_unrelated_pipeline_selections(tmp_path: Path) -> None:
    """
    Restore intentional blank work and reject stale associations without issuing pipeline writes.
    """
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.create_new_pipeline()
QInputDialog.getText = lambda *_: ("Blank saved", True)
editor.save_current_preset()
editor.create_new_pipeline()
blank = deepcopy(editor.document)
reopened = ViewWindow(messages.append)
reopened.pipeline_editor.update_state({"pipeline": blank}, False)
assert reopened.pipeline_editor.current_preset_name is None
assert not reopened.pipeline_editor.recognize_preset
reopened.close()
editor.select_preset(editor.preset_combo.findText("Blank saved"))
QInputDialog.getText = lambda *args, **kwargs: ("Renamed", True)
editor.select_preset(1)
renamed = deepcopy(editor.document)
reopened = ViewWindow(messages.append)
sent = len(messages)
reopened.pipeline_editor.update_state({"pipeline": renamed}, False)
assert len(messages) == sent
assert reopened.pipeline_editor.current_preset_name == "Renamed"
reopened.close()
# A different restored pipeline may recognize its own exact preset, never the stale name.
reopened = ViewWindow(messages.append)
reopened.pipeline_editor.update_state({"pipeline": editor.presets["Yautja GPU"]}, False)
assert reopened.pipeline_editor.current_preset_name == "Yautja GPU"
reopened.close()
editor.update_preset_selection()
QMessageBox.question = lambda *_: QMessageBox.StandardButton.Yes
editor.delete_current_preset()
reopened = ViewWindow(messages.append)
reopened.pipeline_editor.update_state({"pipeline": renamed}, False)
assert reopened.pipeline_editor.current_preset_name is None
assert reopened.pipeline_editor.preset_combo.currentText() == "Create new"
reopened.close()
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr


def test_selection_write_failure_keeps_editing_and_preserves_previous_file(tmp_path: Path) -> None:
    """
    Unwritable metadata cannot reject a pipeline edit; a later successful write can recover.
    """
    script = """
from PySide6.QtWidgets import QApplication
import topdon_duo.pipeline_ui.selection as selection
from topdon_duo.view_window import ViewWindow
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
editor.select_preset(editor.preset_combo.findText("Yautja GPU"))
path = editor.preset_selection.path
before = path.read_bytes()
original_save = selection.save_json
def failed_save(*args: object, **kwargs: object) -> None:
    '''
    Simulate a permissions or disk failure at the metadata write boundary.
    '''
    raise OSError("read-only preferences")
selection.save_json = failed_save
editor.insert("software", "gamma", 1)
assert editor.current_preset_name == "Yautja GPU"
assert editor.document["software"][1]["type"] == "gamma"
assert path.read_bytes() == before
selection.save_json = original_save
editor.update_preset_selection()
assert path.read_bytes() != before
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("data", ["null", "[]", "broken", '{}',
                                  '{"name":3,"recognize":true,"fingerprint":"x"}',
                                  '{"recognize":true,"fingerprint":"x"}'])
def test_malformed_selection_metadata_falls_back_safely(tmp_path: Path, data: str) -> None:
    """
    Ignore malformed metadata and retain recognition of a valid restored preset.
    """
    path = tmp_path / "config" / "topdon-duo" / "pipeline-selection.json"
    path.parent.mkdir(parents=True)
    path.write_text(data)
    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
editor.update_state({"pipeline": editor.presets["Yautja GPU"]}, False)
assert editor.current_preset_name == "Yautja GPU"
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
