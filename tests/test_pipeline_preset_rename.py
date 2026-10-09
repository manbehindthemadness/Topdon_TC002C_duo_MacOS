"""
Renaming user and bundled pipeline presets through the editor.
"""

import subprocess
import sys
from pathlib import Path

from test_capture_panel import popup_environment


def test_rename_preserves_edits_handles_collisions_and_persists(tmp_path: Path) -> None:
    """
    Rename safely across cancellation, locks, name collisions and write failures.
    """
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox
import topdon_duo.pipeline_ui.presets as preset_ui
from topdon_duo.pipeline_presets import load_presets, predefined_presets
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
assert editor.preset_combo.itemText(0) == "Create new"
assert editor.preset_combo.itemText(1) == "Rename current pipeline…"
assert not editor.preset_combo.model().item(1).isEnabled()
prompts = []
def prompt(*args: object, **kwargs: object) -> tuple[str, bool]:
    '''
    Inspect the initial name supplied to the rename dialog.
    '''
    prompts.append(kwargs)
    return "  Renamed pipeline  ", True
QInputDialog.getText = prompt
editor.select_preset(1)
assert prompts == []
editor.insert("software", "gamma", 1)
QInputDialog.getText = lambda *args, **kwargs: ("Original pipeline", True)
editor.save_current_preset()
saved = deepcopy(load_presets()["Original pipeline"])
gamma = editor.document["software"][1]
editor.change(gamma, "amount", 2.3)
working = deepcopy(editor.document)
sent = len(messages)
QInputDialog.getText = prompt
editor.select_preset(1)
assert prompts == [{"text": "Original pipeline"}]
assert editor.document == working and len(messages) == sent
assert "Original pipeline" not in load_presets()
assert load_presets()["Renamed pipeline"] == saved
assert editor.preset_combo.currentText() == "Renamed pipeline"
assert editor.save_preset_button.text() == "Update pipeline"
editor.save_preset_button.click()
assert load_presets()["Renamed pipeline"] == working
before = deepcopy(load_presets())
for response in (("", False), ("   ", True), ("Renamed pipeline", True)):
    QInputDialog.getText = lambda *args, **kwargs: response
    editor.select_preset(1)
    assert load_presets() == before
    assert editor.current_preset_name == "Renamed pipeline"
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, True)
assert not editor.preset_combo.model().item(1).isEnabled()
QInputDialog.getText = prompt
editor.select_preset(1)
assert len(prompts) == 1 and load_presets() == before
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, False)
assert editor.preset_combo.model().item(1).isEnabled()
QInputDialog.getText = lambda *args, **kwargs: ("Yautja", True)
QMessageBox.question = lambda *_: QMessageBox.StandardButton.No
editor.select_preset(1)
assert load_presets() == before and editor.current_preset_name == "Renamed pipeline"
QMessageBox.question = lambda *_: QMessageBox.StandardButton.Yes
editor.select_preset(1)
assert "Renamed pipeline" not in load_presets()
assert load_presets()["Yautja"] == working
assert editor.current_preset_name == "Yautja"
# Renaming a bundled preset creates a user preset and hides the old bundled name.
editor.select_preset(editor.preset_combo.findText("Yautja GPU"))
bundled = deepcopy(predefined_presets()["Yautja GPU"])
QInputDialog.getText = lambda *args, **kwargs: ("My GPU pipeline", True)
editor.select_preset(1)
assert editor.document == bundled
assert "Yautja GPU" not in load_presets()
assert load_presets()["My GPU pipeline"] == bundled
assert predefined_presets()["Yautja GPU"] == bundled
reopened = ViewWindow(lambda _: None)
assert "Yautja GPU" not in reopened.pipeline_editor.presets
assert reopened.pipeline_editor.presets["My GPU pipeline"] == bundled
reopened.close()
before = deepcopy(load_presets())
original_save = preset_ui.save_presets
def failed_save(_candidate: dict) -> None:
    '''
    Fail at the persistence boundary without changing existing presets.
    '''
    raise OSError("disk full")
preset_ui.save_presets = failed_save
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
QInputDialog.getText = lambda *args, **kwargs: ("Failed rename", True)
editor.select_preset(1)
assert warnings and load_presets() == before
assert editor.presets == before and editor.current_preset_name == "My GPU pipeline"
assert editor.preset_combo.currentText() == "My GPU pipeline"
assert editor.document == bundled
preset_ui.save_presets = original_save
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
