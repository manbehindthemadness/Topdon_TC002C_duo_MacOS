"""
One preset control manages Environment and Calibrations through the viewer boundary.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_combined_preset_management_pending_inputs_acknowledgements_and_locks(tmp_path: Path) -> None:
    """
    Exercise the real Imaging widget without UI-side camera access or preset file writes.
    """
    result = run_popup('''
from pathlib import Path
import sys
from unittest.mock import patch
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea
from based_thermals.camera_backends.duo import control_specs
from based_thermals.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
controls = window.environment_presets
assert controls.combo.accessibleName() == "Environment + calibration presets"
assert not controls.save.isEnabled()
assert not window.environment_content.isAncestorOf(controls)
assert not window.calibrations_content.isAncestorOf(controls)
specs = control_specs()
state = {"temperature_unit": "F", "camera_capabilities": {
    "controls": {name: spec.as_dict() for name, spec in specs.items()}, "features": {}},
    "hardware": {name: {"value": spec.minimum, "available": True} for name, spec in specs.items()},
    "environment_presets": {"names": [], "current": None, "ready": True, "locked": False}}
window.update_state(state)
assert controls.combo.count() == 4
assert controls.save.isEnabled() and controls.save.text() == "Save preset"
assert "No saved presets" in controls.status.text()
assert not controls.combo.model().item(2).isEnabled()
assert not controls.combo.model().item(3).isEnabled()
sent = len(messages)
for _ in range(3):
    window.update_state(state)
assert len(messages) == sent
window.environment_button.click()
app.processEvents()
ambient = window.hardware_rows["ambient"]
ambient.input.setValue(75.02)
assert ambient.timer.isActive()
with patch("based_thermals.environment_preset_ui.QInputDialog.getText", return_value=("Desk", True)):
    controls.save.click()
assert messages[-2]["action"] == "hardware" and messages[-2]["name"] == "ambient"
assert abs(messages[-2]["value"] - 23.9) < .01
assert messages[-1] == {"action": "environment_preset", "operation": "save", "name": "Desk", "replace": False}
assert not ambient.timer.isActive()
assert controls.combo.currentIndex() == 0
state["environment_presets"] = {"names": ["Desk", "Studio"], "current": "Desk",
                                "ready": True, "locked": False, "modified": False}
window.update_state(state)
assert controls.combo.currentText() == "Desk"
assert controls.save.text() == "Update preset"
assert controls.combo.itemText(0) == "Current settings (saved)"
assert controls.status.text() == 'Current settings are saved as "Desk".'
controls.combo.showPopup()
app.processEvents()
studio_index = next(i for i in range(controls.combo.count()) if controls.combo.itemData(i) == ("load", "Studio"))
controls.combo.view().setCurrentIndex(controls.combo.model().index(studio_index, 0))
for _ in range(5):
    window.update_state(state)
assert controls.combo.view().isVisible()
assert controls.combo.view().currentIndex().row() == studio_index
controls.combo.hidePopup()
sent = len(messages)
controls.save.click()
assert messages[-1] == {"action": "environment_preset", "operation": "save", "name": "Desk", "replace": True}
assert len(messages) == sent + 1
state["environment_presets"]["modified"] = True
window.update_state(state)
assert controls.combo.currentText() == "Desk (modified)"
assert controls.combo.itemText(0) == "Current settings (modified)"
assert "unsaved changes" in controls.status.text()
ambient.input.setValue(76.1)
assert ambient.timer.isActive()
index = next(i for i in range(controls.combo.count()) if controls.combo.itemData(i) == ("load", "Studio"))
controls.select(index)
assert messages[-1] == {"action": "environment_preset", "operation": "load", "name": "Studio"}
assert not ambient.timer.isActive()
assert controls.combo.currentText() == "Desk (modified)"
window.update_state(state)  # A rejected load retains the last acknowledged name.
assert controls.combo.currentText() == "Desk (modified)"
state["environment_presets"].update(current="Studio", modified=False)
window.update_state(state)
assert controls.combo.currentText() == "Studio"
with patch("based_thermals.environment_preset_ui.QInputDialog.getText", return_value=("Office", True)):
    controls.select(2)
assert messages[-1] == {"action": "environment_preset", "operation": "rename", "name": "Studio", "new_name": "Office", "replace": False}
sent = len(messages)
with patch("based_thermals.environment_preset_ui.QInputDialog.getText", return_value=("Desk", True)), patch("based_thermals.environment_preset_ui.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
    controls.select(1)
assert len(messages) == sent
with patch("based_thermals.environment_preset_ui.QInputDialog.getText", return_value=("", False)):
    controls.select(2)
assert len(messages) == sent
with patch("based_thermals.environment_preset_ui.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
    controls.select(3)
assert messages[-1] == {"action": "environment_preset", "operation": "delete", "name": "Studio"}
for extra in ({"settings_locked": True}, {"camera_operation_busy": True}, {"tone_busy": True},
              {"environment_presets": {**state["environment_presets"], "locked": True}},
              {"environment_presets": {**state["environment_presets"], "ready": False}}):
    window.update_state({**state, **extra})
    assert not controls.save.isEnabled() and not controls.combo.isEnabled()
    sent = len(messages)
    controls.save_current()
    controls.select(index)
    assert len(messages) == sent
window.update_state(state)
for _ in range(4):
    app.processEvents()
scroll = window.findChild(QScrollArea)
assert scroll.horizontalScrollBar().maximum() == 0
assert not list(Path(sys.argv[1]).rglob("*environment-presets.json"))
window.close()
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
