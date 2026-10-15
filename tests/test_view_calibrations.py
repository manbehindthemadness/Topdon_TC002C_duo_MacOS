"""
Calibration tools share the Environment disclosure style and retain independent UI preferences.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_calibrations_disclosure_controls_refresh_and_persistence(tmp_path: Path) -> None:
    """
    Exercise tool visibility, edit messages, locks, units and saved state across Qt processes.
    """
    result = run_popup('''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
from based_thermals.view_settings import VIEW_DEFAULTS

app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
button = window.calibrations_button
content = window.calibrations_content
tools = (window.distance_calibration, window.emissivity_calibration,
         window.reflected_calibration)
assert button.text() == "▸ Calibrations"
assert button.accessibleName() == "Calibrations"
assert button.styleSheet() == window.environment_button.styleSheet()
assert button.width() > 500
assert not button.isChecked() and content.isHidden()
assert all(content.isAncestorOf(tool) for tool in tools)
assert [content.layout().itemAt(i).widget() for i in range(3)] == list(tools)
assert not content.isAncestorOf(window.environment_content)
sent = len(messages)
QTest.mouseClick(button, Qt.MouseButton.LeftButton)
assert button.isChecked() and content.isVisible()
assert button.text() == "▾ Calibrations"
window.environment_button.click()
assert window.environment_content.isVisible() and content.isVisible()
button.setFocus()
QTest.keyClick(button, Qt.Key.Key_Space)
assert content.isHidden() and window.environment_content.isVisible()
assert len(messages) == sent
state = {**VIEW_DEFAULTS, "temperature_unit": "F", "camera_capabilities": {
    "controls": {}, "features": {"host_distance_calibration": True,
        "host_emissivity_calibration": True, "host_reflected_calibration": True}}}
window.update_state(state)
assert content.isHidden() and not any(tool.isVisible() for tool in tools)
assert window.emissivity_calibration.known.suffix() == " °F"
assert window.distance_calibration.distance.suffix() == " in"
assert len(messages) == sent
QTest.keyClick(button, Qt.Key.Key_Space)
assert content.isVisible() and all(tool.isVisible() for tool in tools)
window.distance_calibration.select.click()
assert messages[-1] == {"action": "distance_calibration", "operation": "select"}
window.reflected_calibration.select.click()
assert messages[-1] == {"action": "reflected_calibration", "operation": "select"}
window.update_state({**state, "settings_locked": True})
assert not window.distance_calibration.select.isEnabled()
assert not window.emissivity_calibration.known.isEnabled()
QTest.mouseClick(button, Qt.MouseButton.LeftButton)
assert content.isHidden()
window.update_state(state)
assert content.isHidden() and window.distance_calibration.select.isEnabled()
QTest.mouseClick(button, Qt.MouseButton.LeftButton)
assert content.isVisible()
features = state["camera_capabilities"]["features"]
features["host_emissivity_calibration"] = False
window.update_state(state)
assert content.isVisible() and not window.emissivity_calibration.isVisible()
assert window.distance_calibration.isVisible() and window.reflected_calibration.isVisible()
window.environment_button.click()
assert window.environment_content.isHidden() and content.isVisible()
window.close()
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for expanded in (True, False):
        result = run_popup('''
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
app = QApplication([])
window = ViewWindow(lambda _message: None)
expanded = ''' + repr(expanded) + '''
assert window.calibrations_button.isChecked() == expanded
assert window.calibrations_content.isHidden() != expanded
assert not window.environment_button.isChecked()
window.update_state({"temperature_unit": "C"})
assert window.calibrations_content.isHidden() != expanded
window.calibrations_button.click()
window.close()
''', tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
