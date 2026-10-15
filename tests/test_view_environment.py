"""
Environment disclosure keeps permanent controls together without changing their behavior.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_environment_disclosure_refresh_controls_and_keyboard(tmp_path: Path) -> None:
    """
    Exercise real Qt visibility, capability replacement, edit messages, units and locks.
    """
    result = run_popup('''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.camera_backends.duo import control_specs
from based_thermals.view_window import ViewWindow
from based_thermals.view_settings import VIEW_DEFAULTS

app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
button = window.environment_button
content = window.environment_content
assert button.text() == "▸ Environment" and button.accessibleName() == "Environment"
assert button.isCheckable() and not button.isChecked()
assert button.size().width() > 500
assert content.isHidden()
expected = [window.rows["temperature_unit"], *[
    window.hardware_rows[name] for name in
    ("ambient", "distance", "emissivity", "reflected", "transmission", "humidity")]]
assert [window.hardware_layout.itemAt(i).widget() for i in range(7)] == expected
assert all(content.isAncestorOf(row) for row in expected)
assert not content.isAncestorOf(window.pipeline_editor)
for widget in (window.auto_calibrate, window.distance_calibration,
               window.emissivity_calibration, window.reflected_calibration,
               window.reported_readings, window.spot_hardware):
    assert not content.isAncestorOf(widget)
sent = len(messages)
QTest.mouseClick(button, Qt.MouseButton.LeftButton)
assert button.isChecked() and content.isVisible()
assert button.text() == "▾ Environment"
assert len(messages) == sent
button.setFocus()
QTest.keyClick(button, Qt.Key.Key_Space)
assert content.isHidden() and not button.isChecked()
assert len(messages) == sent

specs = {name: spec.as_dict() for name, spec in control_specs().items()}
hardware = {name: {"value": spec["default"], "enabled": False, "available": True}
            for name, spec in specs.items()}
hardware["humidity"]["value"] = 45.5
hardware["distance"]["value"] = .51
state = {**VIEW_DEFAULTS, "hardware": hardware,
         "camera_capabilities": {"controls": specs, "features": {}}}
old_humidity = window.hardware_rows["humidity"]
window.update_state(state)
humidity = window.hardware_rows["humidity"]
assert humidity is not old_humidity
assert humidity.parentWidget() is content
assert content.isHidden() and not humidity.isVisible()
assert humidity.input.value() == 45.5
assert len(messages) == sent
QTest.keyClick(button, Qt.Key.Key_Space)
assert humidity.isVisible() and button.isChecked()
humidity.input.setValue(50)
humidity._emit()
assert messages[-1] == {"action": "hardware", "name": "humidity", "value": 50,
                        "enabled": True}
units = window.controls["temperature_unit"]
units.setCurrentIndex(units.findData("F"))
window.rows["temperature_unit"].timer.stop()
window.rows["temperature_unit"]._emit()
assert messages[-1] == {"action": "setting", "name": "temperature_unit", "value": "F"}
state["temperature_unit"] = "F"
window.update_state(state)
assert window.hardware_rows["distance"].input.suffix() == " in"
assert abs(window.hardware_rows["distance"].input.value() - 20.08) < .02
assert content.isVisible() and button.isChecked()

QTest.mouseClick(button, Qt.MouseButton.LeftButton)
specs["humidity"] = {**specs["humidity"], "title": "Updated humidity"}
specs["extra"] = {**specs["humidity"], "title": "Additional environment control"}
hardware["extra"] = {"value": 42, "available": True}
sent = len(messages)
window.update_state(state)
assert content.isHidden() and len(messages) == sent
assert window.hardware_rows["extra"].parentWidget() is content
assert window.hardware_layout.indexOf(window.hardware_rows["extra"]) == 7
QTest.mouseClick(button, Qt.MouseButton.LeftButton)
assert window.hardware_rows["extra"].isVisible()
window.update_state({**state, "settings_locked": True})
assert not window.hardware_rows["humidity"].input.isEnabled()
assert not units.isEnabled()
QTest.mouseClick(button, Qt.MouseButton.LeftButton)
assert content.isHidden()
window.update_state({**state, "settings_locked": False})
assert content.isHidden() and window.hardware_rows["humidity"].input.isEnabled()
window.close()
reopened = ViewWindow(messages.append)
assert reopened.environment_content.isHidden()
assert not reopened.environment_button.isChecked()
reopened.environment_button.click()
reopened.close()
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for expanded in (True, False):
        result = run_popup('''
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
expanded = ''' + repr(expanded) + '''
assert window.environment_button.isChecked() == expanded
assert window.environment_content.isHidden() != expanded
window.update_state({"temperature_unit": "F"})
assert window.environment_button.isChecked() == expanded
assert window.environment_content.isHidden() != expanded
window.environment_button.click()
window.close()
''', tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
