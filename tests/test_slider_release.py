"""
Defer mouse slider edits until release without changing debounced numeric or keyboard input.
"""

from pathlib import Path

import pytest

from tests.support.qt_process import run_popup


@pytest.mark.parametrize("gesture", ["handle", "track"])
def test_slider_gestures_send_only_on_release(gesture: str, tmp_path: Path) -> None:
    """
    Exercise stale timers, held gestures, state echoes, single release writes and cancellation.
    """
    script = '''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractSlider, QStyle, QStyleOptionSlider
from based_thermals.view_controls import ControlRow
app = QApplication([])
sent = []
row = ControlRow("Camera value", sent.append, minimum=0, maximum=100, step=1, release_only=True)
row.resize(640, 120)
row.show()
app.processEvents()
slider = row.slider
assert slider is not None
row.input.setValue(30)
assert row.timer.isActive()
style = QStyleOptionSlider()
slider.initStyleOption(style)
handle = slider.style().subControlRect(
    QStyle.ComplexControl.CC_Slider, style, QStyle.SubControl.SC_SliderHandle, slider
)
gesture = "''' + gesture + '''"
point = handle.center() if gesture == "handle" else slider.rect().topRight()
point.setY(handle.center().y())
if gesture == "track":
    point.setX(slider.width() - 5)
QTest.mousePress(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
assert slider.is_interacting() and not row.timer.isActive()
slider.setRepeatAction(QAbstractSlider.SliderAction.SliderNoAction)
for value in (40, 50, 60):
    slider.setValue(value)
    QTest.qWait(300)
    assert row.value() == value and not sent
    assert not row.timer.isActive()
    row._emit()
    assert not sent
    row.update_state(10)
    assert row.value() == value
QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
assert not slider.is_interacting() and sent == [60]
QTest.qWait(300)
assert sent == [60] and not row.timer.isActive()
# Keyboard slider edits and typed numeric values keep their existing debounce.
QTest.keyClick(slider, Qt.Key.Key_Right)
assert row.timer.isActive() and sent == [60]
QTest.qWait(300)
assert sent == [60, 61]
row.input.setValue(62)
assert row.timer.isActive()
QTest.qWait(300)
assert sent == [60, 61, 62]
# Disabling during a held gesture cancels it without writing a camera value.
slider.initStyleOption(style)
handle = slider.style().subControlRect(
    QStyle.ComplexControl.CC_Slider, style, QStyle.SubControl.SC_SliderHandle, slider
)
point = handle.center()
QTest.mousePress(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
slider.setValue(80)
row.update_state(62, False)
QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
QTest.qWait(300)
assert sent == [60, 61, 62] and not row.timer.isActive()
row.update_state(62, True)
assert not slider.is_interacting() and row.value() == 62
row.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_policy_is_limited_to_hardware_controls(tmp_path: Path) -> None:
    """
    Keep software sliders unchanged and enable release-only edits for camera/calibration rows.
    """
    result = run_popup(
        '''
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.camera_backends.contracts import ControlSpec
from based_thermals.pipeline import node
from based_thermals.view_controls import ControlRow
from based_thermals.view_window import ViewWindow
app = QApplication([])
sent = []
software = ControlRow("Software value", sent.append, minimum=0, maximum=100)
assert not software.release_only
software.slider.setValue(40)
assert software.timer.isActive()
QTest.qWait(300)
assert sent == [40]
window = ViewWindow(lambda message: None)
assert all(row.release_only for row in window.hardware_rows.values())
editor = window.pipeline_editor
editor.insert("hardware", "brightness", 0)
hardware = editor.document["hardware"][0]
assert editor.widgets[hardware["id"]][1]["value"].release_only
editor.insert("software", "gamma", 1)
gamma = editor.document["software"][1]
assert not editor.widgets[gamma["id"]][1]["amount"].release_only
# Backend-declared controls and per-spot calibration controls also opt in.
spec = ControlSpec("Ambient", minimum=-50, maximum=100, unit="°C", effect="measurement").as_dict()
spec.update(supported=True, available=True)
state = {"camera_capabilities":{"controls":{"ambient":spec}, "features":{}},
         "hardware":{"ambient":{"value":20, "available":True}}}
window.update_state(state)
assert window.hardware_rows["ambient"].release_only
editor.document["hardware"] = [node("hardware", "device_control", control="gain", value=1)]
editor.rebuild()
gain = ControlSpec("Gain", minimum=0, maximum=3, effect="preview").as_dict()
gain.update(supported=True, available=True)
editor.update_state({"camera_capabilities":{"controls":{"gain":gain}, "features":{}}}, False)
device = editor.document["hardware"][0]
assert editor.hardware_widgets[device["id"]].rows["value"].release_only
spot_spec = ControlSpec("Spot emissivity", minimum=0.01, maximum=1, step=0.01,
                        scope="spot", effect="measurement").as_dict()
spot_spec.update(supported=True, available=True)
window.spot_hardware.update_state({
    "camera_capabilities":{"controls":{"emissivity":spot_spec}},
    "spot_hardware":{"1":{"emissivity":{"value":0.95,"available":True}}},
}, False)
assert window.spot_hardware.rows["emissivity"].release_only
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
