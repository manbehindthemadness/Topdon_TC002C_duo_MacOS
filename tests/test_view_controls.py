"""
Control-row imports and malformed numeric telemetry remain safe across Qt processes.
"""

from pathlib import Path

from tests.support.qt_process import run_popup


def test_control_rows_preserve_units_and_reject_invalid_calibration_estimates(
    tmp_path: Path,
) -> None:
    """
    Exercise the public facade, typed widget variants and optional calibration numbers.
    """
    result = run_popup(
        '''
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ControlRow, DistanceCalibrationControls
from topdon_duo.view_controls import ControlRow as ExtractedControlRow

app = QApplication([])
assert ControlRow is ExtractedControlRow
sent = []
row = ControlRow("Temperature", sent.append, minimum=-50, maximum=100, step=0.1, unit="°C")
row.update_state(20)
row.set_display_unit("F")
assert row.input.value() == 68
assert row.value() == 20
row.set_display_unit("C")
assert row.value() == 20 and not sent
toggle = ControlRow("Toggle", sent.append, options=((0, "Off"), (1, "On")))
toggle.update_state(1)
assert toggle.input.isChecked() and toggle.value() == 1
choice = ControlRow("Choice", sent.append, options=((1, "One"), (2, "Two")))
choice.update_state(2)
assert choice.value() == 2 and not sent
distance = ControlRow("Distance", sent.append, minimum=0, maximum=1, step=0.001, unit="m")
distance.input.setValue(12.3)
assert distance.value() == 0.123
distance.set_display_unit("F")
assert distance.value() == 0.123
distance.set_display_unit("C")
assert distance.value() == 0.123
fine = ControlRow("Fine control", sent.append, minimum=0, maximum=1, step=0.0001)
fine.input.setValue(0.1234)
assert fine.value() == 0.1234
offset = ControlRow("Offset step", sent.append, minimum=0.005, maximum=1, step=0.01)
offset.input.setValue(0.125)
assert offset.value() == 0.125
temperature = ControlRow("Fine temperature", sent.append, minimum=0, maximum=100,
                         step=0.001, unit="°C")
temperature.update_state(21.123)
temperature.set_display_unit("F")
assert temperature.value() == 21.123
temperature.set_display_unit("C")
assert temperature.value() == 21.123
tool = DistanceCalibrationControls(sent.append)
for invalid in (None, True, "1.0", [], {}, float("nan"), float("inf"), 10 ** 400):
    tool.update_state({"estimated_m": invalid}, "C", False)
    assert not tool.apply.isEnabled()
    assert "Estimated distance" not in tool.status.text()
tool.update_state({"estimated_m": 1.5}, "C", False)
assert tool.apply.isEnabled()
assert "Estimated distance: 150.00 cm" in tool.status.text()
tool.update_state({"estimated_m": 1.5}, "C", True)
assert not tool.apply.isEnabled()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_generic_hardware_rows_retain_incompatible_saved_values(tmp_path: Path) -> None:
    """
    Keep foreign primitives visible and editable without coercing the saved document.
    """
    result = run_popup(
        '''
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
from topdon_duo.camera_backends.contracts import ControlSpec

app = QApplication([])
window = ViewWindow(lambda message: None)
editor = window.pipeline_editor
item = node("hardware", "device_control", control="gain", value="high")
editor.document["hardware"] = [item]
editor.rebuild()
spec = ControlSpec("Gain", minimum=0, maximum=3, default=1).as_dict()
spec.update(available=True, supported=True)
state = {"camera_capabilities": {"controls": {"gain": spec}, "features": {}}}
editor.update_state(state, False)
fields = editor.hardware_widgets[item["id"]]
assert item["params"]["value"] == "high"
assert fields.rows["value"].input.value() == 1
assert not fields.warning.isHidden()
editor.update_state(state, True)
assert not fields.rows["value"].input.isEnabled()
assert item["params"]["value"] == "high"
editor.update_state(state, False)
fields.rows["value"].input.setValue(2)
fields.rows["value"].timer.timeout.emit()
assert item["params"]["value"] == 2
fields.update_state(state, False)
assert fields.warning.isHidden()

for spec, saved, expected in (
    (ControlSpec("Toggle", "boolean", default=False), "yes", False),
    (ControlSpec("Mode", "choice", options=(("low", "Low"), ("high", "High")),
                 default="low"), 2, "low"),
    (ControlSpec("Gain", minimum=0, maximum=3, default=1), True, 1),
    (ControlSpec("Gain", minimum=0, maximum=3, default=1), 9, 1),
):
    description = spec.as_dict()
    description.update(available=True, supported=True)
    state["camera_capabilities"]["controls"]["gain"] = description
    item["params"]["value"] = saved
    fields.update_state(state, False)
    assert item["params"]["value"] == saved
    assert fields.rows["value"].value() == expected
    assert not fields.warning.isHidden()
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
