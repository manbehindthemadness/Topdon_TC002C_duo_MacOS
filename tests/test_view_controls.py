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
