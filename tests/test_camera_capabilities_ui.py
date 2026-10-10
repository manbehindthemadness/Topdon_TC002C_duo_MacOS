"""
Qt capability rendering and spot control scope without opening hardware.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_partial_capabilities_new_controls_and_spot_scope(tmp_path: Path) -> None:
    """
    Preserve unavailable nodes, render target bounds, and send only explicit unlocked edits.
    """
    script = '''
from PySide6.QtWidgets import QApplication
from topdon_duo.camera_backends import CameraProfile, ControlSpec
from topdon_duo.camera_backends.controls import capability_state
from topdon_duo.pipeline import default_pipeline, node
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
specs = {
    "gain": ControlSpec("Exposure gain", minimum=0, maximum=3, default=1),
    "humidity": ControlSpec("Measured humidity correction", effect="measurement", unit="%"),
    "ambient": ControlSpec("Ambient correction", minimum=-100, maximum=150, step=0.05,
                           effect="measurement", unit="°C"),
    "distance": ControlSpec("Spot distance", minimum=0.5, maximum=12, step=0.05,
                            scope="spot", effect="measurement", unit="m"),
}
hardware = {name: {"value": 20 if name == "ambient" else spec.default, "available": True}
            for name, spec in specs.items() if spec.scope == "device"}
caps = capability_state(specs, hardware, features={"ready": True})
document = default_pipeline()
unsupported = node("hardware", "preset", value="shadow")
gain = node("hardware", "device_control", control="gain", value=2)
document["hardware"] = [unsupported, gain]
state = {
    "pipeline": document, "pipeline_serial": 0, "hardware": hardware,
    "camera_profile": CameraProfile("new", "New camera", (320, 240), 9).as_dict(),
    "camera_capabilities": caps,
    "spot_hardware": {"A": {"distance": {"value": 2, "available": True}},
                      "B": {"distance": {"value": 3, "available": True}}},
    "reported_readings": [{"name": "Spot temperature", "value": 32.5, "unit": "°C",
                           "spot_id": "B", "valid": True}],
}
window.update_state(state)
app.processEvents()
assert messages == []
editor = window.pipeline_editor
assert editor.document["hardware"][0]["params"]["value"] == "shadow"
widgets = editor.widgets[unsupported["id"]]
assert "Not supported" in widgets[4].text()
assert not widgets[1]["value"].input.isEnabled()
assert widgets[2].isEnabled()  # Unavailable nodes can still be bypassed.
fields = editor.hardware_widgets[gain["id"]]
assert fields.rows["value"].input.maximum() == 3
assert fields.rows["value"].input.value() == 2
fields.rows["value"].input.lineEdit().setText("3")
editor.commit_pending_inputs()
assert gain["params"]["value"] == 2  # Incoming state is not mutated by UI edits.
assert editor.document["hardware"][1]["params"]["value"] == 3
messages.clear()
assert window.hardware_rows["ambient"].input.minimum() == -100
assert window.hardware_rows["ambient"].input.decimals() == 2
assert not window.distance_calibration.isVisible()
assert not window.auto_calibrate.isEnabled()
assert "32.5" in window.reported_readings.text() and "Spot B" in window.reported_readings.text()
spots = window.spot_hardware
spots.selector.setCurrentIndex(1)
assert spots.rows["distance"].input.value() == 300
spots.change("distance", 4.05)
assert messages == [{"action": "hardware", "name": "distance", "value": 4.05,
                     "enabled": True, "spot_id": "B"}]
messages.clear()
window.update_state({**state, "settings_locked": True})
spots.change("distance", 5)
fields.select("gain")
assert messages == []
assert not spots.rows["distance"].input.isEnabled()
window.update_state(state)
assert spots.selector.currentData() == "B"
assert messages == []
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
