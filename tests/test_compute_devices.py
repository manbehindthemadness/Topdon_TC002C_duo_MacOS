"""
Compute-device controls reflect execution without rewriting saved preferences.
"""

import subprocess
import sys

from test_capture_panel import popup_environment


def test_cpu_execution_locks_devices_and_restores_apple_preferences() -> None:
    """
    Cover current and legacy model controls, pending edits, locks and backend switches.
    """
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from topdon_duo.pipeline import default_pipeline, node, validate_pipeline
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
for kind, model in (
    ("enhance", "acnet"), ("enhance", "realesr-general-x4v3"),
    ("onnx_superresolution", "espcn"), ("onnx_denoise", "ffdnet-gray"),
    ("onnx_style", "style-mosaic"),
):
    editor.document = default_pipeline()
    item = node("software", kind, model=model, backend="cpu", apple_compute="ALL")
    editor.document["software"].insert(1, item)
    editor.rebuild()
    row = editor.widgets[item["id"]][1]["apple_compute"]
    assert row.input.accessibleName() == "Compute devices"
    assert row.input.currentText() == "CPU only"
    assert not row.input.isEnabled()
    original = deepcopy(editor.document)
    state = {"pipeline": original, "pipeline_serial": editor.edit_serial,
             "apple_acceleration": {"available": True}}
    editor.update_state(state, False)
    assert not row.isHidden()
    assert row.value() == "CPUOnly" and not row.input.isEnabled()
    assert editor.document == original
    editor.change(item, "backend", "coreml")
    assert row.input.isEnabled() and row.value() == "ALL"
    row.input.setCurrentIndex(row.input.findData("CPUAndGPU"))
    assert row.timer.isActive()
    editor.change(item, "backend", "cpu")
    assert row.value() == "CPUOnly" and not row.input.isEnabled()
    assert not row.timer.isActive()
    row.timer.timeout.emit()
    assert item["params"]["apple_compute"] == "ALL"
    editor.change(item, "backend", "coreml")
    assert row.value() == "ALL" and row.input.isEnabled()
    editor.update_state({"pipeline_serial": editor.edit_serial,
                         "apple_acceleration": {"available": True}}, True)
    assert not row.input.isEnabled()
    editor.update_state({"pipeline_serial": editor.edit_serial,
                         "apple_acceleration": {"available": True}}, False)
    assert row.input.isEnabled()
    editor.change(item, "apple_compute", "CPUOnly")
    assert validate_pipeline(editor.document)["software"][1]["params"]["apple_compute"] == "CPUOnly"
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
