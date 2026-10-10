"""
Optional Custom device controls preserve preferences through fallback, edits and locks.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_custom_device_editor_capabilities_persistence_and_locks(tmp_path: Path) -> None:
    """
    Exercise the real editor without hardware probes, session creation or package execution.
    """
    result = run_popup(
        '''
import json, sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
from topdon_duo.view_window import ViewWindow
from topdon_duo.custom_nodes import devices
from topdon_duo.pipeline import validate_pipeline

def forbidden():
    raise AssertionError("UI must not probe devices")
devices.apple_acceleration.apple_acceleration = forbidden
devices.nvidia_acceleration.nvidia_acceleration = forbidden
root = Path(sys.argv[1])
folder = root / "device node"
folder.mkdir()
saved = {"backend": "coreml", "apple_compute": "ALL"}
data = {"defaults": {"inference": saved}, "controls": [
    {"key": "inference", "label": "Inference device", "type": "device"}]}
(folder / "__init__.py").write_text("CONFIG_JSON = " + repr(data) +
    "\\nraise AssertionError('UI must not import package')\\n")
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
controls = editor.custom_widgets[item["id"]]
QFileDialog.getExistingDirectory = lambda *args: str(folder)
controls.open_folder()
picker = controls.fields.fields["inference"]
count = len(messages)
assert picker.backend.currentData() == "cpu" and not picker.backend.isEnabled()
assert picker.status.text() == "CPU execution · saved GPU settings retained"
assert json.loads(item["params"]["config"])["inference"] == saved

def update(apple, nvidia, locked=False):
    editor.update_state({"pipeline_serial": editor.edit_serial,
        "apple_acceleration": {"available": apple},
        "nvidia_acceleration": {"available": nvidia}}, locked)

update(True, False)
assert picker.backend.currentData() == "coreml" and picker.compute.currentData() == "ALL"
assert picker.compute.isEnabled() and not picker.compute.isHidden()
assert not picker.backend.model().item(picker.backend.findData("cuda")).isEnabled()
assert len(messages) == count
picker.backend.setCurrentIndex(picker.backend.findData("cuda"))
assert picker.backend.currentData() == "coreml" and len(messages) == count
picker.backend.setCurrentIndex(picker.backend.findData("cpu"))
assert json.loads(item["params"]["config"])["inference"] == {"backend": "cpu", "apple_compute": "ALL"}
assert picker.compute.currentData() == "CPUOnly" and not picker.compute.isEnabled()
count = len(messages)
picker.compute.setCurrentIndex(picker.compute.findData("CPUAndGPU"))
assert len(messages) == count and picker.compute.currentData() == "CPUOnly"
picker.backend.setCurrentIndex(picker.backend.findData("coreml"))
assert picker.compute.currentData() == "ALL"
picker.compute.setCurrentIndex(picker.compute.findData("CPUAndNeuralEngine"))
saved = {"backend": "coreml", "apple_compute": "CPUAndNeuralEngine"}
assert json.loads(item["params"]["config"])["inference"] == saved
count = len(messages)
update(False, True)
assert picker.backend.currentData() == "cuda" and picker.compute.isHidden()
assert json.loads(item["params"]["config"])["inference"] == saved
update(False, False)
assert picker.backend.currentData() == "cpu"
update(True, False, True)
assert not picker.backend.isEnabled() and not picker.compute.isEnabled()
picker.change_backend(0)
picker.change_compute(0)
assert json.loads(item["params"]["config"])["inference"] == saved
assert len(messages) == count
update(True, False)
editor.rebuild()
controls = editor.custom_widgets[item["id"]]
picker = controls.fields.fields["inference"]
assert picker.backend.currentData() == "coreml"
assert picker.compute.currentData() == "CPUAndNeuralEngine"
values = json.loads(item["params"]["config"])
values["inference"] = {"backend": "cpu", "apple_compute": "CPUAndGPU"}
controls.set_config(json.dumps(values))
assert picker.backend.currentData() == "cpu" and picker.compute.currentData() == "CPUOnly"
serialized = json.loads(json.dumps(editor.document))
assert validate_pipeline(serialized)["software"][1]["params"]["config"] == item["params"]["config"]
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
