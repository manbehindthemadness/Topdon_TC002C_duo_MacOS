"""
Expose the Head Pose detector choice through literal JSON without model work in the UI.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_detector_choices_reach_editor_and_persist(tmp_path: Path) -> None:
    """
    Load the actual node folder, edit its detector choice and round-trip the saved document.
    """
    result = run_popup(
        '''
import json
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
from based_thermals.view_window import ViewWindow
from based_thermals.custom_nodes.models import ModelResolver
from based_thermals.pipeline import validate_pipeline

def forbidden(*args, **kwargs):
    raise AssertionError("UI must not resolve or download models")
ModelResolver.resolve = forbidden
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
controls = editor.custom_widgets[item["id"]]
folder = Path.cwd() / "examples/custom_nodes/head-pose/nodes/Head Pose"
QFileDialog.getExistingDirectory = lambda *args: str(folder)
controls.open_folder()
picker = controls.fields.fields["detector_model"]
assert [picker.itemText(i) for i in range(picker.count())] == [
    "HGNet-N (fast)", "DINOv3-S (reference)"]
assert picker.currentText() == "HGNet-N (fast)"
assert json.loads(item["params"]["config"])["detector_path"] == ""
assert json.loads(item["params"]["config"])["pose_path"] == ""
picker.setCurrentText("DINOv3-S (reference)")
assert json.loads(item["params"]["config"])["detector_model"] == "DINOv3-S (reference)"
saved = validate_pipeline(json.loads(json.dumps(editor.document)))
assert saved["software"][1]["params"]["config"] == item["params"]["config"]
editor.update_state({"pipeline_serial": editor.edit_serial}, True)
assert not picker.isEnabled()
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
