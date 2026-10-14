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
import json, os, sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
from based_thermals.view_window import ViewWindow
from based_thermals.custom_nodes.models import ModelResolver
from based_thermals.pipeline import validate_pipeline

def forbidden(*args, **kwargs):
    raise AssertionError("UI must not resolve or download models")
ModelResolver.resolve = forbidden
root = Path(sys.argv[1])
os.environ["BASED_THERMALS_MODEL_DIR"] = str(root)
detector_file = root / "custom-source-155c3c6f37dc.onnx"
pose_file = root / "custom-source-be054658797c.onnx"
unrelated = root / "acnet-legacy-hdn0.onnx"
for path in (detector_file, pose_file, unrelated):
    path.touch()
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
detector = controls.fields.fields["detector_path"]
pose = controls.fields.fields["pose_path"]
assert detector.combo.findData(str(detector_file)) >= 0
assert detector.combo.findData(str(pose_file)) == -1
assert pose.combo.findData(str(pose_file)) >= 0
assert pose.combo.findData(str(detector_file)) == -1
assert detector.combo.findData(str(unrelated)) == pose.combo.findData(str(unrelated)) == -1
assert detector.combo.itemText(detector.combo.findData(str(detector_file))) == "HGNet-N head detector"
assert pose.combo.itemText(pose.combo.findData(str(pose_file))) == "HFHPE ViT-T 64×64 (rollgate v10)"
downloaded = root / "custom-future-source-9b64e68cd6cf.onnx"
downloaded.touch()
detector.combo.showPopup()
assert detector.combo.itemText(detector.combo.findData(str(downloaded))) == "DINOv3-S head detector"
detector.combo.hidePopup()
detector.combo.setCurrentIndex(detector.combo.findData(str(detector_file)))
pose.combo.setCurrentIndex(pose.combo.findData(str(pose_file)))
assert json.loads(item["params"]["config"])["detector_path"] == str(detector_file)
assert json.loads(item["params"]["config"])["pose_path"] == str(pose_file)
picker.setCurrentText("DINOv3-S (reference)")
assert json.loads(item["params"]["config"])["detector_model"] == "DINOv3-S (reference)"
saved = validate_pipeline(json.loads(json.dumps(editor.document)))
assert saved["software"][1]["params"]["config"] == item["params"]["config"]
saved_controls = json.loads(saved["software"][1]["params"]["controls"])
saved_pose = next(control for control in saved_controls if control["key"] == "pose_path")
assert saved_pose["models"][0]["name"] == "HFHPE ViT-T 64×64 (rollgate v10)"
editor.update_state({"pipeline_serial": editor.edit_serial}, True)
assert not picker.isEnabled()
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
