"""
Offscreen color and model-control edits in the real Classless YOLO node.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_color_model_browse_cancel_refresh_and_saved_values(tmp_path: Path) -> None:
    """
    Keep picker values across rebuilds and register browsed paths only on acceptance.
    """
    script = '''
import json, os, sys
from pathlib import Path
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QFileDialog, QColorDialog, QMessageBox
from based_thermals.view_window import ViewWindow
root = Path(sys.argv[1])
cache = root / "models"
cache.mkdir()
os.environ["BASED_THERMALS_MODEL_DIR"] = str(cache)
cached = cache / "cached.onnx"
cached.touch()
external = root / "browsed.onnx"
external.touch()
app = QApplication([])
messages, warnings = [], []
QMessageBox.warning = lambda *args: warnings.append(args[-1])
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
controls = editor.custom_widgets[item["id"]]
folder = Path.cwd() / "examples/custom_nodes/Classless YOLO"
QFileDialog.getExistingDirectory = lambda *args: str(folder)
controls.open_folder()
fields = controls.fields.fields
assert not fields["show_scores"].isChecked()
assert fields["filter_scores"].isChecked()
assert fields["score_maximum"].value() == 1.0
assert fields["nested_coverage"].value() == 0.85
assert fields["suppress_duplicates"].isChecked()
assert fields["nesting"].currentText() == "Outside-in"
assert fields["model_path"].combo.currentData() == ""
assert fields["model_path"].combo.findData(str(cached)) >= 0
border = fields["border_color"]
border.combo.setCurrentIndex(border.combo.findData("dynamic"))
assert json.loads(item["params"]["config"])["border_color"] == "dynamic"
count = len(messages)
QColorDialog.getColor = lambda *args: QColor()
border.choose()
assert len(messages) == count
QColorDialog.getColor = lambda *args: QColor("#123456")
border.choose()
assert json.loads(item["params"]["config"])["border_color"] == "#123456"
model = fields["model_path"]
QFileDialog.getOpenFileName = lambda *args: ("", "")
count = len(messages)
model.browse()
assert len(messages) == count and not (cache / "custom-models.json").exists()
QFileDialog.getOpenFileName = lambda *args: (str(external), "")
model.browse()
assert json.loads(item["params"]["config"])["model_path"] == str(external)
assert json.loads((cache / "custom-models.json").read_text()) == [str(external)]
assert model.combo.currentData() == str(external)
assert not (cache / external.name).exists()
count = len(messages)
controls.refresh()
assert len(messages) == count
editor.rebuild()
controls = editor.custom_widgets[item["id"]]
fields = controls.fields.fields
assert fields["border_color"].combo.currentData() == "#123456"
assert fields["model_path"].combo.currentData() == str(external)
fields["nesting"].setCurrentText("Inside-out")
fields["fill_color"].combo.setCurrentIndex(fields["fill_color"].combo.findData("#ff0000"))
fields["fill_opacity"].setValue(0.4)
fields["show_scores"].setChecked(True)
fields["score_threshold"].setValue(0.3)
fields["score_maximum"].setValue(0.8)
fields["nested_coverage"].setValue(0.9)
fields["duplicate_iou"].setValue(0.6)
fields["suppress_duplicates"].setChecked(False)
values = json.loads(item["params"]["config"])
assert values["nesting"] == "Inside-out" and values["fill_color"] == "#ff0000"
assert values["fill_opacity"] == 0.4 and values["show_scores"] is True
assert values["score_threshold"] == 0.3 and values["score_maximum"] == 0.8
assert values["nested_coverage"] == 0.9 and values["duplicate_iou"] == 0.6
assert values["suppress_duplicates"] is False
missing = root / "missing.onnx"
values["model_path"] = str(missing)
controls.set_config(json.dumps(values))
assert fields["model_path"].combo.currentData() == str(missing)
assert "missing" in fields["model_path"].combo.currentText()
fields["model_path"].combo.setCurrentIndex(0)
assert json.loads(item["params"]["config"])["model_path"] == ""
editor.locked = True
controls.refresh()
assert not fields["model_path"].button.isEnabled()
assert not fields["border_color"].button.isEnabled()
assert not warnings
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_popups_survive_frame_refreshes_and_new_downloads(tmp_path: Path) -> None:
    """
    Keep highlighted choices stable while viewer updates and model downloads continue.
    """
    result = run_popup(
        '''
import os, sys
from pathlib import Path
from PySide6.QtCore import QPersistentModelIndex, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from based_thermals.pipeline_ui.custom_choices import ModelPicker
root = Path(sys.argv[1])
os.environ["BASED_THERMALS_MODEL_DIR"] = str(root)
first = root / "first.onnx"
first.touch()
app = QApplication([])
window = QWidget()
layout = QVBoxLayout(window)
selected = {"detector": "", "pose": ""}
pickers = {}
for name in selected:
    picker = ModelPicker("", lambda value, key=name: selected.update({key: value}))
    pickers[name] = picker
    layout.addWidget(picker)
window.show()
app.processEvents()
for name, picker in pickers.items():
    picker.combo.showPopup()
    app.processEvents()
    view = picker.combo.view()
    assert view.isVisible()
    index = picker.combo.model().index(picker.combo.findData(str(first)), 0)
    highlighted = QPersistentModelIndex(index)
    view.setCurrentIndex(index)
    for _ in range(5):
        for key, field in pickers.items():
            field.refresh(selected[key])
        app.processEvents()
    assert highlighted.isValid(), "Frame refresh discarded popup rows"
    assert view.currentIndex() == index, "Frame refresh reset the highlighted model"
    downloaded = root / (name + "-downloaded.onnx")
    downloaded.touch()
    picker.refresh(selected[name])
    assert highlighted.isValid() and view.currentIndex() == index
    QTest.keyClick(view, Qt.Key.Key_Return)
    app.processEvents()
    assert selected[name] == str(first)
    picker.refresh(selected[name])
    assert picker.combo.currentData() == str(first)
    picker.combo.showPopup()
    app.processEvents()
    assert picker.combo.findData(str(downloaded)) >= 0
    picker.combo.hidePopup()
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
