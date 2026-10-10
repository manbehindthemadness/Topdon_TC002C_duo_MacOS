"""
Offscreen custom-node folder, configuration and lock interactions.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_custom_editor_folder_json_publish_lock_and_cancel(tmp_path: Path) -> None:
    """
    Load a package without executing it; publish validated edits and honor locks/cancellation.
    """
    script = """
import json, sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
from topdon_duo.view_window import ViewWindow
root = Path(sys.argv[1])
folder = root / "module"
folder.mkdir()
(folder / "__init__.py").write_text("raise RuntimeError('UI must not import me')\\ndef process(image, config): return image\\n")
(folder / "config.json").write_text('{"gain": 2}')
app = QApplication([])
messages = []
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args[-1])
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
controls = editor.custom_widgets[item["id"]]
QFileDialog.getExistingDirectory = lambda *args: str(folder)
controls.open_folder()
assert item["params"]["name"] == "module"
assert json.loads(item["params"]["config"]) == {"gain": 2}
assert messages[-1]["hardware_operation"] is False
assert "module" in editor.widgets[item["id"]][3].text()
QInputDialog.getMultiLineText = lambda *args: ('{"gain": 3}', True)
controls.edit_json()
assert json.loads(item["params"]["config"]) == {"gain": 3}
config = root / "config.json"
config.write_text('{"gain": 4}')
QFileDialog.getOpenFileName = lambda *args: (str(config), "")
controls.import_json()
assert json.loads(item["params"]["config"]) == {"gain": 4}
count = len(messages)
config.write_text('[]')
controls.import_json()
assert warnings and len(messages) == count
editor.locked = True
controls.refresh()
assert not controls.folder_button.isEnabled()
controls.open_folder()
controls.commit({"config": '{}'})
assert len(messages) == count
editor.locked = False
controls.refresh()
QFileDialog.getExistingDirectory = lambda *args: ''
controls.open_folder()
QInputDialog.getMultiLineText = lambda *args: ('{}', False)
controls.edit_json()
assert len(messages) == count
editor.rebuild()
assert editor.custom_widgets[item["id"]].module_label.text() == "module"
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_embedded_controls_loader_visibility_updates_and_persistence(tmp_path: Path) -> None:
    """
    Render every control kind, hide embedded loaders and keep updates through rebuilds.
    """
    script = '''
import json, sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
from topdon_duo.view_window import ViewWindow
root = Path(sys.argv[1])
folder = root / "module"
folder.mkdir()
data = {"defaults": {"gain": 1.5, "count": 2, "enabled": True, "mode": "A", "name": "hi"},
"controls": [
 {"key":"gain","label":"Gain","type":"number","min":0,"max":3,"step":0.1},
 {"key":"count","label":"Count","type":"integer","min":1,"max":5,"step":1},
 {"key":"enabled","label":"Enabled","type":"boolean"},
 {"key":"mode","label":"Mode","type":"choice","options":["A","B"]},
 {"key":"name","label":"Name","type":"text"}]}
model = {"url": "https://example.com/model", "sha256": "a" * 64}
(folder / "__init__.py").write_text("CONFIG_JSON = " + repr(json.dumps(data)) +
 "\\nMODEL_SOURCE = " + repr(model) + "\\nraise RuntimeError('UI must not import')\\n")
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
controls = editor.custom_widgets[item["id"]]
assert not controls.json_button.isHidden() and not controls.model_button.isHidden()
QFileDialog.getExistingDirectory = lambda *args: str(folder)
controls.open_folder()
assert controls.json_button.isHidden() and controls.model_button.isHidden()
assert not controls.folder_button.isHidden()
assert len(controls.fields.fields) == 5
count = len(messages)
fields = controls.fields.fields
fields["gain"].setValue(2.5)
fields["count"].setValue(4)
fields["enabled"].setChecked(False)
fields["mode"].setCurrentText("B")
fields["name"].setText("updated")
fields["name"].editingFinished.emit()
values = json.loads(item["params"]["config"])
assert values == {"gain":2.5,"count":4,"enabled":False,"mode":"B","name":"updated","model":model}
assert len(messages) == count + 5
controls.refresh()
assert len(messages) == count + 5
editor.rebuild()
controls = editor.custom_widgets[item["id"]]
assert controls.fields.fields["gain"].value() == 2.5
assert controls.json_button.isHidden() and controls.model_button.isHidden()
editor.locked = True
controls.refresh()
controls.change_value("gain", 1)
assert json.loads(item["params"]["config"])["gain"] == 2.5
assert not controls.fields.isEnabled()
editor.locked = False
controls.refresh()
(folder / "__init__.py").write_text("def process(image, config): return image")
controls.open_folder()
assert not controls.json_button.isHidden() and not controls.model_button.isHidden()
assert controls.fields.fields == {}
model_file = root / "model.json"
model_file.write_text(json.dumps(model))
QFileDialog.getOpenFileName = lambda *args: (str(model_file), "")
controls.import_model()
assert json.loads(item["params"]["config"])["model"] == model
config_file = root / "config.json"
config_file.write_text(json.dumps(data))
QFileDialog.getOpenFileName = lambda *args: (str(config_file), "")
controls.import_json()
assert controls.fields.fields["gain"].value() == 1.5
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
