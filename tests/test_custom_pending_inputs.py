"""
Preserve unfinished Custom text across runtime echoes and commit it in pipeline snapshots.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_custom_text_survives_refresh_and_exports_current_bundled_edits(tmp_path: Path) -> None:
    """
    Exercise focused typing, export, saved overrides, authoritative values and locks in Qt.
    """
    script = '''
import json, sys
from copy import deepcopy
from pathlib import Path
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog
from based_thermals.pipeline_presets import load_presets, predefined_presets
from based_thermals.pipeline_ui.transfer.dialogs import ExportDialog
from based_thermals.pipeline_ui.transfer.documents import read_pipelines
from based_thermals.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
name = "Yautja GPU"
bundled = deepcopy(predefined_presets()[name])
editor.select_preset(editor.preset_combo.findText(name))
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
item["expanded"] = True
item["params"].update(
    name="Text example",
    package=json.dumps({"__init__.py": "raise RuntimeError('Must not execute in UI')\\ndef process(image, config): return image\\n"}),
    config=json.dumps({"label": "old", "count": 1}),
    controls=json.dumps([
        {"key": "label", "label": "Label", "type": "text"},
        {"key": "count", "label": "Count", "type": "integer", "min": 1, "max": 5, "step": 1},
    ]),
)
editor.rebuild()
editor.publish()
window.show()
field = editor.custom_widgets[item["id"]].fields.fields["label"]
field.setFocus()
app.processEvents()
assert field.hasFocus()
field.selectAll()
QTest.keyClicks(field, "revised")
assert field.isModified()
assert json.loads(item["params"]["config"])["label"] == "old"
count = len(messages)
for _ in range(3):
    editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
assert field.text() == "revised" and field.isModified()
assert len(messages) == count
editor.custom_widgets[item["id"]].fields.fields["count"].setValue(2)
assert field.text() == "revised" and field.isModified()
assert json.loads(item["params"]["config"]) == {"label": "old", "count": 2}
path = Path(sys.argv[1]) / "revised.pipeline.json"
QFileDialog.getSaveFileName = lambda *_: (str(path), "")
ExportDialog.exec = lambda _: QDialog.DialogCode.Accepted
editor.export_file()
exported = {entry.name: entry.document for entry in read_pipelines(path)}
assert exported[name] == editor.document
assert json.loads(exported[name]["software"][1]["params"]["config"])["label"] == "revised"
assert editor.presets[name] == bundled and predefined_presets()[name] == bundled
assert not field.isModified()
field.editingFinished.emit()
# Finishing an already committed edit may publish again, but cannot change its value.
assert json.loads(item["params"]["config"])["label"] == "revised"
field.setFocus()
field.selectAll()
QTest.keyClicks(field, "saved edit")
editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, False)
editor.save_current_preset()
saved = load_presets()[name]
assert json.loads(saved["software"][1]["params"]["config"])["label"] == "saved edit"
assert predefined_presets()[name] == bundled
# Explicit JSON replacements supersede pending text without publishing it first.
field.selectAll()
QTest.keyClicks(field, "discard this")
editor.custom_widgets[item["id"]].set_config('{"label":"from JSON","count":2}')
assert field.text() == "from JSON" and not field.isModified()
field.selectAll()
QTest.keyClicks(field, "replace same-value draft")
editor.custom_widgets[item["id"]].set_config('{"label":"from JSON","count":2}')
assert field.text() == "from JSON" and not field.isModified()
# A lock discards an uncommitted draft and forbids save-time publication.
field.selectAll()
QTest.keyClicks(field, "locked draft")
editor.update_state({"pipeline": deepcopy(editor.document), "pipeline_serial": editor.edit_serial}, True)
assert field.text() == "from JSON"
before = deepcopy(editor.document)
count = len(messages)
editor.commit_pending_inputs()
assert editor.document == before and len(messages) == count
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
