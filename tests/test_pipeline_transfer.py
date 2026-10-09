"""
Portable pipeline bundles, batch checklists and import conflict resolution.
"""

import json
from pathlib import Path

import pytest
from support.qt_process import run_popup

from topdon_duo.pipeline import default_pipeline, node
from topdon_duo.pipeline_ui.transfer.documents import (
    BUNDLE_FORMAT,
    read_pipelines,
    unique_name,
    write_pipelines,
)


def test_named_bundle_and_legacy_pipeline_round_trip(tmp_path: Path) -> None:
    """
    Preserve names, all branches and settings, while accepting old individual files.
    """
    original = default_pipeline()
    original["branches"]["B"].append(node("software", "gamma", amount=1.7))
    documents = {"Thermal α": original, "Other pipeline": default_pipeline()}
    path = tmp_path / "batch.pipeline.json"
    write_pipelines(path, documents, single=False)
    assert json.loads(path.read_text())["format"] == BUNDLE_FORMAT
    loaded = read_pipelines(path)
    assert {item.name: item.document for item in loaded} == documents
    assert all(item.source == path.name for item in loaded)
    path = tmp_path / "My pipeline.pipeline.json"
    write_pipelines(path, {"Ignored legacy name": original}, single=True)
    assert json.loads(path.read_text()) == original
    loaded = read_pipelines(path)
    assert loaded[0].name == "My pipeline" and loaded[0].document == original
    assert unique_name("Test", {"Test", "Test (2)"}) == "Test (3)"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"version": 99},
        {"format": BUNDLE_FORMAT, "version": True, "pipelines": {}},
        {"format": BUNDLE_FORMAT, "version": 2, "pipelines": {}},
        {"format": BUNDLE_FORMAT, "version": 1, "pipelines": []},
        {"format": BUNDLE_FORMAT, "version": 1, "pipelines": {}},
        {"format": BUNDLE_FORMAT, "version": 1, "pipelines": {" ": {}}},
        {"format": BUNDLE_FORMAT, "version": 1, "pipelines": {" Bad ": {}}},
        {"format": BUNDLE_FORMAT, "version": 1, "pipelines": {"Broken": None}},
    ],
)
def test_invalid_bundles_are_rejected(tmp_path: Path, payload: object) -> None:
    """
    Reject unsupported versions, bad shapes, invalid names and invalid documents.
    """
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        read_pipelines(path)


def test_export_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Clean temporary output and retain the destination if atomic replacement fails.
    """
    path = tmp_path / "preserve.pipeline.json"
    path.write_text("old contents")

    def fail_replace(_source: Path, _destination: Path) -> None:
        """
        Simulate a filesystem error at the commit boundary.
        """
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        write_pipelines(path, {"New": default_pipeline()}, single=False)
    assert path.read_text() == "old contents"
    assert not path.with_suffix(".json.tmp").exists()


def test_batch_dialogs_import_export_conflicts_and_failures(tmp_path: Path) -> None:
    """
    Exercise real checkbox controls and batch actions in a separate Qt process.
    """
    script = """
import json, sys
from copy import deepcopy
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QMessageBox
from topdon_duo.pipeline import default_pipeline, node
from topdon_duo.pipeline_presets import load_presets, save_presets
from topdon_duo.pipeline_ui.transfer import actions
from topdon_duo.pipeline_ui.transfer.dialogs import ExportDialog, ImportDialog
from topdon_duo.pipeline_ui.transfer.documents import IncomingPipeline, read_pipelines, write_pipelines
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
folder = Path(sys.argv[1])
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
existing = "Yautja"
changed = default_pipeline()
changed["software"].insert(1, node("software", "gamma", amount=2.2))
incoming = [IncomingPipeline(existing, changed, "one.json"),
            IncomingPipeline(existing, default_pipeline(), "two.json"),
            IncomingPipeline("Fresh", changed, "three.json")]
dialog = ImportDialog(editor, incoming, set(editor.presets))
assert [field.text() for _, field, _ in dialog.rows] == ["Yautja (2)", "Yautja (3)", "Fresh"]
assert set(dialog.selected_pipelines()) == {"Yautja (2)", "Yautja (3)", "Fresh"}
dialog.rows[0][1].setText(existing)
assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
dialog.rows[0][2].setCurrentText("Replace")
assert existing in dialog.selected_pipelines()
dialog.rows[1][2].setCurrentText("Replace")
assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
dialog.rows[1][2].setCurrentText("Skip")
assert set(dialog.selected_pipelines()) == {existing, "Fresh"}
dialog.rows[2][1].setText(" ")
assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
dialog.rows[2][1].setText("Fresh")
dialog.check_all(False)
assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
dialog.check_all(True)
assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
dialog.close()
# Export multiple named presets; only checked documents are included.
destination = folder / "export.pipeline.json"
QFileDialog.getSaveFileName = lambda *_: (str(destination), "")
def select_exports(self: ExportDialog) -> QDialog.DialogCode:
    '''
    Choose two saved presets using the actual checklist controls.
    '''
    self.check_all(False)
    assert not self.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    for row in range(self.listing.count()):
        if self.listing.item(row).data(Qt.ItemDataRole.UserRole) in ("Yautja", "Yautja GPU"):
            self.listing.item(row).setCheckState(Qt.CheckState.Checked)
    return QDialog.DialogCode.Accepted
ExportDialog.exec = select_exports
editor.export_file()
exported = {item.name: item.document for item in read_pipelines(destination)}
assert set(exported) == {"Yautja", "Yautja GPU"}
assert exported == {name: editor.presets[name] for name in exported}
# Pending input edits are included in the current pipeline snapshot.
editor.create_new_pipeline()
editor.insert("software", "gamma", 1)
gamma = editor.document["software"][1]
row = editor.widgets[gamma["id"]][1]["amount"]
row.input.setValue(2.6)
row.timer.start()
ExportDialog.exec = lambda _: QDialog.DialogCode.Accepted
editor.export_file()
assert json.loads(destination.read_text())["software"][1]["params"]["amount"] == 2.6
# Review multiple files, including a malformed file, before any import is saved.
bundle = folder / "incoming.pipeline.json"
write_pipelines(bundle, {existing: changed, "Fresh": default_pipeline()}, single=False)
legacy = folder / "Legacy.pipeline.json"
write_pipelines(legacy, {"Legacy": changed}, single=True)
bad = folder / "bad.json"
bad.write_text('{"version":99}')
QFileDialog.getOpenFileNames = lambda *_: ([str(bundle), str(legacy), str(bad)], "")
ImportDialog.exec = lambda _: QDialog.DialogCode.Rejected
before = deepcopy(load_presets())
working = deepcopy(editor.document)
editor.import_file()
assert warnings and load_presets() == before and editor.document == working
def select_imports(self: ImportDialog) -> QDialog.DialogCode:
    '''
    Replace one conflict, skip an unchecked row and rename a legacy file.
    '''
    self.rows[0][2].setCurrentText("Replace")
    self.rows[1][0].setChecked(False)
    self.rows[2][1].setText("Renamed legacy")
    self.load_first.setChecked(False)
    return QDialog.DialogCode.Accepted
ImportDialog.exec = select_imports
editor.import_file()
assert load_presets()[existing] == changed
assert "Fresh" not in load_presets()
assert load_presets()["Renamed legacy"] == changed
assert editor.document == working
# Default keep-both names survive restart and the first selection can be loaded.
QFileDialog.getOpenFileNames = lambda *_: ([str(bundle)], "")
ImportDialog.exec = lambda _: QDialog.DialogCode.Accepted
editor.import_file()
assert editor.current_preset_name == "Yautja (2)"
assert editor.document == changed and messages[-1]["document"] == changed
assert load_presets()["Yautja (2)"] == changed
assert "Fresh" in load_presets()
reopened = ViewWindow(lambda _: None)
assert reopened.pipeline_editor.presets == load_presets()
reopened.close()
# A failed batch save changes neither presets nor the active pipeline.
before = deepcopy(load_presets())
working = deepcopy(editor.document)
current = editor.current_preset_name
original_save = actions.save_presets
def failed_save(_candidate: dict) -> None:
    '''
    Fail before committing the imported collection.
    '''
    raise OSError("disk full")
actions.save_presets = failed_save
editor.import_file()
assert load_presets() == before and editor.presets == before
assert editor.document == working and editor.current_preset_name == current
actions.save_presets = original_save
editor.locked = True
editor.import_file()
assert load_presets() == before
editor.locked = False
# Cancellations leave output files and saved pipelines untouched.
old_output = destination.read_text()
ExportDialog.exec = lambda _: QDialog.DialogCode.Rejected
editor.export_file()
assert destination.read_text() == old_output
ExportDialog.exec = lambda _: QDialog.DialogCode.Accepted
QFileDialog.getSaveFileName = lambda *_: ("", "")
editor.export_file()
assert destination.read_text() == old_output
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
