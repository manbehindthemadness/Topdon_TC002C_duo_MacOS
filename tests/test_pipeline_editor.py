"""Pipeline behavior: ordered image operations, safe ownership, and portable state."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture
from support.pipeline import raw_pipeline

from topdon_duo.pipeline import (
    default_pipeline,
    node,
)


def test_pipeline_editor_context_operations_import_export_and_lock(tmp_path: Path) -> None:
    import subprocess
    import sys

    from test_capture_panel import popup_environment

    script = """
import json, sys
from copy import deepcopy
from pathlib import Path
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMenu
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node, default_pipeline
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
state = {"pipeline": default_pipeline(), "pipeline_serial": 12, "processing_preset_available": True,
         "hardware": {"brightness": {"value": 57, "available": True}}}
window.update_state(state)
assert messages == []
editor.insert("hardware", "brightness", 0)
assert editor.document["software"][0]["type"] == "source"
assert editor.document["hardware"][0]["params"]["value"] == 57
assert messages[-1]["serial"] == 13
count = len(messages)
editor.insert("hardware", "brightness", 2)
assert len(messages) == count
# Metadata-only changes cannot be mistaken for camera writes.
item = editor.document["hardware"][0]
_, controls, bypass, title, badge = editor.widgets[item["id"]]
title.click()
assert not item["expanded"]
assert messages[-1]["hardware_operation"] is False
# A queued old state must not overwrite newer edits.
window.update_state(state)
assert editor.document["hardware"][0]["params"]["value"] == 57
editor.insert("software", "gamma", 0)
editor.insert("software", "gamma", 1)
assert sum(n["type"] == "gamma" for n in editor.document["software"]) == 2
software = editor.stacks["software"]
first_id = editor.document["software"][1]["id"]
# Exercise the model move that drag/drop uses, including editor callbacks.
assert software.model().moveRows(QModelIndex(), 1, 1, QModelIndex(), 3)
app.processEvents()
assert editor.document["software"][2]["id"] == first_id
assert messages[-1]["hardware_operation"] is False
menu = QMenu()
editor.stacks["hardware"].add_menu(menu, "Add", 2)
brightness = next(a for a in menu.actions()[0].menu().actions() if a.text() == "Camera brightness")
assert not brightness.isEnabled()
for menu_title in ("Add node", "Insert node above", "Insert node below"):
    software_menu = QMenu()
    software.add_menu(software_menu, menu_title, 1)
    labels = {a.text() for a in software_menu.actions()[0].menu().actions()}
    assert "AI enhancement" in labels and "AI image styling" in labels
    assert not any("experimental" in label or "legacy" in label for label in labels)
    assert not any(label.startswith("ONNX ") or label.startswith("Apple Core ML ACNet") for label in labels)
# Bypassed singleton still cannot be duplicated.
item["bypass"] = True
editor.publish()
editor.insert("hardware", "brightness", 2)
assert sum(n["type"] == "brightness" for n in editor.document["hardware"]) == 1
# Only pipeline data is exported.
destination = Path(sys.argv[1]) / "saved.pipeline.json"
QFileDialog.getSaveFileName = lambda *_args: (str(destination), "")
editor.export_file()
exported = json.loads(destination.read_text())
assert set(exported) == {"version", "hardware", "software", "branches"}
assert exported == editor.document
QFileDialog.getOpenFileName = lambda *_args: (str(destination), "")
editor.clear()
assert not editor.document["hardware"] and len(editor.document["software"]) == 2
editor.import_file()
assert editor.document == exported
before = deepcopy(editor.document)
destination.write_text('{"version":99}')
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
editor.import_file()
assert warnings and editor.document == before
# Logging cancels timers and blocks all mutations, including programmatic calls.
for _, rows, *_ in editor.widgets.values():
    for row in rows.values(): row.timer.start()
editor.update_state({"pipeline_serial": editor.edit_serial, "pipeline": editor.document, "processing_preset_available": True}, True)
count = len(messages)
for _, rows, bypass, title, badge in editor.widgets.values():
    for row in rows.values():
        assert not row.input.isEnabled() and not row.timer.isActive()
editor.clear()
editor.insert("software", "gamma", 0)
editor.remove("software", 0)
assert editor.document == before and len(messages) == count
editor.update_state({"pipeline_serial": editor.edit_serial, "pipeline": editor.document, "processing_preset_available": True}, False)
editor.clear("software")
assert len(editor.document["software"]) == 2 and len(editor.document["hardware"]) > 0
editor.clear("hardware")
assert not editor.document["hardware"]
editor.remove("software", 0)
assert editor.document["software"][0]["type"] == "source"
# Tabs edit independent stacks; A keeps its fixed output and clear honors tab scope.
assert editor.tab_bar.count() == 4
snapshot_a = deepcopy(editor.document["software"])
editor.tab_bar.setCurrentIndex(1)
assert editor.tab == "B" and len(editor.nodes("software")) == 1
assert "idle" in editor.tab_status.text()
editor.insert("software", "brightness", 1)
assert editor.document["branches"]["B"][1]["type"] == "brightness"
assert editor.document["software"] == snapshot_a
brightness_b = editor.nodes("software")[1]
brightness_row = editor.widgets[brightness_b["id"]][1]["amount"]
brightness_row.input.setValue(12.3)
assert brightness_row.timer.isActive()
editor.tab_bar.setCurrentIndex(0)
assert editor.document["branches"]["B"][1]["params"]["amount"] == 12.3
editor.insert("software", "combine", 999)
assert editor.document["software"][-2]["type"] == "combine"
assert editor.document["software"][-1]["type"] == "output"
output = editor.document["software"][-1]
editor.bypass(output, True)
editor.remove("software", len(editor.document["software"]) - 1)
assert editor.document["software"][-1] == output and not output["bypass"]
editor.tab_bar.setCurrentIndex(1)
assert "connected" in editor.tab_status.text()
editor.insert("software", "combine", 2)
combine = editor.nodes("software")[-1]
editor.change(combine, "tab", "A")
assert "cycle" in warnings[-1][-1]
assert editor.nodes("software")[-1]["params"]["tab"] == "C"
editor.clear("software")
assert len(editor.document["branches"]["B"]) == 1
assert editor.document["software"][-2]["type"] == "combine"
# Updates while B is visible must still update hidden A and C/D.
incoming = deepcopy(editor.document)
incoming["software"][-2]["params"]["opacity"] = 0.7
incoming["branches"]["D"].append(node("software", "gamma"))
editor.update_state({"pipeline": incoming, "pipeline_serial": editor.edit_serial + 1,
                     "processing_preset_available": True}, False)
assert editor.document == incoming
editor.tab_bar.setCurrentIndex(3)
assert editor.nodes("software")[1]["type"] == "gamma"
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial,
                     "processing_preset_available": True}, True)
locked_snapshot = deepcopy(editor.document)
editor.insert("software", "filter", 1)
editor.clear()
assert editor.document == locked_snapshot
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial,
                     "processing_preset_available": True}, False)
editor.tab_bar.setCurrentIndex(0)
window.grab().save(str(Path(sys.argv[1]) / "pipeline-editor.png"))
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_desktop_pipeline_command_persists_and_is_locked_while_logging(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo import desktop
    from topdon_duo.settings_preferences import load_settings

    document = raw_pipeline(node("software", "mirror", horizontal=True))
    document["software"][1]["expanded"] = True
    panel = Mock()
    panel.poll.return_value = [{"action": "pipeline", "document": document, "serial": 1}]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    assert load_settings()["pipeline"] == document
    panel.poll.return_value = []
    assert desktop.main([]) == 0
    assert panel.update.call_args.args[0]["pipeline"] == document
    candidate = default_pipeline()
    panel.poll.return_value = [{"action": "pipeline", "document": candidate, "serial": 2}]
    viewer.graphs.logging = True
    assert desktop.main([]) == 0
    assert load_settings()["pipeline"] == document


__all__ = ["viewer_fixture"]
