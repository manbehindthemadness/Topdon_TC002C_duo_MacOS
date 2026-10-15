"""
Camera output footers remain descriptive, fixed and independent of saved processing nodes.
"""

from pathlib import Path
from typing import Any

import pytest
from support.qt_process import run_popup

from based_thermals.pipeline_ui.outputs import output_sources


@pytest.mark.parametrize("profile, stack, expected", [
    (None, "hardware", "Waiting for camera…"),
    ([], "hardware", "Waiting for camera…"),
    ({"id": "duo"}, "hardware", "Waiting for camera…"),
    ({"preview": True, "radiometry": True}, "hardware", "Thermal upsample · Raw thermal"),
    ({"preview": False, "radiometry": True}, "hardware", "Raw thermal"),
    ({"preview": True, "radiometry": False}, "hardware", "Thermal upsample"),
    ({"preview": False, "radiometry": False}, "hardware", "None available"),
    ({"visible": True}, "visible_hardware", "Color camera"),
    ({"visible": False}, "visible_hardware", "None available"),
])
def test_output_source_names_follow_camera_capabilities(
    profile: Any, stack: str, expected: str,
) -> None:
    """
    Advertise supported output types without inferring them from frame arrival or pipeline choice.
    """
    assert output_sources({"camera_profile": profile, "streams": {}}, stack) == expected


def test_hardware_footer_stays_below_edits_without_becoming_a_node(tmp_path: Path) -> None:
    """
    Exercise real editor refresh, reorder, Clear/New, locks and dual-camera source ownership.
    """
    result = run_popup('''
import json
from copy import deepcopy
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import validate_pipeline

app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
editor.create_new_pipeline()
listing = editor.stacks["hardware"]
footer = listing.output_footer
assert listing.count() == 0 and footer is not None
assert footer.title.text() == "Output sources → Waiting for camera…"
assert editor.stacks["software"].output_footer is None
assert footer.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
assert footer.title.focusPolicy() == Qt.FocusPolicy.NoFocus
state = {"camera_profile": {"id": "duo", "preview": True,
                            "radiometry": True, "visible": False}}
sent = len(messages)
editor.update_state(state, False)
app.processEvents()
assert footer.isVisible()
assert footer.title.text() == "Output sources → Thermal upsample · Raw thermal"
assert len(messages) == sent
QTest.mouseClick(footer.title, Qt.MouseButton.LeftButton)
assert len(messages) == sent
editor.insert("hardware", "brightness", 0)
editor.insert("hardware", "contrast", 1)
app.processEvents()
assert listing.count() == len(editor.document["hardware"]) == 2
assert footer.geometry().top() > listing.visualItemRect(listing.item(1)).bottom()
listing.setCurrentRow(0)
assert listing.drop_row(footer.geometry().center()) == 2
assert listing.model().moveRows(QModelIndex(), 0, 1, QModelIndex(), 2)
app.processEvents()
assert [item["type"] for item in editor.document["hardware"]] == ["contrast", "brightness"]
assert listing.output_footer is footer and footer.isVisible()
assert footer.geometry().top() > listing.visualItemRect(listing.item(1)).bottom()
document = deepcopy(editor.document)
saved = validate_pipeline(json.loads(json.dumps(document)))
assert saved == document and len(saved["hardware"]) == 2
sent = len(messages)
editor.update_state(state, True)
assert footer.isVisible() and len(messages) == sent
editor.update_state(state, False)
editor.clear("hardware")
app.processEvents()
assert listing.output_footer is footer and listing.count() == 0 and footer.isVisible()
assert editor.document["hardware"] == []
editor.create_new_pipeline()
app.processEvents()
assert listing.output_footer is footer and footer.isVisible()
assert listing.count() == 0 and editor.document["hardware"] == []
dual = {"camera_profile": {"id": "virtual-dual", "preview": True,
                           "radiometry": True, "visible": True}}
sent = len(messages)
editor.update_state(dual, False)
app.processEvents()
optical = editor.stacks["visible_hardware"]
assert optical.isVisible() and optical.output_footer.isVisible()
assert optical.output_footer.title.text() == "Output sources → Color camera"
assert footer.title.text() == "Output sources → Thermal upsample · Raw thermal"
assert len(messages) == sent
raw_only = {"camera_profile": {"preview": False, "radiometry": True, "visible": False}}
editor.update_state(raw_only, False)
assert footer.title.text() == "Output sources → Raw thermal"
assert not optical.isVisible()
assert editor.document["hardware"] == []
window.close()
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
