"""
User-managed pipeline branches, persistent identities and protected connections.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from support.frames import frame_with_preview
from support.qt_process import run_popup

from topdon_duo.pipeline import default_pipeline, node, validate_pipeline
from topdon_duo.pipeline_model.tabs import add_pipeline_tab, remove_pipeline_tab, tab_names
from topdon_duo.pipeline_processing import PipelineProcessor


def test_new_pipeline_and_dynamic_labels_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Start with A and retain branches beyond D/Z through preferences and validation.
    """
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = default_pipeline()
    assert tab_names(document) == ["A"]
    assert validate_pipeline(document) == document
    for _ in range(27):
        add_pipeline_tab(document)
    assert tab_names(document)[-2:] == ["AA", "AB"]
    document["software"].insert(1, node("software", "combine", tab="AB", mask_source="AA"))
    assert validate_pipeline(document) == document
    save_settings({"pipeline": document})
    assert load_settings()["pipeline"] == document
    remove_pipeline_tab(document, "B")
    assert "C" in document["branches"]
    assert add_pipeline_tab(document) == "B"


def test_dynamic_blend_mask_processing_and_removed_branch_cleanup() -> None:
    """
    Process AA/AB through real blend/mask scheduling and close their deleted caches.
    """
    frame, _ = frame_with_preview()
    document = default_pipeline()
    document["branches"] = {
        tab: [node("software", "source"), node("software", "brightness", amount=100)]
        for tab in ("AA", "AB")
    }
    document["software"] = [
        node("software", "source"),
        node("software", "combine", tab="AB", mask_source="AA", opacity=1),
        node("software", "output"),
    ]
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        image, _ = processor.process(frame, None, document, scale=1)
        assert np.all(image == 255)
        assert set(processor.executors) == {"A", "AA", "AB"}
        document["software"].pop(1)
        remove_pipeline_tab(document, "AA")
        remove_pipeline_tab(document, "AB")
        processor.process(frame, None, document, scale=1)
        assert set(processor.executors) == {"A"}
        assert set(processor.branches) == {"A"}
    finally:
        processor.close()


@pytest.mark.parametrize("owner,target,key,bypassed", [
    ("A", "B", "tab", False), ("A", "B", "mask_source", False),
    ("B", "C", "tab", False), ("C", "B", "mask_source", True),
])
def test_remove_rejects_saved_connections(
    owner: str, target: str, key: str, bypassed: bool
) -> None:
    """
    Protect incoming/outgoing connections and bypassed mask references from deletion.
    """
    document = default_pipeline()
    add_pipeline_tab(document)
    add_pipeline_tab(document)
    item = node("software", "combine", **{key: target})
    item["bypass"] = bypassed
    nodes = document["software"] if owner == "A" else document["branches"][owner]
    nodes.insert(1, item)
    before = deepcopy(document)
    with pytest.raises(ValueError, match="connected"):
        remove_pipeline_tab(document, "B")
    assert document == before
    nodes.remove(item)
    remove_pipeline_tab(document, "B")
    assert "B" not in document["branches"]


@pytest.mark.parametrize("key", ["tab", "mask_source"])
def test_missing_tab_reference_rejected_even_when_bypassed(key: str) -> None:
    """
    Reject dangling references rather than silently interpreting them as native input.
    """
    document = default_pipeline()
    item = node("software", "combine", **{key: "E"})
    item["bypass"] = True
    document["software"].insert(1, item)
    with pytest.raises(ValueError, match="Unknown input tab"):
        validate_pipeline(document)
    with pytest.raises(ValueError, match="viewer output"):
        remove_pipeline_tab(document, "A")


def test_dynamic_tab_controls_connections_import_and_locks(tmp_path: Path) -> None:
    """
    Exercise plus/minus, live choices, locked edits and replacement of a selected tab.
    """
    script = '''
from copy import deepcopy
from PySide6.QtWidgets import QApplication, QMessageBox
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import default_pipeline, node
app = QApplication([])
messages = []
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
window = ViewWindow(messages.append)
editor = window.pipeline_editor
assert editor.tab_bar.count() == 1 and editor.tab == "A"
assert not editor.remove_tab_button.isEnabled()
for _ in range(4):
    editor.add_tab_button.click()
assert editor.tab == "E" and editor.tab_bar.count() == 5
editor.tab_bar.setCurrentIndex(0)
count = len(messages)
editor.insert("software", "combine", 1)
blend = editor.nodes("software")[1]
controls = editor.widgets[blend["id"]][1]
assert {controls["tab"].input.itemData(i) for i in range(controls["tab"].input.count())} == {"preview", "raw", "B", "C", "D", "E"}
editor.change(blend, "mask_source", "E")
editor.bypass(blend, True)
editor.tab_bar.setCurrentIndex(4)
assert not editor.remove_tab_button.isEnabled()
snapshot = deepcopy(editor.document)
editor.remove_tab()
assert warnings and editor.document == snapshot
editor.tab_bar.setCurrentIndex(0)
editor.remove("software", 1)
editor.tab_bar.setCurrentIndex(1)
assert editor.remove_tab_button.isEnabled()
editor.remove_tab_button.click()
assert editor.tab == "A" and list(editor.document["branches"]) == ["C", "D", "E"]
assert [editor.tab_bar.tabData(i) for i in range(editor.tab_bar.count())] == ["A", "C", "D", "E"]
editor.insert("software", "combine", 1)
choices = editor.widgets[editor.nodes("software")[1]["id"]][1]["mask_source"].input
assert "B" not in {choices.itemData(i) for i in range(choices.count())}
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, True)
assert not editor.add_tab_button.isEnabled() and not editor.remove_tab_button.isEnabled()
snapshot = deepcopy(editor.document)
editor.add_tab()
editor.remove_tab()
assert editor.document == snapshot
editor.update_state({"pipeline_serial": editor.edit_serial}, False)
editor.tab_bar.setCurrentIndex(3)
replacement = default_pipeline()
editor.update_state({"pipeline": replacement, "pipeline_serial": editor.edit_serial + 1}, False)
assert editor.tab == "A" and editor.tab_bar.count() == 1
assert editor.document == replacement
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
