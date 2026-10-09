"""
Recognize preset processing settings and rename saved presets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import QInputDialog, QMessageBox

from ..pipeline_presets import save_presets

if TYPE_CHECKING:
    from .editor import PipelineEditor


def pipeline_content(document: dict[str, Any]) -> dict[str, Any]:
    """
    Exclude node identities and expansion state when recognizing a loaded preset.
    """

    def content(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Retain ordered node settings, including bypass state.
        """
        return [
            {key: value for key, value in item.items() if key not in ("id", "expanded")}
            for item in nodes
        ]

    result = {
        "version": document["version"],
        "hardware": content(document["hardware"]),
        "software": content(document["software"]),
        "branches": {tab: content(nodes) for tab, nodes in document["branches"].items()},
    }
    return result


def rename_current_preset(editor: PipelineEditor) -> None:
    """
    Rename the saved preset while preserving the current working pipeline and edits.
    """
    old_name = editor.current_preset_name
    if editor.locked or old_name is None:
        return
    name, accepted = QInputDialog.getText(
        editor, "Rename pipeline preset", "Preset name:", text=old_name
    )
    name = name.strip()
    if not accepted or not name or name == old_name:
        return
    if (
        name in editor.presets
        and QMessageBox.question(editor, "Replace preset", f'Replace the saved preset "{name}"?')
        != QMessageBox.StandardButton.Yes
    ):
        return
    candidate = {key: document for key, document in editor.presets.items() if key != old_name}
    candidate[name] = editor.presets[old_name]
    try:
        save_presets(candidate)
    except (OSError, ValueError, TypeError) as exc:
        QMessageBox.warning(editor, "Could not rename preset", str(exc))
        return
    editor.presets = candidate
    editor.current_preset_name = name
    editor.recognize_preset = True
    editor.refresh_presets()
