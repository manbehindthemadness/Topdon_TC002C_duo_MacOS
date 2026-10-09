"""
Reviewable batch transfers across file dialogs, checklists and preset persistence.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from ...dialog_preferences import load_dialog_directory, remember_dialog_directory
from ...pipeline_presets import save_presets
from .dialogs import ExportDialog, ImportDialog
from .documents import read_pipelines, unique_name, write_pipelines

if TYPE_CHECKING:
    from ..editor import PipelineEditor


def import_pipelines(editor: PipelineEditor) -> None:
    """
    Read multiple files, review included names and save the entire selection atomically.
    """
    if editor.locked:
        return
    directory = load_dialog_directory("pipeline_import")
    paths, _ = QFileDialog.getOpenFileNames(
        editor, "Import pipelines", str(directory or ""), "Pipelines (*.pipeline.json *.json)"
    )
    if not paths:
        return
    incoming = []
    rejected = []
    for selected in paths:
        try:
            incoming.extend(read_pipelines(Path(selected)))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rejected.append(f"{Path(selected).name}: {exc}")
    if rejected:
        QMessageBox.warning(editor, "Some pipelines could not be imported", "\n".join(rejected))
    if not incoming or editor.locked:
        return
    dialog = ImportDialog(editor, incoming, set(editor.presets))
    if dialog.exec() != QDialog.DialogCode.Accepted or editor.locked:
        return
    try:
        selected = dialog.selected_pipelines()
        if not selected:
            return
        candidate = {**editor.presets, **selected}
        save_presets(candidate)
    except (OSError, ValueError, TypeError) as exc:
        QMessageBox.warning(editor, "Import failed", str(exc))
        return
    editor.presets = candidate
    remember_dialog_directory("pipeline_import", paths[0])
    if dialog.load_first.isChecked():
        name, document = next(iter(selected.items()))
        editor.document = deepcopy(document)
        editor.current_preset_name = name
        editor.recognize_preset = True
        editor.rebuild()
        editor.publish()
    editor.refresh_presets()


def export_pipelines(editor: PipelineEditor) -> None:
    """
    Choose current or saved pipelines and export them together to one JSON file.
    """
    if not editor.locked:
        editor.commit_pending_inputs()
    current_name = editor.current_preset_name or unique_name(
        "Current pipeline", set(editor.presets)
    )
    documents = deepcopy(editor.presets)
    documents[current_name] = deepcopy(editor.document)
    names = [current_name] + sorted(
        (name for name in documents if name != current_name), key=str.casefold
    )
    dialog = ExportDialog(editor, names, current_name)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    selected = {name: documents[name] for name in dialog.selected_names()}
    if not selected:
        return
    directory = load_dialog_directory("pipeline_export")
    path, _ = QFileDialog.getSaveFileName(
        editor,
        "Export pipelines",
        str((directory or Path.cwd()) / "camera.pipeline.json"),
        "Pipelines (*.pipeline.json)",
    )
    if not path:
        return
    if not path.endswith(".pipeline.json"):
        path += ".pipeline.json"
    destination = Path(path)
    single = editor.current_preset_name is None and list(selected) == [current_name]
    try:
        write_pipelines(destination, selected, single=single)
        remember_dialog_directory("pipeline_export", destination)
    except (OSError, ValueError, TypeError) as exc:
        QMessageBox.warning(editor, "Export failed", str(exc))
