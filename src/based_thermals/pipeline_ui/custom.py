"""
Folder loading and JSON configuration controls for custom software nodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..custom_nodes.bundle import (
    MAX_CONFIG_BYTES,
    load_folder,
    package_files,
    read_json_object,
    validate_custom,
)
from ..custom_nodes.configuration import configuration, read_controls
from ..custom_nodes.metadata import embedded_metadata
from ..custom_nodes.models import model_source
from .custom_fields import CustomFields

if TYPE_CHECKING:
    from .editor import PipelineEditor


class CustomControls(QWidget):
    """
    Edit an embedded Python package without importing it in the UI process.
    """

    def __init__(self, editor: PipelineEditor, item: dict[str, Any]) -> None:
        """
        Offer package-folder selection and a JSON file/editor.
        """
        super().__init__()
        self.editor, self.item = editor, item
        layout = QVBoxLayout(self)
        self.layout_box = layout
        self.fields: CustomFields | None = None
        self.control_source: str | None = None
        self.module_label = QLabel()
        self.module_label.setWordWrap(True)
        layout.addWidget(self.module_label)
        explanation = QLabel(
            "Package __init__.py must expose process(image, config). "
            "Python and JSON files are included in saved pipelines. "
            "Custom Python runs with application privileges; load packages you trust."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.folder_button = QPushButton("Open module folder…")
        self.json_button = QPushButton("Import JSON configuration…")
        self.model_button = QPushButton("Import model source…")
        self.edit_button = QPushButton("Edit JSON configuration…")
        self.folder_button.clicked.connect(self.open_folder)
        self.json_button.clicked.connect(self.import_json)
        self.model_button.clicked.connect(self.import_model)
        self.edit_button.clicked.connect(self.edit_json)
        for button in (self.folder_button, self.json_button, self.model_button, self.edit_button):
            layout.addWidget(button)
        self.refresh()

    def refresh(self) -> None:
        """
        Reflect accepted package names and the current editor lock.
        """
        self.module_label.setText(self.item["params"]["name"] or "Choose a custom module folder")
        params = self.item["params"]
        source = package_files(params["package"]).get("__init__.py", "")
        metadata = embedded_metadata(source)
        self.json_button.setVisible("CONFIG_JSON" not in metadata)
        self.model_button.setVisible("MODEL_SOURCE" not in metadata)
        values = read_json_object(params["config"])
        control_source = params.get("controls", "[]")
        controls = read_controls(control_source, values)
        if self.control_source != control_source:
            if self.fields is not None:
                self.layout_box.removeWidget(self.fields)
                self.fields.deleteLater()
            fields = CustomFields(
                controls, values, self.change_value, getattr(self.editor, "last_state", {})
            )
            self.fields = fields
            self.layout_box.addWidget(fields)
            self.control_source = control_source
        elif self.fields is not None:
            self.fields.refresh(values, getattr(self.editor, "last_state", {}))
        self.setEnabled(not self.editor.locked)

    def change_value(self, key: str, value: Any) -> None:
        """
        Publish one configuration edit from a generated control unless locked.
        """
        if self.editor.locked:
            self.refresh()
            return
        values = read_json_object(self.item["params"]["config"])
        values[key] = value
        try:
            self.set_config(json.dumps(values, allow_nan=False))
        except ValueError as exc:
            self.refresh()
            QMessageBox.warning(self, "Custom configuration", str(exc))

    def commit(self, params: dict[str, str]) -> None:
        """
        Publish one complete update after all file/config validation succeeds.
        """
        if self.editor.locked:
            return
        candidate = {**self.item["params"], **params}
        validate_custom(candidate)
        self.item["params"].update(candidate)
        self.refresh()
        self.editor.publish()
        self.editor.update_titles()

    def open_folder(self) -> None:
        """
        Snapshot a selected folder, including its optional config.json defaults.
        """
        if self.editor.locked:
            return
        path = QFileDialog.getExistingDirectory(self, "Open custom module folder")
        if not path:
            return
        try:
            params = load_folder(Path(path))
            self.commit(params)
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, "Custom module", str(exc))

    def set_config(self, text: str) -> None:
        """
        Validate and normalize a JSON object before publishing it.
        """
        config = read_json_object(text)
        read_controls(self.item["params"].get("controls", "[]"), config)
        self.commit({"config": json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False)})

    def import_json(self) -> None:
        """
        Replace configuration from a selected JSON file.
        """
        if self.editor.locked:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import custom configuration", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            file = Path(path)
            if file.stat().st_size > MAX_CONFIG_BYTES:
                raise ValueError("Custom configuration exceeds 64 KiB")
            values, controls = configuration(file.read_text(encoding="utf-8"))
            self.commit({"config": json.dumps(values), "controls": json.dumps(controls)})
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, "Custom configuration", str(exc))

    def import_model(self) -> None:
        """
        Load a pinned model descriptor when the package does not embed one.
        """
        if self.editor.locked:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import model source", "", "JSON (*.json)")
        if not path:
            return
        try:
            file = Path(path)
            if file.stat().st_size > MAX_CONFIG_BYTES:
                raise ValueError("Custom model source exceeds 64 KiB")
            source = read_json_object('{"model":' + file.read_text(encoding="utf-8") + '}')["model"]
            model_source(source)
            values = read_json_object(self.item["params"]["config"])
            values["model"] = source
            self.set_config(json.dumps(values))
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, "Custom model source", str(exc))

    def edit_json(self) -> None:
        """
        Edit configuration with a multiline dialog and retain invalid edits for correction.
        """
        if self.editor.locked:
            return
        text = self.item["params"]["config"]
        while True:
            text, accepted = QInputDialog.getMultiLineText(
                self, "Custom configuration", "JSON object", text
            )
            if not accepted:
                return
            try:
                self.set_config(text)
                return
            except ValueError as exc:
                QMessageBox.warning(self, "Custom configuration", str(exc))
