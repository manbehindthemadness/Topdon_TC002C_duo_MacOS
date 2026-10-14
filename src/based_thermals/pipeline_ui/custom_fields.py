"""
Render validated custom-node controls as ordinary Qt configuration widgets.
"""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from .custom_choices import ColorPicker, ModelPicker
from .custom_devices import DevicePicker


class CustomFields(QWidget):
    """
    Bind declarative controls to the custom node's saved configuration.
    """

    def __init__(self, controls: list[dict[str, Any]], values: dict[str, Any],
                 changed: Callable[[str, Any], None], state: dict[str, Any] | None = None) -> None:
        """
        Initialize widgets before connecting signals, avoiding unsolicited updates.
        """
        super().__init__()
        self.changed = changed
        self.fields: dict[str, QWidget] = {}
        self.text_values: dict[str, str] = {}
        layout = QFormLayout(self)
        for control in controls:
            key, kind = control["key"], control["type"]
            if kind == "device":
                widget = DevicePicker(values[key], lambda value, name=key: changed(name, value), state)
            elif kind == "color":
                widget = ColorPicker(values[key], lambda value, name=key: changed(name, value))
            elif kind == "model":
                widget = ModelPicker(values[key], lambda value, name=key: changed(name, value),
                                     control.get("models"))
            elif kind == "boolean":
                widget = QCheckBox()
                widget.setChecked(values[key])
                widget.toggled.connect(lambda value, name=key: changed(name, value))
            elif kind == "choice":
                widget = QComboBox()
                widget.addItems(control["options"])
                widget.setCurrentText(values[key])
                widget.currentTextChanged.connect(lambda value, name=key: changed(name, value))
            elif kind == "text":
                widget = QLineEdit(values[key])
                self.text_values[key] = values[key]
                widget.editingFinished.connect(
                    lambda field=widget, name=key: changed(name, field.text())
                )
            else:
                widget = QSpinBox() if kind == "integer" else QDoubleSpinBox()
                if isinstance(widget, QDoubleSpinBox):
                    widget.setDecimals(9)
                widget.setRange(control["min"], control["max"])
                widget.setSingleStep(control["step"])
                widget.setValue(values[key])
                # noinspection PyUnresolvedReferences
                widget.valueChanged.connect(lambda value, name=key: changed(name, value))
            self.fields[key] = widget
            layout.addRow(control["label"], widget)

    def commit_pending_inputs(self) -> None:
        """
        Publish pending text and numeric input before saving or exporting a pipeline.
        """
        for key, widget in list(self.fields.items()):
            if isinstance(widget, QLineEdit) and widget.text() != self.text_values[key]:
                self.changed(key, widget.text())
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.interpretText()

    def refresh(
        self, values: dict[str, Any], state: dict[str, Any] | None = None,
        *, preserve_pending: bool = True,
    ) -> None:
        """
        Reflect accepted values while retaining drafts through unchanged runtime echoes.
        """
        for key, widget in self.fields.items():
            blocker = QSignalBlocker(widget)
            if isinstance(widget, DevicePicker):
                widget.refresh(values[key], state)
            elif isinstance(widget, (ColorPicker, ModelPicker)):
                widget.refresh(values[key])
            elif isinstance(widget, QCheckBox):
                widget.setChecked(values[key])
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(values[key])
            elif isinstance(widget, QLineEdit):
                if (not preserve_pending or not widget.isModified()
                        or values[key] != self.text_values[key]):
                    if widget.text() != values[key]:
                        widget.setText(values[key])
                    widget.setModified(False)
                self.text_values[key] = values[key]
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(values[key])
            blocker.unblock()
