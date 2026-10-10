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


class CustomFields(QWidget):
    """
    Bind declarative controls to the custom node's saved configuration.
    """

    def __init__(self, controls: list[dict[str, Any]], values: dict[str, Any],
                 changed: Callable[[str, Any], None]) -> None:
        """
        Initialize widgets before connecting signals, avoiding unsolicited updates.
        """
        super().__init__()
        self.fields: dict[str, QWidget] = {}
        layout = QFormLayout(self)
        for control in controls:
            key, kind = control["key"], control["type"]
            if kind in ("color", "model"):
                factory = ColorPicker if kind == "color" else ModelPicker
                widget = factory(values[key], lambda value, name=key: changed(name, value))
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

    def refresh(self, values: dict[str, Any]) -> None:
        """
        Reflect JSON edits without emitting processing updates from widget signals.
        """
        for key, widget in self.fields.items():
            blocker = QSignalBlocker(widget)
            if isinstance(widget, (ColorPicker, ModelPicker)):
                widget.refresh(values[key])
            elif isinstance(widget, QCheckBox):
                widget.setChecked(values[key])
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(values[key])
            elif isinstance(widget, QLineEdit):
                widget.setText(values[key])
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(values[key])
            blocker.unblock()
