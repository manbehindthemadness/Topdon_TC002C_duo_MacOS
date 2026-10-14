"""
Per-measurement display styling controls in the isolated Qt editor.
"""

from collections.abc import Callable

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
)


class AppearanceControls(QWidget):
    """
    Edit RGB line color, optional same-color fill and shared shape opacity.
    """

    def __init__(self, changed: Callable[[dict], None], dirty: Callable[[], None]) -> None:
        """
        Build controls without sending hardware or parent-process commands.
        """
        super().__init__()
        self.changed = changed
        self.dirty = dirty
        self.color = "#78c5ff"
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        colors = QHBoxLayout()
        self.line_color = QPushButton()
        self.line_color.setAccessibleName("Measurement line color")
        self.line_color.clicked.connect(self._choose_color)
        self.dynamic = QCheckBox("Dynamic")
        self.dynamic.setToolTip("Use contrasting outlines that follow the thermal image")
        self.dynamic.toggled.connect(self._changed)
        colors.addWidget(self.line_color)
        colors.addWidget(self.dynamic)
        self.fill = QCheckBox("Fill with line color")
        self.fill.setToolTip(
            "Fill closed areas and positive-radius spots; single-pixel spots stay crosses"
        )
        self.fill.toggled.connect(self._changed)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 100)
        self.opacity.setSuffix(" %")
        self.opacity.setDecimals(0)
        self.opacity.setKeyboardTracking(False)
        editor = self.opacity.lineEdit()
        assert editor is not None
        editor.textEdited.connect(lambda _text: self.dirty())
        self.opacity.setToolTip("Opacity of the outline and fill; labels remain readable")
        self.opacity.valueChanged.connect(self._changed)
        form.addRow("Line Color", colors)
        form.addRow("Fill", self.fill)
        form.addRow("Opacity", self.opacity)
        self.set_region({})

    def values(self, *, commit: bool = True) -> dict:
        """
        Return JSON-safe appearance fields with opacity normalized to 0–1.
        """
        if commit:
            self.opacity.interpretText()
        return {
            "line_color": "dynamic" if self.dynamic.isChecked() else self.color,
            "fill": self.fill.isChecked(),
            "opacity": self.opacity.value() / 100,
        }

    def set_region(self, region: dict) -> None:
        """
        Restore selected settings without marking the configuration dirty.
        """
        color = region.get("line_color", "dynamic")
        if color != "dynamic":
            self.color = color
        with QSignalBlocker(self.dynamic), QSignalBlocker(self.fill), QSignalBlocker(self.opacity):
            self.dynamic.setChecked(color == "dynamic")
            self.fill.setChecked(region.get("fill", False))
            self.opacity.setValue(region.get("opacity", 1.0) * 100)
        self._refresh_color()

    def _refresh_color(self) -> None:
        """
        Show the saved RGB choice while Dynamic controls the effective color.
        """
        self.line_color.setText(self.color.upper())
        self.line_color.setEnabled(not self.dynamic.isChecked())

    def _changed(self, *_args: object) -> None:
        """
        Update the local preview through the editor's normal dirty-state boundary.
        """
        self._refresh_color()
        self.changed(self.values(commit=False))

    def _choose_color(self) -> None:
        """
        Choose an opaque RGB color; opacity has its own explicit control.
        """
        color = QColorDialog.getColor(QColor(self.color), self, "Measurement line color")
        if color.isValid():
            self.color = color.name()
            self._changed()
