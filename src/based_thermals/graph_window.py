"""Graph configuration in a separate Qt process."""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .capture_window import run_window
from .graph_settings import GRAPH_DEFAULTS
from .view_window import NoWheelComboBox, NoWheelSpinBox


class GraphWindow(QWidget):
    def __init__(self, send: Callable[[dict], None]) -> None:
        """
        Build chart history and sample-based display controls.
        """
        super().__init__()
        self.send = send
        self.setWindowTitle("Graph configuration")
        self.resize(420, 380)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.range_mode = NoWheelComboBox()
        self.range_mode.addItem("Entire session (automatic range)", "session")
        self.range_mode.addItem("Recent time window", "fixed")
        self.minutes = NoWheelSpinBox()
        self.minutes.setRange(0.1, 10080)
        self.minutes.setDecimals(1)
        self.minutes.setKeyboardTracking(False)
        self.minutes.setSuffix(" min")
        self.compression = QCheckBox("Compress older history")
        self.points = NoWheelSpinBox()
        self.points.setRange(256, 65536)
        self.points.setDecimals(0)
        self.points.setKeyboardTracking(False)
        self.points.setSingleStep(256)
        self.averaging = NoWheelSpinBox()
        self.averaging.setRange(1, 256)
        self.averaging.setDecimals(0)
        self.averaging.setKeyboardTracking(False)
        self.averaging.setSpecialValueText("Off")
        self.interpolation = NoWheelSpinBox()
        self.interpolation.setRange(0, 32)
        self.interpolation.setDecimals(0)
        self.interpolation.setKeyboardTracking(False)
        self.interpolation.setSpecialValueText("Off")
        for control in (self.averaging, self.interpolation):
            control.setMinimumWidth(max(100, control.fontMetrics().horizontalAdvance("Off") + 64))
        for control in (self.minutes, self.points, self.averaging, self.interpolation):
            control.editingFinished.connect(self._edit_finished)
        form.addRow("Average over samples", self.averaging)
        form.addRow("Extra points per sample interval", self.interpolation)
        form.addRow("Visible range", self.range_mode)
        form.addRow("Time window", self.minutes)
        form.addRow(self.compression)
        form.addRow("Points per chart", self.points)
        layout.addLayout(form)
        description = QLabel(
            "Compression preserves older peaks and gaps while keeping the newest half of the "
            "history budget at full detail. With compression off, older points are discarded "
            "when the budget is full. Averaging uses trailing retained samples; interpolation "
            "smooths between them. Both preserve gaps and affect only chart display. "
            "CSV logging always saves every valid sample."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)
        self.update_state({"settings": GRAPH_DEFAULTS})
        self.range_mode.currentIndexChanged.connect(self._changed)
        self.minutes.valueChanged.connect(self._changed)
        self.compression.toggled.connect(self._changed)
        self.points.valueChanged.connect(self._changed)
        self.averaging.valueChanged.connect(self._changed)
        self.interpolation.valueChanged.connect(self._changed)

    def _edit_finished(self) -> None:
        """
        Allow parent refreshes after a numeric edit has been committed.
        """
        control = self.sender()
        if isinstance(control, NoWheelSpinBox):
            editor = control.lineEdit()
            if editor is not None:
                editor.setModified(False)

    def _changed(self, *_args: object) -> None:
        """
        Publish committed chart preferences through the JSON boundary.
        """
        self.minutes.setEnabled(self.range_mode.currentData() == "fixed")
        self.send(
            {
                "action": "settings",
                "settings": {
                    "range_mode": self.range_mode.currentData(),
                    "range_seconds": self.minutes.value() * 60,
                    "compression": self.compression.isChecked(),
                    "history_points": int(self.points.value()),
                    "averaging_samples": int(self.averaging.value()),
                    "interpolation_samples": int(self.interpolation.value()),
                },
            }
        )

    def update_state(self, state: dict[str, Any]) -> None:
        """
        Restore preferences and lock all settings during temperature logging.
        """
        settings = {**GRAPH_DEFAULTS, **state.get("settings", {})}
        for control, value in (
            (self.range_mode, settings["range_mode"]),
            (self.minutes, settings["range_seconds"] / 60),
            (self.compression, settings["compression"]),
            (self.points, settings["history_points"]),
            (self.averaging, settings["averaging_samples"]),
            (self.interpolation, settings["interpolation_samples"]),
        ):
            if isinstance(control, NoWheelSpinBox) and not state.get("locked", False):
                editor = control.lineEdit()
                if editor is not None and editor.isModified():
                    control.setEnabled(True)
                    continue
            with QSignalBlocker(control):
                if control is self.range_mode:
                    control.setCurrentIndex(control.findData(value))
                elif control is self.compression:
                    control.setChecked(value)
                else:
                    control.setValue(value)
            control.setEnabled(not state.get("locked", False))
        self.minutes.setEnabled(
            not state.get("locked", False) and settings["range_mode"] == "fixed"
        )
        self.status.setText(
            "Stop logging to change graph settings."
            if state.get("locked")
            else state.get("status", "")
        )
        if state.get("raise_window"):
            self.showNormal()
            self.raise_()
            self.activateWindow()


def main() -> int:
    """
    Run chart configuration in the isolated Qt popup process.
    """
    result = run_window(GraphWindow, "Graph configuration")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
