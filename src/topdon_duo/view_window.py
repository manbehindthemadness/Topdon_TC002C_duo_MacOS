"""Camera and display controls in a separate Qt process."""

import math
import sys
from collections.abc import Callable
from typing import Any, cast

from PySide6.QtCore import QPoint, QSettings, QSignalBlocker, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .capture_window import run_window
from .hardware_controls import HARDWARE_CONTROLS, camera_operation_title
from .view_controls import (
    ControlRow,
    NoWheelCheckBox,
    NoWheelComboBox,
    NoWheelSlider,
    NoWheelSpinBox,
)
from .view_settings import DISTANCE_METERS_PER_UNIT, TEMPERATURE_UNITS, VIEW_DEFAULTS

__all__ = [
    "ControlRow",
    "DistanceCalibrationControls",
    "EmissivityCalibrationControls",
    "NoWheelCheckBox",
    "NoWheelComboBox",
    "NoWheelSlider",
    "NoWheelSpinBox",
    "ReflectedCalibrationControls",
    "ViewWindow",
    "main",
]


def _finite_number(value: object) -> float | None:
    """
    Read optional numeric calibration telemetry without formatting malformed values.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    result = number if math.isfinite(number) else None
    return result


class DistanceCalibrationControls(QWidget):
    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        """
        Build calibration inputs and actions using the viewer message callback.
        """
        super().__init__()
        self._send = send
        self._unit = "C"
        self._reference = None
        self._state: dict[str, Any] = {}
        self._locked = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Distance calibration")
        heading.setStyleSheet("font-size: 16px; font-weight: bold")
        layout.addWidget(heading)
        instructions = QLabel(
            "Use a standard 3 × 3 inch Post-it (76 × 76 mm; measure yours and edit the size if needed). "
            "Hold the cooler note flat in front of your warm palm, with skin "
            "visible around its edges. Face it toward the camera and hold steady "
            "for automatic detection. Enter a tape-measured distance from the front lens, "
            "Detect reference square, then Save reference. Use Measure square at a new distance "
            "and Apply distance to camera to update the camera. Saved references load at startup; "
            "applied distance settings are also restored. Check estimates against a tape measure."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.side = NoWheelSpinBox()
        self.distance = NoWheelSpinBox()
        self._physical_values = {self.side: (0.076, 7.6), self.distance: (0, 0)}
        for title, control, maximum, step in (
            ("Square side length", self.side, 10, 0.001),
            ("Measured reference distance", self.distance, 99, 0.01),
        ):
            line = QHBoxLayout()
            line.addWidget(QLabel(title), 1)
            control.setAccessibleName(title)
            control.setRange(0, maximum * 100)
            control.setDecimals(4)
            control.setSingleStep(step * 100)
            control.setSuffix(" cm")
            control.setKeyboardTracking(False)
            control.valueChanged.connect(self._update_buttons)
            line.addWidget(control)
            layout.addLayout(line)
        self.select = QPushButton("Detect reference square")
        self.save = QPushButton("Save reference")
        self.measure = QPushButton("Measure square")
        self.apply = QPushButton("Apply distance to camera")
        self.clear = QPushButton("Clear calibration")
        self.select.clicked.connect(
            lambda: self._send(
                {
                    "action": "distance_calibration",
                    "operation": "cancel" if self._state.get("selecting") else "select",
                }
            )
        )
        self.save.clicked.connect(self._save)
        for button, operation in (
            (self.measure, "measure"),
            (self.apply, "apply"),
            (self.clear, "clear"),
        ):
            button.clicked.connect(
                lambda _checked=False, selected_operation=operation: self._send(
                    {"action": "distance_calibration", "operation": selected_operation}
                )
            )
        for controls in ((self.select, self.save), (self.measure, self.apply, self.clear)):
            line = QHBoxLayout()
            for button in controls:
                line.addWidget(button)
            layout.addLayout(line)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.side.setValue(7.6)

    def _meters(self, control):
        meters, shown = self._physical_values[control]
        if control.value() != shown:
            meters = control.value() * DISTANCE_METERS_PER_UNIT[self._unit]
            self._physical_values[control] = (meters, control.value())
        return meters

    def _save(self):
        if not self._locked:
            self.side.interpretText()
            self.distance.interpretText()
            self._send(
                {
                    "action": "distance_calibration",
                    "operation": "save",
                    "side_m": self._meters(self.side),
                    "distance_m": self._meters(self.distance),
                }
            )

    def _update_buttons(self, *_args: Any) -> None:
        """
        Apply calibration availability and lock state to each action.
        """
        selecting = bool(self._state.get("selecting"))
        self.select.setText("Cancel detection" if selecting else "Detect reference square")
        self.select.setEnabled(not self._locked)
        self.save.setEnabled(
            not self._locked
            and self._state.get("ready", False)
            and self.side.value() > 0
            and self.distance.value() > 0
        )
        self.measure.setEnabled(not self._locked and bool(self._reference) and not selecting)
        estimated = _finite_number(self._state.get("estimated_m"))
        self.apply.setEnabled(not self._locked and estimated is not None and 0.3 <= estimated <= 99)
        self.clear.setEnabled(not self._locked and bool(self._reference))

    def update_state(self, state: dict[str, Any], unit: str, locked: bool) -> None:
        """
        Apply calibration state and display units without sending user edits.
        """
        if unit != self._unit:
            self.side.interpretText()
            self.distance.interpretText()
        values = (self._meters(self.side), self._meters(self.distance))
        reference = state.get("reference")
        update_inputs = unit != self._unit or bool(reference and reference != self._reference)
        if isinstance(reference, dict) and reference != self._reference and reference:
            reference = cast(dict[str, Any], reference)
            values = (reference["side_m"], reference["distance_m"])
        self._reference, self._state, self._unit, self._locked = reference, state, unit, locked
        factor = float(1 / DISTANCE_METERS_PER_UNIT[unit])
        for control, value, maximum, step in (
            (self.side, values[0], 10, 0.001),
            (self.distance, values[1], 99, 0.01),
        ):
            if update_inputs:
                with QSignalBlocker(control):
                    control.setRange(0, maximum * factor)
                    control.setSingleStep(step * factor)
                    control.setSuffix(" in" if unit == "F" else " cm")
                    control.setValue(value * factor)
                    self._physical_values[control] = (value, control.value())
            control.setEnabled(not locked)
        message = state.get("status", "Select a reference square to begin.")
        estimated = _finite_number(state.get("estimated_m"))
        if estimated is not None:
            message = (
                f"Estimated distance: {estimated * factor:.2f} {'in' if unit == 'F' else 'cm'}. "
                + message
            )
            if not 0.3 <= estimated <= 99:
                message += " Outside the camera's distance range."
        self.status.setText(message)
        self._update_buttons()


class EmissivityCalibrationControls(QWidget):
    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        """
        Build calibration inputs and actions using the viewer message callback.
        """
        super().__init__()
        self._send = send
        self._unit = "C"
        self._locked = False
        self._reference = None
        self._selection_id = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Emissivity calibration")
        heading.setStyleSheet("font-size: 16px; font-weight: bold")
        layout.addWidget(heading)
        instructions = QLabel(
            "Select a surface point, then enter its independently measured temperature. "
            "Other markers are hidden; the reference point remains visible. Its current "
            "temperature is copied for you to correct. Keep the point and surface steady "
            "while fitting. Ambient, reflected temperature and distance must already "
            "suit the measurement. Applying emissivity affects the whole image."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.known = NoWheelSpinBox()
        self.known.setAccessibleName("Known surface temperature")
        self.known.setRange(-50, 550)
        self.known.setDecimals(2)
        self.known.setSingleStep(0.1)
        self.known.setSuffix(" °C")
        self.known.setValue(20)
        self.known.setKeyboardTracking(False)
        line = QHBoxLayout()
        line.addWidget(QLabel("Known surface temperature"), 1)
        line.addWidget(self.known)
        layout.addLayout(line)
        self.select = QPushButton("Select reference point")
        self.fit = QPushButton("Fit emissivity")
        self.apply = QPushButton("Apply emissivity to camera")
        self.cancel = QPushButton("Close calibration")
        for button, operation in (
            (self.select, "select"),
            (self.apply, "apply"),
            (self.cancel, "cancel"),
        ):
            button.clicked.connect(
                lambda _checked=False, selected_operation=operation: self._send(
                    {"action": "emissivity_calibration", "operation": selected_operation}
                )
            )
        self.fit.clicked.connect(self._fit)
        for controls in ((self.select, self.fit), (self.apply, self.cancel)):
            line = QHBoxLayout()
            for button in controls:
                line.addWidget(button)
            layout.addLayout(line)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.update_state({}, "C", False)

    def _celsius(self):
        return (self.known.value() - 32) / 1.8 if self._unit == "F" else self.known.value()

    def _fit(self):
        if not self._locked:
            self.known.interpretText()
            self._send(
                {
                    "action": "emissivity_calibration",
                    "operation": "fit",
                    "known_celsius": self._celsius(),
                }
            )

    def update_state(self, state: dict[str, Any], unit: str, locked: bool) -> None:
        """
        Apply calibration state and display units without sending user edits.
        """
        if unit != self._unit:
            self.known.interpretText()
            value = self._celsius()
            self._unit = unit
            with QSignalBlocker(self.known):
                self.known.setRange(-58 if unit == "F" else -50, 1022 if unit == "F" else 550)
                self.known.setSingleStep(0.18 if unit == "F" else 0.1)
                self.known.setSuffix(f" °{unit}")
                self.known.setValue(value * 1.8 + 32 if unit == "F" else value)
        reference = state.get("reference")
        if isinstance(reference, dict) and reference and reference != self._reference:
            reference = cast(dict[str, Any], reference)
            with QSignalBlocker(self.known):
                value = float(reference["known_celsius"])
                self.known.setValue(value * 1.8 + 32 if unit == "F" else value)
        self._reference = reference
        selection_id = state.get("selection_id")
        if state.get("selected_celsius") is not None and selection_id != self._selection_id:
            known = state.get("known_celsius")
            selected = known if known is not None else state["selected_celsius"]
            value = float(cast(float, selected))
            with QSignalBlocker(self.known):
                self.known.setValue(value * 1.8 + 32 if unit == "F" else value)
            self._selection_id = selection_id
        self._locked = locked
        running = bool(state.get("running"))
        self.known.setEnabled(not locked)
        self.select.setEnabled(not locked and not running)
        self.fit.setEnabled(not locked and state.get("point") is not None and not running)
        self.apply.setEnabled(
            not locked and (state.get("result") is not None or bool(reference)) and not running
        )
        self.apply.setText(
            "Apply saved emissivity"
            if reference and state.get("result") is None
            else "Apply emissivity to camera"
        )
        self.cancel.setEnabled(bool(state.get("active")) and (not locked or running))
        message = state.get("status", "Select a point to begin.")
        measured = _finite_number(state.get("measured_celsius"))
        if measured is not None and state.get("point") is not None:
            display = measured * 1.8 + 32 if unit == "F" else measured
            message += f" Selected point: {display:.2f} °{unit}."
        if state.get("result") is not None:
            message += f" Fitted emissivity: {state['result']:.2f}."
        elif isinstance(reference, dict) and reference:
            message += f" Saved emissivity: {reference['emissivity']:.2f}."
        self.status.setText(message)


class ReflectedCalibrationControls(QWidget):
    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        """
        Build calibration inputs and actions using the viewer message callback.
        """
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Reflected-temperature calibration")
        heading.setStyleSheet("font-size: 16px; font-weight: bold")
        layout.addWidget(heading)
        instructions = QLabel(
            "Cover the center sampling circle with a shiny, bare-metal spoon or foil reflector. "
            "Keep it steady and avoid reflecting yourself or the camera. Spoon curvature makes the reading "
            "orientation-dependent; compare with crumpled then flattened foil before relying on calibration. "
            "Choose Show reflector target, cover the entire circle, then Measure stable temperature. "
            "The stable result saves automatically and clears the circle. Apply saved reflected temperature "
            "updates the camera. Saved references load at startup; applied settings are also restored. "
            "Measure temporarily uses emissivity 1 and optical transmission 100%, then restores both settings."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.select = QPushButton("Show reflector target")
        self.measure = QPushButton("Measure stable temperature")
        self.apply = QPushButton("Apply saved reflected temperature")
        self.cancel = QPushButton("Close calibration")
        for button, operation in (
            (self.select, "select"),
            (self.measure, "measure"),
            (self.apply, "apply"),
            (self.cancel, "cancel"),
        ):
            button.clicked.connect(
                lambda _checked=False, selected_operation=operation: send(
                    {"action": "reflected_calibration", "operation": selected_operation}
                )
            )
        for buttons in ((self.select, self.measure), (self.apply, self.cancel)):
            row = QHBoxLayout()
            for button in buttons:
                row.addWidget(button)
            layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.update_state({}, "C", False)

    def update_state(self, state: dict[str, Any], unit: str, locked: bool) -> None:
        """
        Apply calibration state and display units without sending user edits.
        """
        running = bool(state.get("running"))
        self.select.setEnabled(not locked)
        self.measure.setEnabled(not locked and bool(state.get("active")))
        self.apply.setEnabled(not locked and bool(state.get("reference")))
        self.cancel.setEnabled(bool(state.get("active")) and (not locked or running))
        message = state.get("status", "Show the sampling target to begin.")
        reference = state.get("reference")
        if isinstance(reference, dict) and reference:
            value = float(reference["celsius"])
            if unit == "F":
                value = value * 1.8 + 32
            message += f" Saved: {value:.1f} °{unit}."
        self.status.setText(message)


class ViewWindow(QWidget):
    def __init__(self, send: Any) -> None:
        """
        Build pipeline and permanent calibration controls with the viewer message callback.
        """
        super().__init__()
        from .pipeline_editor import PipelineEditor
        self._send_to_viewer = send
        self._settings_locked = False
        self._anchor_top_right = None
        self.setWindowTitle("Camera")
        self.setMinimumWidth(650)
        self._settings = QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, "topdon-duo", "desktop"
        )
        layout = QVBoxLayout(self)
        heading = QLabel("Camera")
        heading.setStyleSheet("font-size: 20px; font-weight: bold")
        layout.addWidget(heading)
        self.operation_status = QLabel()
        self.operation_status.setWordWrap(True)
        self.operation_status.setStyleSheet("font-weight: bold; padding: 8px")
        self.operation_status.hide()
        layout.addWidget(self.operation_status)
        self.auto_calibrate = NoWheelCheckBox("Auto calibrate")
        self.auto_calibrate.toggled.connect(lambda value: self._send({"action": "auto_calibrate", "value": value}))
        layout.addWidget(self.auto_calibrate)
        self.cancel_tone = QPushButton("Cancel tone update")
        self.cancel_tone.clicked.connect(lambda: self._send({"action": "cancel_tone"}))
        self.cancel_tone.hide()
        layout.addWidget(self.cancel_tone)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        rows = QVBoxLayout(body)
        self.pipeline_editor = PipelineEditor(self._send, ControlRow)
        rows.addWidget(self.pipeline_editor)
        self.rows = {}
        self.hardware_rows = {}
        units = ControlRow("Measurement units", lambda value: self._send({"action": "setting", "name": "temperature_unit", "value": value}), options=tuple(TEMPERATURE_UNITS.items()))
        self.rows["temperature_unit"] = units
        self.controls = {"temperature_unit": units.input}
        rows.addWidget(units)
        for name in ("ambient", "distance", "emissivity", "reflected", "transmission", "humidity"):
            spec = HARDWARE_CONTROLS[name]
            row = ControlRow(spec.title, lambda value, field=name: self._send({"action": "hardware", "name": field, "value": value, "enabled": True}), minimum=spec.minimum, maximum=spec.maximum, step=spec.step, options=spec.options, unit=spec.unit, preserve_input=name == "ambient")
            self.hardware_rows[name] = row
            rows.addWidget(row)
        self.distance_calibration = DistanceCalibrationControls(self._send)
        self.emissivity_calibration = EmissivityCalibrationControls(self._send)
        self.reflected_calibration = ReflectedCalibrationControls(self._send)
        for tool in (self.distance_calibration, self.emissivity_calibration, self.reflected_calibration):
            rows.addWidget(tool)
        rows.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.restore_button = QPushButton("Restore camera settings")
        self.restore_button.clicked.connect(self._restore_hardware)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.restore_button)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.update_state(VIEW_DEFAULTS)
        size = self._settings.value("camera/window_size", QSize(700, 850))
        self.resize(size if isinstance(size, QSize) and size.isValid() else QSize(700, 850))
        self._position_top_right()
        application = QApplication.instance()
        if isinstance(application, QApplication):
            application.aboutToQuit.connect(self._save_window_size)

    def showEvent(self, event):
        super().showEvent(event)
        # Recompute after the window manager supplies title-bar/frame dimensions.
        if sys.platform != "darwin":
            QTimer.singleShot(0, self._position_top_right)

    def _position_top_right(self) -> None:
        """
        Position non-macOS windows within the selected screen bounds.
        """
        # Keep macOS's native placement; delayed Qt moves cause a visible jump.
        if sys.platform == "darwin":
            return
        point = self._anchor_top_right
        screen = QApplication.screenAt(point) if point is not None else self.screen()
        screen = screen or QApplication.primaryScreen()
        bounds = screen.availableGeometry()
        right = point.x() if point is not None else bounds.right()
        top = point.y() if point is not None else bounds.top()
        frame = self.frameGeometry()
        x = max(bounds.left(), min(right - frame.width() + 1, bounds.right() - frame.width() + 1))
        y = max(bounds.top(), min(top, bounds.bottom() - frame.height() + 1))
        self.move(x, y)

    def _save_window_size(self):
        self._settings.setValue("camera/window_size", self.size())
        self._settings.sync()

    def closeEvent(self, event):
        self._save_window_size()
        super().closeEvent(event)

    def _send(self, command):
        title = camera_operation_title(command)
        if title:
            self.operation_status.setText(f"Applying {title.lower()}… Image and graph updates may pause briefly.")
            self.operation_status.show()
            self.operation_status.repaint()
        self._send_to_viewer(command)

    def _restore_hardware(self):
        if not self._settings_locked:
            for row in self.hardware_rows.values():
                row.timer.stop()
            self._send({"action": "restore_hardware"})

    def update_state(self, state):
        anchor = state.get("anchor_top_right")
        if (
            sys.platform != "darwin"
            and isinstance(anchor, list)
            and len(anchor) == 2
            and all(type(value) is int for value in anchor)
        ):
            self._anchor_top_right = QPoint(*anchor)
            QTimer.singleShot(0, self._position_top_right)
        tone_busy = bool(state.get("tone_busy", False))
        operation_busy = bool(state.get("camera_operation_busy", False))
        operation = state.get("camera_operation", "")
        if tone_busy:
            operation = f"Updating camera gamma… {state.get('tone_progress', 0)}%"
        self.operation_status.setText(operation)
        self.operation_status.setVisible(bool(operation))
        locked = bool(state.get("settings_locked", False)) or tone_busy or operation_busy
        if locked and not self._settings_locked:
            focused = QApplication.focusWidget()
            if focused is not None and self.isAncestorOf(focused):
                self.setFocus(Qt.FocusReason.OtherFocusReason)
        self._settings_locked = locked
        self.auto_calibrate.setEnabled(not locked)
        with QSignalBlocker(self.auto_calibrate):
            self.auto_calibrate.setChecked(state.get("auto_calibrate", False))
        self.cancel_tone.setVisible(tone_busy)
        self.cancel_tone.setEnabled(tone_busy and not state.get("settings_locked", False))
        self.pipeline_editor.update_state(state, locked)
        self.restore_button.setEnabled(not locked)
        unit = state.get("temperature_unit", "C")
        self.rows["temperature_unit"].update_state(unit, not locked)
        for name, row in self.hardware_rows.items():
            row.set_display_unit(unit)
            setting = state.get("hardware", {}).get(name, {})
            available = setting.get("available", False) and not locked
            row.update_state(setting.get("value", HARDWARE_CONTROLS[name].minimum), available)
        for tool, key in ((self.distance_calibration, "distance_calibration"), (self.emissivity_calibration, "emissivity_calibration"), (self.reflected_calibration, "reflected_calibration")):
            tool.update_state(state.get(key, {}), unit, locked)
        self.status.setText(state.get("status", ""))
        if state.get("raise_window"):
            self.showNormal()
            self.raise_()
            self.activateWindow()


def main() -> int:
    return run_window(ViewWindow, "Camera")


if __name__ == "__main__":
    raise SystemExit(main())
