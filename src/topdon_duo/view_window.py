"""Camera and display controls in a separate Qt process."""

from PySide6.QtCore import QSettings, QSignalBlocker, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .capture_window import run_window
from .hardware_controls import HARDWARE_CONTROLS
from .view_settings import (
    COLOR_PALETTES,
    DISTANCE_METERS_PER_UNIT,
    ENHANCEMENT_INPUTS,
    IMAGE_FILTERS,
    IMAGE_SOURCES,
    PALETTE_SOURCES,
    TEMPERATURE_UNITS,
    UPSCALING_MODES,
    VIEW_DEFAULTS,
)


class NoWheelSlider(QSlider):
    """Let the surrounding scroll area handle mouse-wheel input."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    """Keep choices unchanged while scrolling the surrounding menu."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelSpinBox(QDoubleSpinBox):
    """Keep numeric values unchanged while scrolling the surrounding menu."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelCheckBox(QCheckBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class ControlRow(QWidget):
    """An editable value with a slider for range controls."""

    def __init__(
        self,
        title,
        changed,
        *,
        minimum=0,
        maximum=1,
        step=1,
        options=(),
        unit="",
        preserve_input=False,
    ):
        super().__init__()
        self.preserve_input = preserve_input
        self.changed = changed
        self.options = tuple(options)
        self.is_switch = tuple(text for _, text in self.options) == ("Off", "On")
        self.step = step
        self.minimum = minimum
        self.maximum = maximum
        self.is_temperature = unit == "°C"
        self.is_distance = unit == "m"
        self.temperature_unit = "C"
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._emit)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600")
        layout.addWidget(label)
        line = QHBoxLayout()
        if self.is_switch:
            self.input = NoWheelCheckBox("Enabled")
            self.input.toggled.connect(self._input_changed)
        elif self.options:
            self.input = NoWheelComboBox()
            for value, text in self.options:
                self.input.addItem(text, value)
            self.input.currentIndexChanged.connect(self._input_changed)
        else:
            self.input = NoWheelSpinBox()
            self.input.setRange(self._display_value(minimum), self._display_value(maximum))
            self.input.setDecimals(
                0 if self.is_distance else (2 if step == 0.01 else (1 if step == 0.1 else 0))
            )
            self.input.setSingleStep(step * 100 if self.is_distance else step)
            self.input.setKeyboardTracking(False)
            self.input.setSuffix(" cm" if self.is_distance else f" {unit}" if unit else "")
            self.input.valueChanged.connect(self._input_changed)
        self.input.setMinimumWidth(160)
        self.input.setAccessibleName(title)
        line.addWidget(self.input)
        self.slider = None
        if not self.options:
            self.slider = NoWheelSlider(Qt.Horizontal)
            self.slider.setAccessibleName(f"{title} slider")
            self.slider.setRange(0, round((maximum - minimum) / step))
            self.slider.valueChanged.connect(self._slider_changed)
            self.slider.sliderReleased.connect(self._emit)
            line.addWidget(self.slider, 1)
        else:
            line.addStretch()
        layout.addLayout(line)

    def value(self):
        if self.is_switch:
            return self.options[int(self.input.isChecked())][0]
        if self.options:
            return self.input.currentData()
        value = self.input.value()
        if self.is_temperature or self.is_distance:
            if self.is_distance:
                value *= DISTANCE_METERS_PER_UNIT[self.temperature_unit]
            elif self.temperature_unit == "F":
                value = (value - 32) / 1.8
            if self.preserve_input:
                return round(value, 10)
            # Preserve hardware precision in Celsius and meters, regardless of display units.
            value = self.minimum + round((value - self.minimum) / self.step) * self.step
        return round(value, 2)

    def _display_value(self, value):
        if self.is_distance:
            return value / DISTANCE_METERS_PER_UNIT[self.temperature_unit]
        if self.temperature_unit == "F" and self.is_temperature:
            return value * 1.8 + 32
        return value

    def set_display_unit(self, unit):
        if not (self.is_temperature or self.is_distance) or unit == self.temperature_unit:
            return
        value = self.value()
        self.temperature_unit = unit
        if self.is_temperature:
            decimals = 2 if unit == "F" else 1
            step = self.step * 1.8 if unit == "F" else self.step
            suffix = f" °{unit}"
        else:
            decimals = 2 if unit == "F" else 0
            step = self.step / DISTANCE_METERS_PER_UNIT[unit]
            suffix = " in" if unit == "F" else " cm"
        with QSignalBlocker(self.input):
            self.input.setDecimals(decimals)
            self.input.setRange(
                self._display_value(self.minimum), self._display_value(self.maximum)
            )
            self.input.setSingleStep(step)
            self.input.setSuffix(suffix)
        self._set_value(value)

    def _set_value(self, value):
        with QSignalBlocker(self.input):
            if self.is_switch:
                self.input.setChecked(value == self.options[1][0])
                position = int(self.input.isChecked())
            elif self.options:
                index = self.input.findData(value)
                self.input.setCurrentIndex(index)
                position = max(0, index)
            else:
                self.input.setValue(self._display_value(value))
                position = round((value - self.minimum) / self.step)
        if self.slider is not None:
            with QSignalBlocker(self.slider):
                self.slider.setValue(position)

    def _set_enabled(self, enabled):
        if not enabled:
            self.timer.stop()
        self.input.setEnabled(enabled)
        if self.slider is not None:
            self.slider.setEnabled(enabled)

    def _input_changed(self, *_args):
        self._set_value(self.value())
        if self.input.isEnabled():
            self.timer.start()

    def _slider_changed(self, position):
        value = self.minimum + position * self.step
        self._set_value(value)
        if self.input.isEnabled() and not self.slider.isSliderDown():
            self.timer.start()

    def _emit(self):
        self.timer.stop()
        if self.input.isEnabled():
            self.changed(self.value())

    def update_state(self, value, available=True):
        self._set_enabled(available)
        if self.timer.isActive() or (self.slider is not None and self.slider.isSliderDown()):
            return
        if value != self.value():
            self._set_value(value)


class DistanceCalibrationControls(QWidget):
    def __init__(self, send):
        super().__init__()
        self._send = send
        self._unit = "C"
        self._reference = None
        self._state = {}
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
                lambda _checked=False, operation=operation: self._send(
                    {"action": "distance_calibration", "operation": operation}
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

    def _update_buttons(self, *_args):
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
        estimated = self._state.get("estimated_m")
        self.apply.setEnabled(not self._locked and estimated is not None and 0.3 <= estimated <= 99)
        self.clear.setEnabled(not self._locked and bool(self._reference))

    def update_state(self, state, unit, locked):
        if unit != self._unit:
            self.side.interpretText()
            self.distance.interpretText()
        values = (self._meters(self.side), self._meters(self.distance))
        reference = state.get("reference")
        update_inputs = unit != self._unit or bool(reference and reference != self._reference)
        if reference != self._reference and reference:
            values = (reference["side_m"], reference["distance_m"])
        self._reference, self._state, self._unit, self._locked = reference, state, unit, locked
        factor = 1 / DISTANCE_METERS_PER_UNIT[unit]
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
        estimated = state.get("estimated_m")
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
    def __init__(self, send):
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
                lambda _checked=False, operation=operation: self._send(
                    {"action": "emissivity_calibration", "operation": operation}
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

    def update_state(self, state, unit, locked):
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
        if reference and reference != self._reference:
            with QSignalBlocker(self.known):
                value = reference["known_celsius"]
                self.known.setValue(value * 1.8 + 32 if unit == "F" else value)
        self._reference = reference
        selection_id = state.get("selection_id")
        if state.get("selected_celsius") is not None and selection_id != self._selection_id:
            value = state.get("known_celsius")
            if value is None:
                value = state["selected_celsius"]
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
        measured = state.get("measured_celsius")
        if measured is not None and state.get("point") is not None:
            display = measured * 1.8 + 32 if unit == "F" else measured
            message += f" Selected point: {display:.2f} °{unit}."
        if state.get("result") is not None:
            message += f" Fitted emissivity: {state['result']:.2f}."
        elif reference:
            message += f" Saved emissivity: {reference['emissivity']:.2f}."
        self.status.setText(message)


class ReflectedCalibrationControls(QWidget):
    def __init__(self, send):
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
                lambda _checked=False, operation=operation: send(
                    {"action": "reflected_calibration", "operation": operation}
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

    def update_state(self, state, unit, locked):
        running = bool(state.get("running"))
        self.select.setEnabled(not locked)
        self.measure.setEnabled(not locked and bool(state.get("active")))
        self.apply.setEnabled(not locked and bool(state.get("reference")))
        self.cancel.setEnabled(bool(state.get("active")) and (not locked or running))
        message = state.get("status", "Show the sampling target to begin.")
        reference = state.get("reference")
        if reference:
            value = reference["celsius"]
            if unit == "F":
                value = value * 1.8 + 32
            message += f" Saved: {value:.1f} °{unit}."
        self.status.setText(message)


class ViewWindow(QWidget):
    def __init__(self, send) -> None:
        super().__init__()
        self._send = send
        self._settings_locked = False
        self.controls = {}
        self.rows = {}
        self.hardware_rows = {}
        self.setWindowTitle("Camera")
        self.setMinimumWidth(570)
        self._settings = QSettings(
            QSettings.IniFormat, QSettings.UserScope, "topdon-duo", "desktop"
        )
        layout = QVBoxLayout(self)
        heading = QLabel("Camera")
        heading.setStyleSheet("font-size: 20px; font-weight: bold")
        layout.addWidget(heading)
        self.auto_calibrate = NoWheelCheckBox("Auto calibrate")
        self.auto_calibrate.setToolTip(
            "Allow the camera to calibrate automatically. Use Calibrate now in the image's right-click menu when off."
        )
        self.auto_calibrate.toggled.connect(
            lambda value: self._send({"action": "auto_calibrate", "value": value})
        )
        layout.addWidget(self.auto_calibrate)
        self.fixed_range = NoWheelCheckBox("Fixed mode")
        self.fixed_range.setToolTip(
            "Flatten thermal shading and retain enhanced detail in Camera preview. "
            "Requires detail enhancement. Turning detail enhancement off also turns this off. "
            "Camera processing is restored on exit."
        )
        self.fixed_range.toggled.connect(
            lambda value: self._send({"action": "fixed_range", "value": value})
        )
        self.processing_preset = ControlRow(
            "Camera processing preset",
            lambda value: self._send({"action": "processing_preset", "value": value}),
            options=(("balanced", "Balanced"), ("shadow", "Shadow"), ("soft", "Soft")),
        )
        self.processing_preset.setToolTip(
            "Changes processing inside the camera, independently of its color palette. "
            "Affects Camera preview, not the app's raw thermal colors. "
            "Balanced is the normal preset; Shadow is darker; Soft has a gentler look. "
            "Available with Fixed mode off. Original processing is restored on exit."
        )
        description = QLabel(
            "Adjust controls directly. Use Restore camera settings to return to the "
            "original values. Camera overrides are also restored on exit."
        )
        self.camera_gamma = ControlRow(
            "Camera gamma adjustment",
            lambda value: self._send({"action": "tone", "gamma": int(value)}),
            minimum=0, maximum=100, step=1,
        )
        self.camera_gamma.setToolTip(
            "Adjust camera preview midtones. 50 is neutral and preserves the processing preset. "
            "Updates take several seconds; the image and temperature sampling continue."
        )
        self.camera_boost = ControlRow(
            "Camera tone boost",
            lambda value: self._send({"action": "tone", "boost": int(value)}),
            options=((0, "Off"), (1, "Mode 1"), (2, "Mode 2"), (3, "Mode 3")),
        )
        self.camera_boost.setToolTip(
            "Camera preview tone boost. Modes 1, 2 and 3 set different camera flags; "
            "our tests found similar contrast increases, not ordered strength levels. "
            "Previous On corresponds to Mode 3."
        )
        self.tone_status = QLabel()
        self.cancel_tone = QPushButton("Cancel tone update")
        self.cancel_tone.clicked.connect(lambda: self._send({"action": "cancel_tone"}))
        description.setWordWrap(True)
        layout.addWidget(description)
        self.advanced_auto = NoWheelCheckBox("Advanced / Auto")
        self.advanced_auto.setChecked(True)
        self.advanced_auto.setToolTip(
            "Reserved for future automatic controls. All inputs stay enabled."
        )
        self.advanced_auto.toggled.connect(
            lambda value: self._send({"action": "advanced_auto", "value": value})
        )
        layout.addWidget(self.advanced_auto)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        rows = QVBoxLayout(body)
        rows.setSpacing(16)
        display_options = {
            "image_source": ("Image source", tuple(IMAGE_SOURCES.items())),
            "palette_source": ("Color source", tuple(PALETTE_SOURCES.items())),
            "temperature_unit": ("Measurement units", tuple(TEMPERATURE_UNITS.items())),
            "image_filter": ("Image filter", tuple(IMAGE_FILTERS.items())),
            "upsampling": ("Enhancement algorithm", tuple(UPSCALING_MODES.items())),
            "enhancement_input": ("Enhancement input size", tuple(ENHANCEMENT_INPUTS.items())),
            "color_palette": ("App palette (raw / grayscale)", tuple(COLOR_PALETTES.items())),
            "mirror_horizontal": ("Mirror left / right", ((False, "Off"), (True, "On"))),
            "mirror_vertical": ("Mirror top / bottom", ((False, "Off"), (True, "On"))),
            "antialiasing": ("Antialiasing", ((False, "Off"), (True, "On"))),
        }
        numeric_options = {
            "enhancement_amount": ("Enhancement amount", 0, 1, 0.01),
            "anime4k_passes": ("Anime4K09 passes", 1, 5, 1),
        }
        hardware_switches = tuple(
            name
            for name, spec in HARDWARE_CONTROLS.items()
            if tuple(text for _, text in spec.options) == ("Off", "On")
        )
        for title, display_names, hardware_names in (
            (
                "Display controls",
                (
                    "image_source",
                    "temperature_unit",
                    "image_filter",
                    "palette_source",
                    "color_palette",
                    "mirror_horizontal",
                    "mirror_vertical",
                    "antialiasing",
                ),
                ("palette", *hardware_switches),
            ),
            (
                "AI enhancement",
                (
                    "upsampling",
                    "enhancement_input",
                    "enhancement_amount",
                    "anime4k_passes",
                ),
                (),
            ),
            (
                "Camera adjustments",
                (),
                tuple(
                    name
                    for name in HARDWARE_CONTROLS
                    if name not in hardware_switches and name != "palette"
                ),
            ),
        ):
            heading = QLabel(title)
            heading.setStyleSheet("font-size: 16px; font-weight: bold")
            rows.addWidget(heading)
            if title == "Display controls":
                rows.addWidget(self.processing_preset)
                rows.addWidget(self.camera_boost)
            switches = QGridLayout() if title == "Display controls" else None
            switch_count = 0
            for name in display_names:
                if name in numeric_options:
                    row_title, minimum, maximum, step = numeric_options[name]
                    arguments = {"minimum": minimum, "maximum": maximum, "step": step}
                else:
                    row_title, options = display_options[name]
                    arguments = {"options": options}
                row = ControlRow(
                    row_title,
                    lambda value, name=name: self._send(
                        {"action": "setting", "name": name, "value": value}
                    ),
                    **arguments,
                )
                self.rows[name] = row
                self.controls[name] = row.input
                if row.is_switch and switches is not None:
                    switches.addWidget(row, switch_count // 2, switch_count % 2)
                    switch_count += 1
                else:
                    rows.addWidget(row)
                if name == "color_palette":
                    row.setToolTip(
                        "Colors raw thermal images and grayscale previews. Camera color previews use the camera palette instead."
                    )
                    self.color_status = QLabel()
                    self.color_status.setWordWrap(True)
                    rows.addWidget(self.color_status)
                if name == "enhancement_amount":
                    row.setToolTip("0 gives the original image; 1 gives full enhancement.")
                elif name == "anime4k_passes":
                    row.setToolTip("Used by Anime4K09 only. The phone's setting is 3 passes.")
                elif name == "enhancement_input":
                    row.setToolTip(
                        "Native uses the phone's input size. Full preview retains all supplied "
                        "pixels and takes more processing time."
                    )
            for name in hardware_names:
                spec = HARDWARE_CONTROLS[name]
                row = ControlRow(
                    "Camera palette (color preview)" if name == "palette" else spec.title,
                    lambda value, name=name: self._send(
                        {"action": "hardware", "name": name, "value": value, "enabled": True}
                    ),
                    minimum=spec.minimum,
                    maximum=spec.maximum,
                    step=spec.step,
                    options=spec.options,
                    unit=spec.unit,
                    preserve_input=name == "ambient",
                )
                if name == "palette":
                    row.setToolTip(
                        "Sets colors inside the camera. Used only with Camera preview; choosing a palette enables camera colors for that preview."
                    )
                self.hardware_rows[name] = row
                if row.is_switch and switches is not None:
                    switches.addWidget(row, switch_count // 2, switch_count % 2)
                    switch_count += 1
                else:
                    rows.addWidget(row)
                if name == "brightness":
                    rows.addWidget(self.camera_gamma)
                    rows.addWidget(self.tone_status)
                    rows.addWidget(self.cancel_tone)
                if name == "detail_enabled":
                    switches.addWidget(self.fixed_range, switch_count // 2, switch_count % 2)
                    switch_count += 1
            if switches is not None:
                switches.setHorizontalSpacing(24)
                switches.setVerticalSpacing(16)
                switches.setColumnStretch(0, 1)
                switches.setColumnStretch(1, 1)
                rows.addLayout(switches)
                self.display_switches = switches
        self.distance_calibration = DistanceCalibrationControls(self._send)
        rows.addWidget(self.distance_calibration)
        self.emissivity_calibration = EmissivityCalibrationControls(self._send)
        rows.addWidget(self.emissivity_calibration)
        self.reflected_calibration = ReflectedCalibrationControls(self._send)
        rows.addWidget(self.reflected_calibration)
        rows.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        reset = self.reset_button = QPushButton("Reset display settings")
        reset.clicked.connect(self._reset_display)
        restore = self.restore_button = QPushButton("Restore camera settings")
        restore.clicked.connect(self._restore_hardware)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for button in (reset, restore, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.update_state(VIEW_DEFAULTS)
        size = self._settings.value("camera/window_size", QSize(620, 780))
        self.resize(size if isinstance(size, QSize) and size.isValid() else QSize(620, 780))
        QApplication.instance().aboutToQuit.connect(self._save_window_size)

    def _save_window_size(self) -> None:
        self._settings.setValue("camera/window_size", self.size())
        self._settings.sync()

    def closeEvent(self, event) -> None:
        self._save_window_size()
        super().closeEvent(event)

    def _reset_display(self):
        if self._settings_locked:
            return
        for row in self.rows.values():
            row.timer.stop()
        self._send({"action": "reset"})

    def _restore_hardware(self):
        if self._settings_locked:
            return
        for row in self.hardware_rows.values():
            row.timer.stop()
        self._send({"action": "restore_hardware"})

    def update_state(self, state: dict) -> None:
        tone_busy = bool(state.get("tone_busy", False))
        selected_source = state.get("image_source", VIEW_DEFAULTS["image_source"])
        actual_source = state.get("actual_image_source", selected_source)
        preview_view = selected_source == "preview" and actual_source == "preview"
        settings_locked = bool(state.get("settings_locked", False)) or tone_busy
        if settings_locked and not self._settings_locked:
            focused = QApplication.focusWidget()
            if focused is not None and self.isAncestorOf(focused):
                # Disabling a focused input advances Qt's focus chain into the
                # calibration controls, which scrolls the menu to the bottom.
                self.setFocus(Qt.OtherFocusReason)
        self._settings_locked = settings_locked
        preset_available = (
            not self._settings_locked
            and state.get("processing_preset_available", False)
            and not state.get("fixed_range", False)
            and preview_view
        )
        if not preset_available:
            self.processing_preset.timer.stop()
        self.processing_preset.update_state(
            state.get("processing_preset", "balanced"),
            preset_available,
        )
        for row, value in ((self.camera_gamma, state.get("camera_gamma", 50)),
                           (self.camera_boost, state.get("camera_boost", 0))):
            if not preset_available:
                row.timer.stop()
            row.update_state(value, preset_available)
        self.tone_status.setVisible(tone_busy)
        self.tone_status.setText(f"Updating camera gamma… {state.get('tone_progress', 0)}%")
        self.cancel_tone.setVisible(tone_busy)
        self.cancel_tone.setEnabled(tone_busy and not state.get("settings_locked", False))
        if self._settings_locked:
            for row in (*self.rows.values(), *self.hardware_rows.values()):
                row.timer.stop()
        self.advanced_auto.setEnabled(not self._settings_locked)
        self.auto_calibrate.setEnabled(not self._settings_locked)
        detail = state.get("hardware", {}).get("detail_enabled", {})
        self.fixed_range.setEnabled(
            preview_view and not self._settings_locked and detail.get("available", True)
            and (detail.get("value", 0) == 1 or state.get("fixed_range", False))
            and state.get("processing_preset", "balanced") == "balanced"
            and state.get("camera_gamma", 50) == 50
            and not state.get("camera_boost", 0)
        )
        with QSignalBlocker(self.fixed_range):
            self.fixed_range.setChecked(state.get("fixed_range", False))
        with QSignalBlocker(self.auto_calibrate):
            self.auto_calibrate.setChecked(state.get("auto_calibrate", False))
        self.reset_button.setEnabled(not self._settings_locked)
        self.restore_button.setEnabled(not self._settings_locked)
        with QSignalBlocker(self.advanced_auto):
            self.advanced_auto.setChecked(state.get("advanced_auto", True))
        camera_palette_selected = (
            state.get("palette_source", VIEW_DEFAULTS["palette_source"]) == "camera"
        )
        camera_colors = preview_view and camera_palette_selected
        raw_view = not preview_view
        for name, row in self.rows.items():
            row.update_state(
                state.get(name, VIEW_DEFAULTS[name]),
                not self._settings_locked
                and not (name == "color_palette" and camera_colors)
                and not (name == "palette_source" and selected_source == "raw"),
            )
        for name, row in self.hardware_rows.items():
            row.set_display_unit(state.get("temperature_unit", VIEW_DEFAULTS["temperature_unit"]))
            setting = state.get("hardware", {}).get(name, {})
            row.update_state(
                setting.get("value", HARDWARE_CONTROLS[name].minimum),
                setting.get("available", True)
                and not self._settings_locked
                and not ((HARDWARE_CONTROLS[name].selector == 2 or name == "center_overlay")
                         and not preview_view)
                and not (state.get("fixed_range", False) and HARDWARE_CONTROLS[name].selector == 2
                         and name not in ("detail_enabled", "detail"))
                and not (name == "palette" and not camera_palette_selected),
            )
        if camera_colors:
            self.color_status.setText(
                "Colors: Camera palette. Choose App colors under Color source to use the app palette."
            )
        elif raw_view:
            self.color_status.setText("Colors: App palette. Camera colors require Camera preview.")
        else:
            self.color_status.setText(
                "Colors: App palette. Choose Camera colors with Camera preview to use the camera palette."
            )
        self.distance_calibration.update_state(
            state.get("distance_calibration", {}),
            state.get("temperature_unit", VIEW_DEFAULTS["temperature_unit"]),
            self._settings_locked,
        )
        self.emissivity_calibration.update_state(
            state.get("emissivity_calibration", {}),
            state.get("temperature_unit", VIEW_DEFAULTS["temperature_unit"]),
            self._settings_locked,
        )
        self.reflected_calibration.update_state(
            state.get("reflected_calibration", {}),
            state.get("temperature_unit", "C"),
            self._settings_locked,
        )
        self.status.setText(state.get("status", ""))
        if state.get("raise_window"):
            self.showNormal()
            self.raise_()
            self.activateWindow()


def main() -> int:
    return run_window(ViewWindow, "Camera")


if __name__ == "__main__":
    raise SystemExit(main())
