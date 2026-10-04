"""Camera and display controls in a separate Qt process."""

from PySide6.QtCore import QSettings, QSignalBlocker, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    ENHANCEMENT_INPUTS,
    IMAGE_FILTERS,
    IMAGE_SOURCES,
    TEMPERATURE_UNITS,
    UPSCALING_MODES,
    VIEW_DEFAULTS,
)


class NoWheelSlider(QSlider):
    """Let the surrounding scroll area handle mouse-wheel input."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class ControlRow(QWidget):
    """An editable value with a slider for range controls."""

    def __init__(self, title, changed, *, minimum=0, maximum=1, step=1, options=(), unit=""):
        super().__init__()
        self.changed = changed
        self.options = tuple(options)
        self.is_switch = tuple(text for _, text in self.options) == ("Off", "On")
        self.step = step
        self.minimum = minimum
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
        if self.options:
            self.input = QComboBox()
            for value, text in self.options:
                self.input.addItem(text, value)
            self.input.currentIndexChanged.connect(self._input_changed)
        else:
            self.input = QDoubleSpinBox()
            self.input.setRange(minimum, maximum)
            self.input.setDecimals(2 if step == 0.01 else (1 if step == 0.1 else 0))
            self.input.setSingleStep(step)
            self.input.setKeyboardTracking(False)
            self.input.setSuffix(f" {unit}" if unit else "")
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
        return self.input.currentData() if self.options else round(self.input.value(), 2)

    def _set_value(self, value):
        with QSignalBlocker(self.input):
            if self.options:
                index = self.input.findData(value)
                self.input.setCurrentIndex(index)
                position = max(0, index)
            else:
                self.input.setValue(value)
                position = round((value - self.minimum) / self.step)
        if self.slider is not None:
            with QSignalBlocker(self.slider):
                self.slider.setValue(position)

    def _set_enabled(self, enabled):
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
        self.changed(self.value())

    def update_state(self, value, available=True):
        self._set_enabled(available)
        if self.timer.isActive() or (self.slider is not None and self.slider.isSliderDown()):
            return
        if value != self.value():
            self._set_value(value)


class ViewWindow(QWidget):
    def __init__(self, send) -> None:
        super().__init__()
        self._send = send
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
        description = QLabel(
            "Adjust controls directly. Use Restore camera settings to return to the "
            "original values. Camera overrides are also restored on exit."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.advanced_auto = QCheckBox("Advanced / Auto")
        self.advanced_auto.setChecked(True)
        self.advanced_auto.setToolTip(
            "Reserved for future automatic controls. All inputs stay enabled."
        )
        layout.addWidget(self.advanced_auto)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        rows = QVBoxLayout(body)
        rows.setSpacing(16)
        display_options = {
            "image_source": ("Image source", tuple(IMAGE_SOURCES.items())),
            "temperature_unit": ("Temperature unit", tuple(TEMPERATURE_UNITS.items())),
            "image_filter": ("Image filter", tuple(IMAGE_FILTERS.items())),
            "upsampling": ("Upsampling algorithm", tuple(UPSCALING_MODES.items())),
            "enhancement_input": ("Enhancement input size", tuple(ENHANCEMENT_INPUTS.items())),
            "color_palette": ("Display color gradient", tuple(COLOR_PALETTES.items())),
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
                    "color_palette",
                ),
                (),
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
                tuple(name for name in HARDWARE_CONTROLS if name not in hardware_switches),
            ),
            (
                "On / off settings",
                ("mirror_horizontal", "mirror_vertical", "antialiasing"),
                hardware_switches,
            ),
        ):
            heading = QLabel(title)
            heading.setStyleSheet("font-size: 16px; font-weight: bold")
            rows.addWidget(heading)
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
                rows.addWidget(row)
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
                    spec.title,
                    lambda value, name=name: self._send(
                        {"action": "hardware", "name": name, "value": value, "enabled": True}
                    ),
                    minimum=spec.minimum,
                    maximum=spec.maximum,
                    step=spec.step,
                    options=spec.options,
                    unit=spec.unit,
                )
                self.hardware_rows[name] = row
                rows.addWidget(row)
        rows.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        reset = QPushButton("Reset display settings")
        reset.clicked.connect(self._reset_display)
        restore = QPushButton("Restore camera settings")
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
        for row in self.rows.values():
            row.timer.stop()
        self._send({"action": "reset"})

    def _restore_hardware(self):
        for row in self.hardware_rows.values():
            row.timer.stop()
        self._send({"action": "restore_hardware"})

    def update_state(self, state: dict) -> None:
        for name, row in self.rows.items():
            row.update_state(state.get(name, VIEW_DEFAULTS[name]))
        for name, row in self.hardware_rows.items():
            setting = state.get("hardware", {}).get(name, {})
            row.update_state(
                setting.get("value", HARDWARE_CONTROLS[name].minimum),
                setting.get("available", True),
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
