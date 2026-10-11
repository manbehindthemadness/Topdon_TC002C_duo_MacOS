"""
Typed control rows shared by camera calibration and the pipeline editor.
"""

from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .view_settings import DISTANCE_METERS_PER_UNIT


class NoWheelSlider(QSlider):
    """
    Let the surrounding scroll area handle mouse-wheel input.
    """

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Leave scrolling to the surrounding scroll area.
        """
        event.ignore()


class NoWheelComboBox(QComboBox):
    """
    Keep choices unchanged while scrolling the surrounding menu.
    """

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Leave scrolling to the surrounding scroll area.
        """
        event.ignore()


class NoWheelSpinBox(QDoubleSpinBox):
    """
    Keep numeric values unchanged while scrolling the surrounding menu.
    """

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Leave scrolling to the surrounding scroll area.
        """
        event.ignore()


class NoWheelCheckBox(QCheckBox):
    """
    Keep the toggle unchanged while scrolling its surrounding menu.
    """

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Leave scrolling to the surrounding scroll area.
        """
        event.ignore()


class ControlRow(QWidget):
    """
    An editable value with a slider for range controls.
    """

    input: NoWheelCheckBox | NoWheelComboBox | NoWheelSpinBox

    def __init__(
        self,
        title: str,
        changed: Callable[[Any], None],
        *,
        minimum: float = 0,
        maximum: float = 1,
        step: float = 1,
        options: Sequence[tuple[Any, str]] = (),
        unit: str = "",
        preserve_input: bool = False,
    ) -> None:
        """
        Build one numeric, choice or toggle control with debounced user edits.
        """
        super().__init__()
        self.preserve_input = preserve_input
        self.changed = changed
        self.options = tuple(options)
        self.is_switch = tuple(text for _, text in self.options) == ("Off", "On")
        self.step = step
        self.minimum = minimum
        self.maximum = maximum
        exponents = (Decimal(str(number)).normalize().as_tuple().exponent
                     for number in (step, minimum))
        self.native_decimals = max(0, *(-exponent for exponent in exponents
                                       if isinstance(exponent, int)))
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
            toggle = NoWheelCheckBox("Enabled")
            self.input = toggle
            toggle.toggled.connect(self._input_changed)
        elif self.options:
            combo = NoWheelComboBox()
            self.input = combo
            for value, text in self.options:
                combo.addItem(text, value)
            combo.currentIndexChanged.connect(self._input_changed)
        else:
            numeric = NoWheelSpinBox()
            self.input = numeric
            numeric.setRange(self._display_value(minimum), self._display_value(maximum))
            increment = step * 100 if self.is_distance else step
            exponent = Decimal(str(increment)).normalize().as_tuple().exponent
            step_decimals = max(0, -exponent) if isinstance(exponent, int) else 0
            native_decimals = self.native_decimals - 2 if self.is_distance else self.native_decimals
            numeric.setDecimals(max(0, step_decimals, native_decimals))
            numeric.setSingleStep(step * 100 if self.is_distance else step)
            numeric.setKeyboardTracking(False)
            numeric.setSuffix(" cm" if self.is_distance else f" {unit}" if unit else "")
            numeric.valueChanged.connect(self._input_changed)
        self.input.setMinimumWidth(160)
        self.input.setAccessibleName(title)
        line.addWidget(self.input)
        self.slider = None
        if not self.options:
            self.slider = NoWheelSlider(Qt.Orientation.Horizontal)
            self.slider.setAccessibleName(f"{title} slider")
            self.slider.setRange(0, round((maximum - minimum) / step))
            self.slider.valueChanged.connect(self._slider_changed)
            self.slider.sliderReleased.connect(self._emit)
            line.addWidget(self.slider, 1)
        else:
            line.addStretch()
        layout.addLayout(line)

    def value(self) -> Any:
        """
        Return the selected choice or the numeric value in native units.
        """
        if isinstance(self.input, NoWheelCheckBox):
            return self.options[int(self.input.isChecked())][0]
        if isinstance(self.input, NoWheelComboBox):
            selected = self.input.currentData()
            return selected
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
        return round(value, self.native_decimals)

    def _display_value(self, value: float) -> float:
        """
        Convert a native numeric value into the current display units.
        """
        if self.is_distance:
            return value / DISTANCE_METERS_PER_UNIT[self.temperature_unit]
        if self.temperature_unit == "F" and self.is_temperature:
            return value * 1.8 + 32
        return value

    def set_display_unit(self, unit: str) -> None:
        """
        Change display units while preserving the native control value.
        """
        if (
            not isinstance(self.input, NoWheelSpinBox)
            or not (self.is_temperature or self.is_distance)
            or unit == self.temperature_unit
        ):
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
        exact_step = Decimal(str(self.step))
        if self.is_temperature and unit == "F":
            exact_step *= Decimal("1.8")
        elif self.is_distance:
            exact_step /= Decimal(str(DISTANCE_METERS_PER_UNIT[unit]))
        exponent = exact_step.normalize().as_tuple().exponent
        if isinstance(exponent, int):
            decimals = min(8, max(decimals, -exponent))
        if unit == "C":
            native_decimals = self.native_decimals - 2 if self.is_distance else self.native_decimals
            decimals = max(decimals, native_decimals)
        with QSignalBlocker(self.input):
            self.input.setDecimals(decimals)
            self.input.setRange(
                self._display_value(self.minimum), self._display_value(self.maximum)
            )
            self.input.setSingleStep(step)
            self.input.setSuffix(suffix)
        self._set_value(value)

    def _set_value(self, value: Any) -> None:
        """
        Synchronize the input and slider without emitting user edits.
        """
        with QSignalBlocker(self.input):
            if isinstance(self.input, NoWheelCheckBox):
                self.input.setChecked(value == self.options[1][0])
                position = int(self.input.isChecked())
            elif isinstance(self.input, NoWheelComboBox):
                index = self.input.findData(value)
                self.input.setCurrentIndex(index)
                position = max(0, index)
            else:
                self.input.setValue(self._display_value(value))
                position = round((value - self.minimum) / self.step)
        if self.slider is not None:
            with QSignalBlocker(self.slider):
                self.slider.setValue(position)

    def _set_enabled(self, enabled: bool) -> None:
        """
        Enable the row or cancel its pending edit before disabling it.
        """
        if not enabled:
            self.timer.stop()
        self.input.setEnabled(enabled)
        if self.slider is not None:
            self.slider.setEnabled(enabled)

    def set_numeric_limits(self, minimum: float, maximum: float) -> None:
        """
        Change numeric bounds without generating a user edit.
        """
        if not isinstance(self.input, NoWheelSpinBox):
            return
        self.minimum, self.maximum = minimum, maximum
        with QSignalBlocker(self.input):
            self.input.setRange(self._display_value(minimum), self._display_value(maximum))
        if self.slider is not None:
            with QSignalBlocker(self.slider):
                self.slider.setRange(0, round((maximum - minimum) / self.step))

    def _input_changed(self, *_args: Any) -> None:
        """
        Synchronize an input edit and debounce its outgoing value.
        """
        self._set_value(self.value())
        if self.input.isEnabled():
            self.timer.start()

    def _slider_changed(self, position: int) -> None:
        """
        Synchronize a slider edit and debounce changes outside an active drag.
        """
        value = self.minimum + position * self.step
        self._set_value(value)
        if self.input.isEnabled() and self.slider is not None and not self.slider.isSliderDown():
            self.timer.start()

    def _emit(self) -> None:
        """
        Send the current enabled value and clear its pending debounce.
        """
        self.timer.stop()
        if self.input.isEnabled():
            self.changed(self.value())

    def update_state(self, value: Any, available: bool = True) -> None:
        """
        Apply external state while preserving active local edits.
        """
        self._set_enabled(available)
        if self.timer.isActive() or (self.slider is not None and self.slider.isSliderDown()):
            return
        if value != self.value():
            self._set_value(value)
