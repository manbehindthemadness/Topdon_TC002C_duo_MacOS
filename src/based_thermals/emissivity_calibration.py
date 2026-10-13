"""Fit hardware emissivity to a known surface temperature using live feedback."""

import math
from itertools import pairwise

import numpy as np

from .camera import CameraError


def validate_reference(value):
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        raise ValueError("Unsupported emissivity calibration reference")
    emissivity, known = value.get("emissivity"), value.get("known_celsius")
    for number in (emissivity, known):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError("Emissivity calibration values must be finite numbers")
    if (
        not 0.01 <= emissivity <= 1
        or abs(emissivity * 100 - round(emissivity * 100)) > 1e-8
        or not -50 <= known <= 550
    ):
        raise ValueError("Invalid emissivity calibration values")
    return {"version": 1, "emissivity": emissivity, "known_celsius": known}


class EmissivityCalibrator:
    SETTLE_SECONDS = 1.0
    SAMPLE_COUNT = 5
    TOLERANCE_C = 0.3
    TIMEOUT_SECONDS = 30

    def __init__(self, hardware, reference=None):
        self.hardware = hardware
        self.reference = validate_reference(reference) if reference is not None else None
        self.active = self.selecting = self.running = False
        self.point = None
        self.selected_celsius = None
        self.selection_id = 0
        self.known_celsius = self.measured_celsius = self.result = None
        self.message = "Select a surface point and enter its independently measured temperature."
        self._original = None
        self._samples = []
        self._observations = {}
        self._finish_pending = None
        self._closing = False

    def begin(self):
        if self.running:
            raise ValueError("Cancel the current emissivity fit first")
        self.active = self.selecting = True
        self._closing = False
        self.point = self.result = self.known_celsius = None
        self.measured_celsius = None
        self.selected_celsius = None
        self.message = "Markers hidden. Click the reference surface on the main image."

    def select_point(self, point, size, temperature_celsius=None):
        x, y = point
        if not self.selecting or not 0 <= x < size[0] or not 0 <= y < size[1]:
            raise ValueError("Select a point inside the thermal image")
        self.point = (int(x), int(y))
        self.selection_id += 1
        self.selected_celsius = temperature_celsius
        if temperature_celsius is not None:
            self.known_celsius = self.measured_celsius = float(temperature_celsius)
        self.selecting = False
        self.message = (
            "Point selected. Edit the copied temperature to its known value, then Fit emissivity."
        )

    def start(self, known_celsius, now):
        if self.running or self.point is None or not self.active:
            raise ValueError("Select a reference point before fitting emissivity")
        if (
            isinstance(known_celsius, bool)
            or not isinstance(known_celsius, (int, float))
            or not math.isfinite(known_celsius)
            or not -50 <= known_celsius <= 550
        ):
            raise ValueError("Known surface temperature must be between -50 and 550 °C")
        setting = self.hardware.state().get("emissivity", {})
        if not setting.get("available"):
            raise ValueError("Camera emissivity control is unavailable")
        self.known_celsius = float(known_celsius)
        self._original = (setting["value"], setting.get("enabled", False))
        self.running = True
        self.result = None
        self._observations = {}
        self._finish_pending = None
        self._started = now
        try:
            self._probe(100, now)
        except (CameraError, TypeError, ValueError):
            self._finish(None, "Fit failed; previous emissivity restored.")
            raise

    def _probe(self, value, now):
        self.hardware.set("emissivity", value / 100, True)
        self._current = value
        self._samples = []
        self._settled_at = now + self.SETTLE_SECONDS
        self.message = f"Fitting emissivity: testing {value / 100:.2f}. Keep the reference steady."

    def _finish(self, result, message):
        # Retain the restore request if a USB error occurs so the next frame retries.
        self._finish_pending = (result, message)
        self.hardware.set("emissivity", *self._original)
        self.running = False
        self._original = None
        self._finish_pending = None
        self.result = result
        if result is not None:
            self.reference = validate_reference(
                {"version": 1, "emissivity": result, "known_celsius": self.known_celsius}
            )
            self.point = None
            self.selected_celsius = None
        self.message = message
        if self._closing:
            self.active = self.selecting = False
            self.point = None
            self.selected_celsius = None

    def cancel(self):
        self._closing = True
        if self.running:
            self._finish(None, "Calibration cancelled; previous emissivity restored.")
        self.active = self.selecting = False
        self.point = None
        self.selected_celsius = None
        self.result = None
        self.message = "Calibration closed. Image markers restored."

    def apply(self):
        if self.running or (self.result is None and self.reference is None):
            raise ValueError("Fit an emissivity value before applying it")
        fitted = self.result if self.result is not None else self.reference["emissivity"]
        self.hardware.set("emissivity", fitted, True)
        value = self.hardware.state()["emissivity"]["value"]
        self.active = self.selecting = False
        self.point = None
        self.selected_celsius = None
        self.message = f"Emissivity {value:.2f} applied to the camera."
        return value

    def update(self, temperatures, now):
        if not self.active or self.point is None:
            return
        if self.running and self._finish_pending is not None:
            self._finish(*self._finish_pending)
            return
        x, y = self.point
        if not 0 <= y < temperatures.shape[0] or not 0 <= x < temperatures.shape[1]:
            self.cancel()
            return
        value = float(temperatures[y, x])
        self.measured_celsius = value if math.isfinite(value) else None
        if not self.running:
            return
        if now - self._started > self.TIMEOUT_SECONDS:
            self._finish(
                None,
                "Fit timed out; previous emissivity restored. Keep the target steady and retry.",
            )
            return
        if now < self._settled_at or not math.isfinite(value):
            return
        self._samples.append(value)
        self._samples = self._samples[-self.SAMPLE_COUNT :]
        if len(self._samples) < self.SAMPLE_COUNT or np.ptp(self._samples) > 0.5:
            return
        measured = float(np.median(self._samples))
        self._observations[self._current] = measured
        if 1 not in self._observations:
            self._probe(1, now)
            return
        low_t, high_t = self._observations[1], self._observations[100]
        if abs(high_t - low_t) < 0.5:
            self._finish(
                None,
                "The camera reading barely responds to emissivity. Choose a surface with more thermal contrast.",
            )
            return
        target = self.known_celsius
        if (
            not min(low_t, high_t) - self.TOLERANCE_C
            <= target
            <= max(low_t, high_t) + self.TOLERANCE_C
        ):
            self._finish(
                None,
                "Known temperature is outside the camera's emissivity response. Previous setting restored.",
            )
            return
        increasing = high_t > low_t
        # Reject a changing target or a response that cannot be bracketed reliably.
        ordered = sorted(self._observations.items())
        if any(
            (second[1] - first[1]) * (1 if increasing else -1) < -0.5
            for first, second in pairwise(ordered)
        ):
            self._finish(
                None,
                "Temperature response changed during fitting. Previous setting restored; retry with a steady target.",
            )
            return
        best, best_temperature = min(ordered, key=lambda item: abs(item[1] - target))
        if abs(best_temperature - target) <= self.TOLERANCE_C:
            self._finish(
                best / 100,
                "Fit ready. Previous emissivity restored; Apply keeps the fitted value globally.",
            )
            return
        bracket = next(
            (
                (first[0], second[0])
                for first, second in pairwise(ordered)
                if min(first[1], second[1]) <= target <= max(first[1], second[1])
            ),
            None,
        )
        if bracket is None or bracket[1] - bracket[0] <= 1:
            self._finish(
                None,
                "No emissivity setting matches within 0.3 °C at the camera's 0.01 precision. Previous setting restored.",
            )
            return
        self._probe((bracket[0] + bracket[1]) // 2, now)

    def state(self):
        return {
            "active": self.active,
            "selecting": self.selecting,
            "running": self.running,
            "point": list(self.point) if self.point else None,
            "selection_id": self.selection_id,
            "selected_celsius": self.selected_celsius,
            "known_celsius": self.known_celsius,
            "measured_celsius": self.measured_celsius,
            "result": self.result,
            "reference": self.reference,
            "status": self.message,
        }
