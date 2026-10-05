"""Measure reflected apparent temperature using a manually positioned IR reflector."""

import math

import numpy as np

from .camera import CameraError


def validate_reference(value):
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        raise ValueError("Unsupported reflected-temperature reference")
    temperature = value.get("celsius")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not -50 <= temperature <= 100
    ):
        raise ValueError("Reflected temperature must be between -50 and 100 °C")
    return {"version": 1, "celsius": round(temperature, 1)}


class ReflectedCalibrator:
    RADIUS = 4
    SETTLE_SECONDS = 1.0
    STABLE_SECONDS = 2.0
    TIMEOUT_SECONDS = 30.0

    def __init__(self, hardware, reference=None):
        self.hardware = hardware
        self.reference = validate_reference(reference) if reference is not None else None
        self.active = self.running = False
        self.measured_celsius = None
        self.message = "Position a shiny metal reflector over the sampling circle."
        self._original = {}
        self._samples = []
        self._pending = None
        self._closing = False

    def begin(self):
        if self.running or self._original:
            raise ValueError("Close the current measurement first")
        self.active = True
        self.measured_celsius = None
        self.message = "Cover the entire circle with a shiny spoon or foil reflector, then Measure."

    def start(self, now):
        if not self.active or self.running or self._original:
            raise ValueError("Open the reflector target before measuring")
        state = self.hardware.state()
        for name in ("emissivity", "transmission"):
            if not state.get(name, {}).get("available"):
                raise ValueError(f"Camera {name} control is unavailable")
        self._original = {
            name: (state[name]["value"], state[name].get("enabled", False))
            for name in ("emissivity", "transmission")
        }
        self.running = True
        self._closing = False
        self._pending = None
        self._samples = []
        self._started = now
        try:
            self.hardware.set("emissivity", 1.0, True)
            self.hardware.set("transmission", 100, True)
        except (CameraError, ValueError, TypeError):
            self._finish(None, "Measurement failed; original camera settings restored.")
            raise
        self.message = "Measuring reflected apparent temperature. Keep the spoon and camera steady."

    def _finish(self, result, message):
        self._pending = (result, message)
        # Remove only successfully restored fields; retry the others on the next frame.
        for name in list(self._original):
            self.hardware.set(name, *self._original[name])
            del self._original[name]
        self.running = False
        self._pending = None
        if result is not None:
            self.reference = validate_reference({"version": 1, "celsius": result})
            self.active = False
        if self._closing:
            self.active = False
        self.message = message

    def cancel(self):
        self._closing = True
        if self._original:
            self._finish(None, "Calibration closed; original camera settings restored.")
        self.active = self.running = False
        self.message = "Calibration closed."

    def update(self, temperatures, now):
        if self._pending is not None:
            self._finish(*self._pending)
            return
        if not self.active:
            return
        height, width = temperatures.shape
        y, x = np.ogrid[:height, :width]
        values = temperatures[(x - width // 2) ** 2 + (y - height // 2) ** 2 <= self.RADIUS**2]
        if not len(values) or not np.isfinite(values).all():
            self.measured_celsius = None
            self._samples = []
        else:
            self.measured_celsius = float(np.median(values))
        if not self.running:
            return
        if now - self._started > self.TIMEOUT_SECONDS:
            self._finish(
                None,
                "No stable reading found. Previous settings restored; reposition the reflector and retry.",
            )
            return
        if now - self._started < self.SETTLE_SECONDS or self.measured_celsius is None:
            return
        # Sample at most ten times per second, independent of camera frame rate.
        if self._samples and now - self._samples[-1][0] < 0.1:
            return
        self._samples.append((now, self.measured_celsius))
        while len(self._samples) > 1 and self._samples[1][0] <= now - self.STABLE_SECONDS:
            self._samples.pop(0)
        if len(self._samples) < 5 or now - self._samples[0][0] < self.STABLE_SECONDS:
            return
        readings = [sample[1] for sample in self._samples]
        if np.ptp(readings) > 0.3:
            return
        result = float(np.median(readings))
        if not -50 <= result <= 100:
            self._finish(
                None,
                "Reading is outside the camera's reflected-temperature range. Settings restored.",
            )
            return
        self._finish(
            result,
            "Stable reflected temperature saved. Original settings restored; Apply sends the saved value to the camera.",
        )

    def apply(self):
        if self.running or self._original or self.reference is None:
            raise ValueError("Save a stable reflected temperature first")
        self.hardware.set("reflected", self.reference["celsius"], True)
        value = self.hardware.state()["reflected"]["value"]
        self.active = False
        self.message = "Saved reflected temperature applied to the camera."
        return value

    def state(self):
        return {
            "active": self.active,
            "running": self.running,
            "measured_celsius": self.measured_celsius,
            "reference": self.reference,
            "status": self.message,
        }
