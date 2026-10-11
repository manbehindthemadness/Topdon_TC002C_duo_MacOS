"""
Offline synthetic dual-camera backend; no USB, real scene or thermometry claims.
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import Event
from time import monotonic

import cv2
import numpy as np

from ..camera import CameraError
from .contracts import CameraProfile, ControlSpec, ControlValue
from .controls import DeviceControls
from .frames import CameraFrame
from .visible import VisibleFrame

PROFILE = CameraProfile(
    "virtual-dual",
    "Virtual dual camera — experimental; physical hardware untested",
    (80, 60),
    25,
    visible=True,
    experimental=True,
    calibration_key="synthetic-v1",
)


def synthetic_temperatures(
    counts: np.ndarray,
    *,
    ambient_celsius: float | None = None,
    native: bool = True,
) -> np.ndarray:
    """
    Convert synthetic centi-Celsius counts without asserting a real camera encoding.
    """
    del ambient_celsius, native
    return counts.astype(np.float32) / 100


class VirtualDualCamera:
    """
    Generate independent 25 fps thermal and 15 fps optical streams with interruptible pacing.
    """

    profile = PROFILE
    timeout_ms = 100

    def __init__(self) -> None:
        """
        Construct without opening devices or starting threads.
        """
        self._stop = Event()
        self._opened = False
        self._brightness = {"thermal": 50, "visible": 50}

    def open(self) -> VirtualDualCamera:
        """
        Start an offline stream and permit reopen after orderly cancellation.
        """
        self._stop.clear()
        self._opened = True
        return self

    def make_frame(
        self,
        sequence: int,
        *,
        thermal_valid: bool = True,
        visible_valid: bool = True,
        registered: bool = True,
    ) -> CameraFrame:
        """
        Produce deterministic unequal-size/rate scenes with injectable independent faults.
        """
        timestamp = sequence / self.profile.frame_rate
        optical_sequence = int(timestamp * 15)
        optical_time = optical_sequence / 15
        y, x = np.indices((60, 80))
        cx = 40 + 20 * np.sin(timestamp)
        counts = np.rint((22 + 30 * np.exp(-((x - cx) ** 2 + (y - 30) ** 2) / 90)) * 100).astype(
            np.uint16
        )
        preview = cv2.resize(
            ((counts.astype(float) - 2200) / 30).clip(0, 255).astype(np.uint8), (160, 120)
        )
        optical = np.zeros((240, 320, 3), np.uint8)
        optical[:] = (100, 60, 25)
        optical[:, ::20] = (180, 150, 80)
        cv2.circle(
            optical, (round((40 + 20 * np.sin(optical_time)) * 4), 120), 24, (30, 220, 240), -1
        )
        cv2.putText(
            optical, "SYNTHETIC", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        preview = (
            (np.asarray(preview).astype(float) + (self._brightness["thermal"] - 50) * 2)
            .clip(0, 255)
            .astype(np.uint8)
        )
        optical = (
            (optical.astype(float) + (self._brightness["visible"] - 50) * 2)
            .clip(0, 255)
            .astype(np.uint8)
        )
        stream = VisibleFrame(
            optical if visible_valid else None,
            optical_time,
            optical_sequence,
            "" if visible_valid else "Synthetic optical stream missing",
            (0.25, 0, 0, 0, 0.25, 0, 0, 0, 1) if registered else None,
            "Synthetic common coordinate grid; not physical lens calibration" if registered else "",
        )
        frame = CameraFrame(
            self.profile,
            counts,
            preview,
            measurement_status="" if thermal_valid else "Synthetic thermal FFC",
            converter=synthetic_temperatures,
            timestamp=timestamp,
            sequence=sequence,
            visible=stream,
        )
        return frame

    def frames(self) -> Iterator[CameraFrame]:
        """
        Yield the newest synthetic scene rather than queueing delayed frames.
        """
        if not self._opened:
            raise CameraError("Open the virtual camera before streaming")
        started = monotonic()
        sequence = 0
        while not self._stop.is_set():
            yield self.make_frame(sequence)
            sequence = max(sequence + 1, int((monotonic() - started) * self.profile.frame_rate))
            delay = started + sequence / self.profile.frame_rate - monotonic()
            if self._stop.wait(max(0, delay)):
                return

    def stop_stream(self) -> None:
        """
        Wake a paced acquisition immediately without freeing owned frame data.
        """
        self._stop.set()

    def close(self) -> None:
        """
        Stop acquisition without touching USB devices or host drivers.
        """
        self.stop_stream()
        self._opened = False
        self._brightness = {"thermal": 50, "visible": 50}

    def create_controls(self) -> DeviceControls:
        """
        Create the synthetic thermal settings owner without changing any physical device.
        """
        return self._controls("thermal")

    def create_visible_controls(self) -> DeviceControls:
        """
        Create an independent optical owner, including independently named brightness.
        """
        return self._controls("visible")

    def _controls(self, stream: str) -> DeviceControls:
        """
        Bind reads and writes to one synthetic camera's settings bank.
        """
        spec = ControlSpec("Synthetic brightness", "number", minimum=0, maximum=100, default=50)

        def read(name: str, spot: str | None) -> ControlValue:
            """
            Read the selected synthetic bank.
            """
            if name != "brightness" or spot is not None:
                raise CameraError("Unknown synthetic control")
            return self._brightness[stream]

        def write(name: str, value: ControlValue, spot: str | None) -> None:
            """
            Write only the selected synthetic bank.
            """
            if name != "brightness" or spot is not None:
                raise CameraError("Unknown synthetic control")
            spec.validate(value)
            self._brightness[stream] = int(value)

        return DeviceControls(self.profile, {"brightness": spec}, read, write)
