"""
Decoded preview/radiometry and independent device or spot telemetry.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .contracts import CameraProfile


@dataclass(frozen=True)
class ReportedReading:
    """
    Preserve the meaning, source, units and validity of a camera-reported reading.
    """

    name: str
    value: float | None
    unit: str
    spot_id: str | None = None
    sensor_pixel: tuple[int, int] | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        """
        Reject ambiguous or non-finite telemetry before JSON transport and capture.
        """
        if (not self.name or not isinstance(self.unit, str) or type(self.valid) is not bool
                or self.value is not None and (type(self.value) not in (int, float)
                                               or not math.isfinite(self.value))):
            raise ValueError("Invalid camera-reported reading")


@dataclass(frozen=True)
class CameraFrame:
    """
    Keep native measurement data distinct from display pixels and reported telemetry.
    """

    profile: CameraProfile
    raw_counts: np.ndarray | None = None
    preview: np.ndarray | None = None
    preview_bgr: np.ndarray | None = None
    measurement_status: str = ""
    converter: Callable[..., np.ndarray] | None = field(default=None, repr=False)
    source_bytes: bytes | None = field(default=None, repr=False)
    readings: tuple[ReportedReading, ...] = ()
    timestamp: float | None = None
    sequence: int | None = None

    def __post_init__(self) -> None:
        """
        Validate shapes and own immutable copies before asynchronous processing.
        """
        for name in ("raw_counts", "preview", "preview_bgr"):
            plane = getattr(self, name)
            if plane is None:
                continue
            channels = 3 if name == "preview_bgr" else None
            if (not isinstance(plane, np.ndarray)
                    or plane.ndim != (3 if channels else 2)
                    or (channels and plane.shape[2] != channels)
                    or not plane.shape[0] or not plane.shape[1]
                    or plane.shape[0] * plane.shape[1] > 4_000_000
                    or plane.dtype.kind not in "uif"):
                raise ValueError(f"Invalid camera {name} plane")
            if name == "raw_counts" and plane.shape != self.profile.native_size[::-1]:
                raise ValueError("Radiometry does not match native camera geometry")
            if not np.isfinite(plane).all():
                if name != "raw_counts":
                    raise ValueError("Non-finite camera preview")
                if not self.measurement_status:
                    object.__setattr__(self, "measurement_status", "Invalid camera measurements")
            copied = plane.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if self.raw_counts is not None and (not self.profile.radiometry or self.converter is None):
            raise ValueError("Radiometric counts require a verified temperature converter")
        if self.raw_counts is None and self.preview is None and self.preview_bgr is None:
            raise ValueError("Camera frame has no image planes")
        if self.preview is not None and self.preview_bgr is not None and (
            self.preview.shape != self.preview_bgr.shape[:2]
        ):
            raise ValueError("Camera preview planes have different geometry")
        for reading in self.readings:
            if not isinstance(reading, ReportedReading):
                raise TypeError("Invalid camera-reported telemetry")
            if reading.sensor_pixel is not None:
                x, y = reading.sensor_pixel
                width, height = self.profile.native_size
                if type(x) is not int or type(y) is not int or not 0 <= x < width or not 0 <= y < height:
                    raise ValueError("Reported spot lies outside the native sensor")
        if self.preview is None and self.preview_bgr is not None:
            gray = cv2.cvtColor(self.preview_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            gray.setflags(write=False)
            object.__setattr__(self, "preview", gray)
        if self.preview_bgr is None and self.preview is not None:
            bgr = np.repeat(self.preview[..., None], 3, axis=2)
            bgr.setflags(write=False)
            object.__setattr__(self, "preview_bgr", bgr)

    def temperatures(
        self, counts: np.ndarray | None = None, *, ambient_celsius: float | None = None,
        native: bool = True,
    ) -> np.ndarray:
        """
        Convert native counts using this backend's verified numerical interpretation.
        """
        if self.raw_counts is None or self.converter is None:
            raise ValueError("Camera radiometry is unavailable")
        selected = self.raw_counts if counts is None else counts
        result = np.asarray(self.converter(
            selected, ambient_celsius=ambient_celsius,
            native=native if self.profile.id == "duo" else True,
        ))
        if result.shape != np.shape(selected):
            raise ValueError("Temperature converter changed measurement geometry")
        if result.dtype.kind not in "uif" or not np.isfinite(result).all():
            raise ValueError("Temperature converter returned invalid readings")
        return result


def decode_frame(frame: bytes | CameraFrame) -> CameraFrame:
    """
    Accept decoded backend frames or the existing Duo byte-frame public API.
    """
    if isinstance(frame, CameraFrame):
        return frame
    if not isinstance(frame, bytes):
        raise TypeError("Expected decoded camera frame or legacy Duo bytes")
    from .duo import decode_frame as decode_duo

    return decode_duo(frame)
