"""
Experimental independent visible-light images, freshness and optical registration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .frames import CameraFrame


@dataclass(frozen=True)
class VisibleFrame:
    """
    Own optical pixels and timing independently of thermal display/radiometry.

    Timestamps use the same monotonic clock as the enclosing CameraFrame. An
    optional 3x3 homography maps optical pixel centers to native thermal centers.
    """

    bgr: np.ndarray | None = field(default=None, repr=False)
    timestamp: float | None = None
    sequence: int | None = None
    status: str = ""
    to_thermal: tuple[float, ...] | None = None
    registration_source: str = ""

    def __post_init__(self) -> None:
        """
        Validate finite metadata and copy pixels before asynchronous branch dispatch.
        """
        if self.timestamp is not None and (
            type(self.timestamp) not in (int, float) or not math.isfinite(self.timestamp)
        ):
            raise ValueError("Invalid visible timestamp")
        if self.sequence is not None and (type(self.sequence) is not int or self.sequence < 0):
            raise ValueError("Invalid visible sequence")
        if not isinstance(self.status, str) or not isinstance(self.registration_source, str):
            raise TypeError("Invalid visible status/provenance")
        if self.bgr is not None:
            if (
                not isinstance(self.bgr, np.ndarray)
                or self.bgr.dtype != np.uint8
                or self.bgr.ndim != 3
                or self.bgr.shape[2] != 3
                or min(self.bgr.shape[:2]) <= 0
                or math.prod(self.bgr.shape[:2]) > 4_000_000
            ):
                raise ValueError("Visible image must be bounded uint8 BGR")
            image = self.bgr.copy()
            image.setflags(write=False)
            object.__setattr__(self, "bgr", image)
        if self.to_thermal is not None:
            matrix = np.asarray(self.to_thermal)
            if (
                matrix.shape != (9,)
                or matrix.dtype.kind not in "uif"
                or not np.isfinite(matrix).all()
                or not self.registration_source
                or abs(np.linalg.det(matrix.reshape(3, 3))) < 1e-12
            ):
                raise ValueError("Registration requires an invertible homography and provenance")
            object.__setattr__(self, "to_thermal", tuple(float(value) for value in matrix))


def visible_status(frame: CameraFrame) -> str:
    """
    Reject absent, invalid or stale optical data without affecting thermal measurements.
    """
    stream = frame.visible
    if not frame.profile.visible:
        return "Visible light is unsupported by this camera"
    if stream is None or stream.bgr is None:
        return "Visible light frame is unavailable"
    if stream.status:
        return stream.status
    if (
        frame.timestamp is not None
        and stream.timestamp is not None
        and (abs(frame.timestamp - stream.timestamp) > frame.profile.visible_max_age)
    ):
        return "Visible light frame is stale or unsynchronized"
    return ""


def visible_image(frame: CameraFrame) -> np.ndarray:
    """
    Return optical imagery, registering it to native thermal geometry when supplied.
    """
    status = visible_status(frame)
    if status:
        raise ValueError(status)
    stream = frame.visible
    assert stream is not None and stream.bgr is not None
    if stream.to_thermal is None:
        return stream.bgr.astype(np.float32)
    image = cv2.warpPerspective(
        stream.bgr,
        np.asarray(stream.to_thermal).reshape(3, 3),
        frame.profile.native_size,
        flags=cv2.INTER_LINEAR,
    )
    return image.astype(np.float32)


def visible_mapping_valid(frame: CameraFrame) -> bool:
    """
    Allow spatial temperature overlays only for fresh registration covering the full grid.
    """
    stream = frame.visible
    if (
        visible_status(frame)
        or stream is None
        or stream.bgr is None
        or stream.to_thermal is None
        or frame.timestamp is None
        or stream.timestamp is None
    ):
        return False
    thermal_time = (
        frame.thermal_timestamp if frame.thermal_timestamp is not None else frame.timestamp
    )
    if abs(thermal_time - stream.timestamp) > frame.profile.visible_max_age:
        return False
    coverage = cv2.warpPerspective(
        np.ones(stream.bgr.shape[:2], np.uint8),
        np.asarray(stream.to_thermal).reshape(3, 3),
        frame.profile.native_size,
        flags=cv2.INTER_NEAREST,
    )
    return bool(np.all(coverage))


def stream_metadata(frame: CameraFrame) -> dict:
    """
    Serialize independent stream timing, freshness and registration for captures/UI.
    """
    stream = frame.visible
    return {
        "thermal": {
            "timestamp": frame.thermal_timestamp
            if frame.thermal_timestamp is not None
            else frame.timestamp,
            "sequence": frame.thermal_sequence
            if frame.thermal_sequence is not None
            else frame.sequence,
            "status": frame.measurement_status,
        },
        "visible": {
            "available": not bool(visible_status(frame)),
            "status": visible_status(frame),
            "timestamp": stream.timestamp if stream else None,
            "sequence": stream.sequence if stream else None,
            "size": list(stream.bgr.shape[1::-1]) if stream and stream.bgr is not None else None,
            "to_thermal": list(stream.to_thermal) if stream and stream.to_thermal else None,
            "registration_source": stream.registration_source if stream else "",
            "mapping_valid": visible_mapping_valid(frame),
        },
    }


def hold_thermal_frame(current: CameraFrame, held: CameraFrame) -> CameraFrame:
    """
    Hold thermal pixels during invalid radiometry while retaining the latest optical stream.
    """
    if not current.profile.visible:
        return held
    return replace(
        held,
        visible=current.visible,
        timestamp=current.timestamp,
        sequence=current.sequence,
        thermal_timestamp=held.timestamp,
        thermal_sequence=held.sequence,
        measurement_status=current.measurement_status,
    )
