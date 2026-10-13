"""
Compatibility boundary around the existing audited TC002C Duo backend.
"""

import cv2
import numpy as np

from ..camera import (
    IMAGE_OFFSET,
    decode_duo_frame,
    has_yuy2_preview,
    measurement_frame_status,
    raw_temperatures,
)
from .contracts import CameraProfile, ControlSpec
from .frames import CameraFrame

PROFILE = CameraProfile("duo", "TOPDON TC002C Duo", (256, 192), 25,
                        calibration_key="tc002c-duo-v1")


def temperatures(
    counts: np.ndarray, *, ambient_celsius: float | None = None, native: bool = True,
) -> np.ndarray:
    """
    Preserve desktop camera conversion and the web viewer's legacy ambient anchoring.
    """
    anchor = ambient_celsius if ambient_celsius is not None else 22.0
    return raw_temperatures(counts, ambient_celsius=anchor, offset=50 if native else None)


def decode_frame(frame: bytes) -> CameraFrame:
    """
    Decode Duo layouts only here, preserving their preview fallback and validity rules.
    """
    telemetry, counts, preview = decode_duo_frame(frame)
    bgr = None
    if has_yuy2_preview(frame):
        yuyv = np.frombuffer(frame, np.uint8, offset=IMAGE_OFFSET * 2).reshape(*preview.shape, 2)
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    result = CameraFrame(PROFILE, counts, preview, bgr,
                         measurement_frame_status(telemetry, counts), temperatures, frame)
    # Short Duo planes are transport luminance, not a verified color-preview source.
    if not has_yuy2_preview(frame):
        object.__setattr__(result, "preview_bgr", None)
    return result


def control_specs() -> dict[str, ControlSpec]:
    """
    Publish display metadata without exposing USB selectors, offsets or serializers.
    """
    from ..hardware_controls import HARDWARE_CONTROLS

    return {
        name: ControlSpec(
            spec.title, "choice" if spec.options else "number", spec.minimum, spec.maximum,
            spec.step, spec.unit, spec.options,
            "measurement" if spec.selector == 3 else "preview", default=spec.minimum,
        ) for name, spec in HARDWARE_CONTROLS.items()
    }


def validate_hardware(document: dict) -> None:
    """
    Enforce evidenced Duo combinations without imposing them on another backend.
    """
    enabled = {item["type"]: item["params"] for item in document["hardware"]
               if not item["bypass"]}
    detail = enabled.get("detail", {})
    if detail.get("fixed") and (
        not detail.get("enabled")
        or enabled.get("preset", {}).get("value", "balanced") != "balanced"
        or enabled.get("gamma", {}).get("value", 50) != 50
        or enabled.get("boost", {}).get("value", 0) != 0
    ):
        raise ValueError("Fixed detail requires detail enhancement, Balanced, gamma 50 and boost Off")
