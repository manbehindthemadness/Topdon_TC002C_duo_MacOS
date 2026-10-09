"""
Deterministic camera frame builders shared by application tests.
"""

import numpy as np

from topdon_duo.camera import (
    FRAME_MAGIC,
    FRAME_U16,
    HEADER_U16,
    IMAGE_OFFSET,
    SENSOR_HEIGHT,
    SENSOR_PIXELS,
    SENSOR_WIDTH,
)


def make_frame(raw_value: int = 20_000) -> bytes:
    """
    Build a complete native sensor frame with constant raw counts.
    """
    values = np.zeros(FRAME_U16, dtype="<u2")
    values[0] = FRAME_MAGIC & 0xFFFF
    values[1] = FRAME_MAGIC >> 16
    values[HEADER_U16 : HEADER_U16 + SENSOR_PIXELS] = raw_value
    return values.tobytes()


def frame_with_preview(preview_scale: int = 2) -> tuple[bytes, np.ndarray]:
    """
    Append preview detail that tests can distinguish from sensor measurements.
    """
    preview = np.zeros((SENSOR_HEIGHT * preview_scale, SENSOR_WIDTH * preview_scale), np.uint8)
    # Place detail beyond the first 256x192 pixels of the large preview.
    preview[preview.shape[0] // 2 :, preview.shape[1] // 2 :] = 200
    words = preview.astype("<u2") | 0x8000
    return make_frame()[: IMAGE_OFFSET * 2] + words.tobytes(), preview
