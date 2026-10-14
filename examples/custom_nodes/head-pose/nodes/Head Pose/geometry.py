"""
Whole-head crop and pose axes adapted from PINTO0309 HFHPE's MIT-licensed demo.

Copyright (c) 2026 Katsuya Hyodo. The complete upstream MIT notice is in __init__.py.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Head:
    """
    Retain a detector box in upright model-image pixels.
    """

    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class Pose:
    """
    Store yaw, pitch, roll and per-angle concentration for one detected head.
    """

    head: Head
    angles: tuple[float, float, float]
    kappa: tuple[float, float, float]


def crop_head(image: np.ndarray, head: Head, size: int) -> np.ndarray:
    """
    Expand each side 5%, shift the long-side square inside the image, then resize.

    Match upstream's zero-padding when the requested square exceeds image dimensions.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = head.box
    side = 1.1 * max(x2 - x1, y2 - y1)
    left = min(max((x1 + x2 - side) / 2, 0), max(width - side, 0))
    top = min(max((y1 + y2 - side) / 2, 0), max(height - side, 0))
    crop = image[round(top):round(min(top + side, height)),
                 round(left):round(min(left + side, width))]
    h, w = int(crop.shape[0]), int(crop.shape[1])
    if not h or not w:
        raise ValueError("Detected head has an empty crop")
    if h != w:
        padded_side = max(h, w)
        oy, ox = (padded_side - h) // 2, (padded_side - w) // 2
        crop = cv2.copyMakeBorder(crop, oy, padded_side - h - oy,
                                  ox, padded_side - w - ox, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    interpolation = cv2.INTER_AREA if size < crop.shape[0] else cv2.INTER_LANCZOS4
    resized = cv2.resize(crop, (size, size), interpolation=interpolation)
    return resized


def draw_axes(image: np.ndarray, pose: Pose) -> None:
    """
    Project R_image_roll @ R_yaw @ R_pitch onto the original detected head center.
    """
    yaw, pitch, roll = np.radians(pose.angles)
    cy, cp, cr = np.cos([yaw, pitch, roll])
    sy, sp, sr = np.sin([yaw, pitch, roll])
    rotation = np.array([
        [cr * cy, sr * cp - cr * sy * sp, cr * sy * cp + sr * sp],
        [-sr * cy, cr * cp + sr * sy * sp, cr * sp - sr * sy * cp],
        [-sy, -cy * sp, cy * cp],
    ])
    x1, y1, x2, y2 = pose.head.box
    center = (round((x1 + x2) / 2), round((y1 + y2) / 2))
    length = 0.45 * max(x2 - x1, y2 - y1)
    for basis, color in (((1, 0, 0), (0, 0, 255)),
                         ((0, 1, 0), (0, 255, 0)),
                         ((0, 0, -1), (255, 0, 0))):
        vector = rotation @ np.array(basis)
        tip = (round(center[0] + length * vector[0]),
               round(center[1] + length * vector[1]))
        cv2.line(image, center, tip, color, 2, cv2.LINE_AA)


def original_anchor(x: int, y: int, shape: tuple[int, int], turns: int) -> tuple[int, int]:
    """
    Map an upright model-image label anchor back through the input-only rotation.
    """
    height, width = shape
    anchors = ((x, y), (width - 1 - y, x),
               (width - 1 - x, height - 1 - y), (y, height - 1 - x))
    anchor = anchors[turns]
    return anchor
