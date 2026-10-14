"""
Display-only measurement color and alpha compositing, independent of native sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .overlays import display_boundary, draw_boundary

if TYPE_CHECKING:
    from ..desktop_app.spots import SampleSpots

Contrast = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def styled_shapes(
    image: np.ndarray,
    spots: SampleSpots,
    scale: int,
    contrast: Contrast,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite Z layers, unioning legacy dynamic outlines within each layer.
    """
    result = image.copy()
    default_mask = np.zeros(image.shape[:2], np.uint8)
    current_z = None
    for number, center in spots.display_active:
        region = spots.region(number)
        z_index = region.get("z_index", 0)
        if current_z is not None and z_index != current_z:
            result = contrast(result, default_mask, np.zeros_like(default_mask))
            default_mask.fill(0)
        current_z = z_index
        color = region.get("line_color", "dynamic")
        fill, opacity = region.get("fill", False), region.get("opacity", 1.0)
        if color == "dynamic" and not fill and opacity == 1:
            draw_boundary(default_mask, region, center, scale)
            continue
        if opacity == 0:
            continue
        lines = np.zeros_like(default_mask)
        draw_boundary(lines, region, center, scale)
        filled = np.zeros_like(default_mask)
        if fill and not (region["kind"] == "spot" and region["radius"] == 0):
            points = display_boundary(region, center, scale)
            cv2.fillPoly(filled, [points], 255, lineType=cv2.LINE_AA)
            if region["exclude"]:
                perimeter = np.zeros_like(default_mask)
                cv2.polylines(perimeter, [points], True, 255, 1, cv2.LINE_AA)
                filled[perimeter > 0] = 0
        if color == "dynamic":
            outlined = contrast(result, lines, np.zeros_like(lines))
            target = blend_color(result, filled, color, 1.0)
            halo = cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0
            target[halo] = outlined[halo]
            coverage = cv2.dilate(cv2.max(filled, lines), np.ones((3, 3), np.uint8))
            x, y, width, height = cv2.boundingRect(coverage)
            view = result[y : y + height, x : x + width]
            overlay = target[y : y + height, x : x + width]
            view[:] = np.rint(view * (1 - opacity) + overlay * opacity).astype(np.uint8)
        else:
            result = blend_color(result, cv2.max(lines, filled), color, opacity)
    return result, default_mask


def blend_color(image: np.ndarray, mask: np.ndarray, color: str, opacity: float) -> np.ndarray:
    """
    Blend RGB hex colors into BGR display pixels, or invert pixels for Dynamic fill.
    """
    result = image.copy()
    x, y, width, height = cv2.boundingRect(mask)
    if width == 0 or height == 0:
        return result
    view = image[y : y + height, x : x + width]
    target = (
        255 - view
        if color == "dynamic"
        else np.array([int(color[i : i + 2], 16) for i in (5, 3, 1)], dtype=np.uint8)
    )
    alpha = (mask[y : y + height, x : x + width].astype(np.float32) * (opacity / 255))[..., None]
    result[y : y + height, x : x + width] = np.rint(view * (1 - alpha) + target * alpha).astype(
        np.uint8
    )
    return result
