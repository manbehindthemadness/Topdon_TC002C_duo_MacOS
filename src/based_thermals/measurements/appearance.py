"""
Display-only measurement color and alpha compositing, independent of native sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .display_masks import shape_masks

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
    pending_lines = False
    spots.display_cache.prune(spots.numbers)
    for number, center in spots.display_active:
        region = spots.region(number)
        z_index = region.get("z_index", 0)
        if current_z is not None and z_index != current_z and pending_lines:
            result = contrast(result, default_mask, np.zeros_like(default_mask))
            default_mask.fill(0)
            pending_lines = False
        current_z = z_index
        color = region.get("line_color", "dynamic")
        fill, opacity = region.get("fill", False), region.get("opacity", 1.0)
        if opacity == 0:
            continue
        masks = shape_masks(spots, number, center, scale, image.shape[:2])
        if not masks.lines.size:
            continue
        if color == "dynamic" and not fill and opacity == 1:
            view = masks.view(default_mask)
            cv2.max(view, masks.lines, dst=view)
            pending_lines = True
            continue
        view = masks.view(result)
        if color == "dynamic":
            outlined = contrast(view, masks.lines, np.zeros_like(masks.lines))
            target = blend_color(view, masks.filled, color, 1.0)
            target[masks.halo] = outlined[masks.halo]
            x, y, width, height = cv2.boundingRect(masks.coverage)
            local = view[y : y + height, x : x + width]
            overlay = target[y : y + height, x : x + width]
            local[:] = np.rint(local * (1 - opacity) + overlay * opacity).astype(np.uint8)
        else:
            blend_color(view, masks.combined, color, opacity, copy=False)
    return result, default_mask


def blend_color(
    image: np.ndarray,
    mask: np.ndarray,
    color: str,
    opacity: float,
    *,
    copy: bool = True,
) -> np.ndarray:
    """
    Blend RGB hex colors into BGR display pixels, or invert pixels for Dynamic fill.
    """
    result = image.copy() if copy else image
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
