"""
Display-only extrema and center markers from the oriented native temperature plane.
"""

import cv2
import numpy as np

from ..render import RenderedThermalFrame
from .overlays import draw_contrasting_overlay


def draw_dynamic_markers(image: np.ndarray, rendered: RenderedThermalFrame) -> np.ndarray:
    """
    Draw colored 10-pixel circle lines over inverted, dynamically contrasting halos.

    Positions follow the same oriented temperatures as the global readouts. Pixel filters
    and palettes do not influence selection; the input image and measurement arrays are retained.
    """
    temperatures = rendered.temperatures_celsius
    if (
        not rendered.measurements_valid
        or not rendered.measurement_mapping_valid
        or not rendered.radiometry_available
        or temperatures.ndim != 2
        or not temperatures.size
        or not np.isfinite(temperatures).all()
    ):
        return image
    height, width = temperatures.shape
    minimum = np.unravel_index(np.argmin(temperatures), temperatures.shape)
    maximum = np.unravel_index(np.argmax(temperatures), temperatures.shape)
    center = (height // 2, width // 2)
    mask = np.zeros(image.shape[:2], np.uint8)
    points = []
    for y, x in (minimum, maximum, center):
        point = (min(image.shape[1] - 1, int((x + 0.5) * image.shape[1] / width)),
                 min(image.shape[0] - 1, int((y + 0.5) * image.shape[0] / height)))
        cv2.circle(mask, point, 5, 255, 1, cv2.LINE_AA)
        points.append(point)
    halo = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    result = draw_contrasting_overlay(image, halo, np.zeros_like(mask))
    for point, color in zip(points, ((255, 128, 0), (0, 0, 255), (0, 255, 0)), strict=True):
        cv2.circle(result, point, 5, color, 1, cv2.LINE_AA)
    return result
