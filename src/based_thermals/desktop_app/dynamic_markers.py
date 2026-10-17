"""
Display-only extrema and center markers from the oriented native temperature plane.
"""

import cv2
import numpy as np

from ..render import RenderedThermalFrame
from .overlays import draw_contrasting_overlay


def _marker_regions(
    points: list[tuple[int, int]], shape: tuple[int, ...],
) -> list[tuple[tuple[int, int, int, int], list[int]]]:
    """
    Group intersecting padded circles so overlapping halos are composited together.

    Ten pixels cover the radius, antialiasing and both halo dilations with room to spare.
    """
    regions = [
        ((max(0, x - 10), max(0, y - 10), min(shape[1], x + 11), min(shape[0], y + 11)),
         [index])
        for index, (x, y) in enumerate(points)
    ]
    while True:
        for first, (a, indices) in enumerate(regions):
            for second in range(first + 1, len(regions)):
                b, other_indices = regions[second]
                if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                    bounds = (min(a[0], b[0]), min(a[1], b[1]),
                              max(a[2], b[2]), max(a[3], b[3]))
                    regions[first] = (bounds, sorted(indices + other_indices))
                    del regions[second]
                    break
            else:
                continue
            break
        else:
            return regions


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
    points = []
    for y, x in (minimum, maximum, center):
        point = (min(image.shape[1] - 1, int((x + 0.5) * image.shape[1] / width)),
                 min(image.shape[0] - 1, int((y + 0.5) * image.shape[0] / height)))
        points.append(point)
    colors = ((255, 128, 0), (0, 0, 255), (0, 255, 0))
    result = image.copy()
    for (x0, y0, x1, y1), indices in _marker_regions(points, image.shape):
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for index in indices:
            x, y = points[index]
            cv2.circle(mask, (x - x0, y - y0), 5, 255, 1, cv2.LINE_AA)
        halo = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        patch = draw_contrasting_overlay(image[y0:y1, x0:x1], halo, np.zeros_like(mask))
        for index in indices:
            x, y = points[index]
            cv2.circle(patch, (x - x0, y - y0), 5, colors[index], 1, cv2.LINE_AA)
        result[y0:y1, x0:x1] = patch
    return result
