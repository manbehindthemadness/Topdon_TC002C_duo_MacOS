"""Cached local raster masks including antialiasing and contrast padding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .display_cache import geometry_key
from .overlays import draw_boundary

if TYPE_CHECKING:
    from ..desktop_app.spots import SampleSpots


@dataclass(frozen=True)
class ShapeMasks:
    """
    Local outline/fill masks and their expanded contrast footprint.
    """

    x: int
    y: int
    lines: np.ndarray
    filled: np.ndarray
    combined: np.ndarray
    halo: np.ndarray
    coverage: np.ndarray

    @property
    def byte_size(self) -> int:
        """
        Account for all retained raster arrays in the display cache budget.
        """
        return sum(
            mask.nbytes
            for mask in (self.lines, self.filled, self.combined, self.halo, self.coverage)
        )

    def view(self, image: np.ndarray) -> np.ndarray:
        """
        Select the matching image rectangle without making another full-frame copy.
        """
        height, width = self.lines.shape
        return image[self.y : self.y + height, self.x : self.x + width]


def shape_masks(
    spots: SampleSpots,
    number: int,
    center: tuple[int, int],
    scale: int,
    shape: tuple[int, int],
) -> ShapeMasks:
    """
    Rasterize one clipped bounding rectangle and retain it until geometry changes.
    """
    region = spots.region(number)
    key = geometry_key(region, center), scale, shape, region["exclude"], region.get("fill", False)
    cached = spots.display_cache.masks.get(number)
    if cached is not None and cached[0] == key:
        spots.display_cache.masks.move_to_end(number)
        return cached[1]
    points = spots.display_cache.boundary(number, region, center, scale)
    padding = 7 if region["kind"] == "spot" and region["radius"] == 0 else 3
    left, top = np.maximum(points.min(axis=0) - padding, 0)
    right, bottom = np.minimum(points.max(axis=0) + padding + 1, (shape[1], shape[0]))
    x, y = min(int(left), shape[1]), min(int(top), shape[0])
    width, height = max(0, int(right) - x), max(0, int(bottom) - y)
    lines = np.zeros((height, width), np.uint8)
    filled = np.zeros_like(lines)
    if width and height:
        draw_boundary(lines, region, center, scale, points=points, offset=(x, y))
        if region.get("fill", False) and not (region["kind"] == "spot" and region["radius"] == 0):
            local_points = points - (x, y)
            # Keep the frozen full-canvas test reference independent of this raster path.
            # noinspection DuplicatedCode
            cv2.fillPoly(filled, [local_points], 255, lineType=cv2.LINE_AA)
            if region["exclude"]:
                perimeter = np.zeros_like(lines)
                cv2.polylines(perimeter, [local_points], True, 255, 1, cv2.LINE_AA)
                filled[perimeter > 0] = 0
        halo = cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0
        combined = cv2.max(filled, lines)
        coverage = cv2.dilate(combined, np.ones((3, 3), np.uint8))
    else:
        halo, combined, coverage = lines.astype(bool), lines, lines
    masks = ShapeMasks(x, y, lines, filled, combined, halo, coverage)
    spots.display_cache.store_masks(number, key, masks)
    return masks
