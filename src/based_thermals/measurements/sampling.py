"""
Native-temperature kernels and exclusion subtraction; display pixels are never sampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paths import boundary


@dataclass(frozen=True)
class AreaReading:
    """
    Native Celsius statistics, with NaNs for an empty or unavailable kernel.
    """

    minimum: float
    maximum: float
    average: float
    count: int

    def graph_values(self) -> tuple[float, float, float]:
        """
        Return the conventional graph ordering of minimum, average, maximum.
        """
        return self.minimum, self.average, self.maximum


def kernel(region: dict, center: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """
    Rasterize sensor pixel centers, including boundary pixels and clipping to the grid.
    """
    y, x = np.ogrid[: shape[0], : shape[1]]
    kind = region["kind"]
    if kind == "spot":
        return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= region["radius"] ** 2
    if kind in ("rectangle", "oval"):
        (x0, y0), (x1, y1) = [[float(v) for v in point] for point in region["points"]]
        left, right, top, bottom = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        inside = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
        if kind == "rectangle":
            return inside
        rx, ry = (right - left) / 2, (bottom - top) / 2
        return inside & (
            ((x - (left + right) / 2) / (rx or 1)) ** 2
            + ((y - (top + bottom) / 2) / (ry or 1)) ** 2
            <= 1
        )
    return polygon_kernel(boundary(region, center), shape)


def polygon_kernel(points: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    """
    Include pixel centers using even-odd containment and exact boundary membership.

    Restrict the work to the shape bounds. Unlike display line rasterization, this
    rule preserves the same sensor samples through rotations and reflections.
    """
    mask = np.zeros(shape, bool)
    vertices = np.asarray(points, dtype=float)
    left, top = np.maximum(np.ceil(vertices.min(axis=0)).astype(int), 0)
    right, bottom = np.minimum(
        np.floor(vertices.max(axis=0)).astype(int), (shape[1] - 1, shape[0] - 1)
    )
    if left > right or top > bottom:
        return mask
    y, x = np.ogrid[top : bottom + 1, left : right + 1]
    inside: np.ndarray = np.zeros((bottom - top + 1, right - left + 1), bool)
    edge: np.ndarray = inside.copy()
    for index, point in enumerate(points):
        x0, y0 = float(point[0]), float(point[1])
        following = points[(index + 1) % len(points)]
        x1, y1 = float(following[0]), float(following[1])
        dx, dy = x1 - x0, y1 - y0
        cross = (x - x0) * dy - (y - y0) * dx
        edge |= (
            (np.abs(cross) <= 1e-8)
            & (x >= min(x0, x1) - 1e-8)
            & (x <= max(x0, x1) + 1e-8)
            & (y >= min(y0, y1) - 1e-8)
            & (y <= max(y0, y1) + 1e-8)
        )
        if dy:
            inside ^= ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * dx / dy)
    mask[top : bottom + 1, left : right + 1] = inside | edge
    return mask


def sample(temperatures: np.ndarray, mask: np.ndarray, excluded: np.ndarray) -> AreaReading:
    """
    Reduce finite native Celsius values after subtracting every enabled exclusion.
    """
    values = temperatures[mask & ~excluded & np.isfinite(temperatures)]
    if not values.size:
        return AreaReading(float("nan"), float("nan"), float("nan"), 0)
    reading = AreaReading(
        float(values.min()), float(values.max()), float(values.mean()), int(values.size)
    )
    return reading
