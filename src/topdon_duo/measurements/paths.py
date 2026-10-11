"""
Pure geometry paths for raster sampling and Qt editing previews.
"""

from __future__ import annotations

import math


def boundary(region: dict, center: tuple[int, int], steps: int = 48) -> list[tuple[float, float]]:
    """
    Tessellate the outside boundary in native pixel coordinates.
    """
    kind, points = region["kind"], region["points"]
    if kind == "spot":
        radius = region["radius"]
        return [
            (
                center[0] + radius * math.cos(t * math.tau / steps),
                center[1] + radius * math.sin(t * math.tau / steps),
            )
            for t in range(steps)
        ]
    if kind in ("rectangle", "oval"):
        (x0, y0), (x1, y1) = [[float(v) for v in point] for point in points]
        left, right, top, bottom = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        if kind == "rectangle":
            return [(left, top), (right, top), (right, bottom), (left, bottom)]
        return [
            (
                (left + right) / 2 + (right - left) / 2 * math.cos(t * math.tau / steps),
                (top + bottom) / 2 + (bottom - top) / 2 * math.sin(t * math.tau / steps),
            )
            for t in range(steps)
        ]
    if kind == "polygon":
        return [(float(point[0]), float(point[1])) for point in points]
    result = []
    for index in range(0, len(points), 3):
        controls = points[index : index + 3] + [points[(index + 3) % len(points)]]
        for step in range(steps):
            t = step / steps
            weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3)
            result.append(
                tuple(
                    sum(p[axis] * w for p, w in zip(controls, weights, strict=True))
                    for axis in (0, 1)
                )
            )
    return result


def initial_region(kind: str, size: tuple[int, int]) -> tuple[tuple[int, int], dict]:
    """
    Create an editable centered shape, including a four-segment cubic oval.
    """
    from .schema import default_region

    width, height = size
    cx, cy = width // 2, height // 2
    rx = min(max(1, width // 6), cx, width - 1 - cx)
    ry = min(max(1, height // 6), cy, height - 1 - cy)
    region = default_region(kind)
    if kind in ("rectangle", "oval"):
        region["points"] = [[cx - rx, cy - ry], [cx + rx, cy + ry]]
    elif kind == "polygon":
        region["points"] = [[cx - rx, cy + ry], [cx, cy - ry], [cx + rx, cy + ry]]
    elif kind == "bezier":
        k = 0.5522847498
        region["points"] = [
            [cx + rx, cy],
            [cx + rx, cy + ry * k],
            [cx + rx * k, cy + ry],
            [cx, cy + ry],
            [cx - rx * k, cy + ry],
            [cx - rx, cy + ry * k],
            [cx - rx, cy],
            [cx - rx, cy - ry * k],
            [cx - rx * k, cy - ry],
            [cx, cy - ry],
            [cx + rx * k, cy - ry],
            [cx + rx, cy - ry * k],
        ]
    return (cx, cy), region
