"""
Pure-data measurement validation shared with the Qt configuration window.
"""

from __future__ import annotations

import math
import re
from typing import Any

KINDS = ("spot", "rectangle", "oval", "polygon", "bezier")
LABEL_FIELDS = ("number", "name", "min", "max", "average")


def default_region(kind: str = "spot") -> dict:
    """
    Return fresh geometry and label defaults for a point or measurement area.
    """
    return {
        "kind": kind,
        "radius": 0.0,
        "points": [],
        "exclude": False,
        "line_color": "dynamic",
        "fill": False,
        "opacity": 1.0,
        "z_index": 0,
        "labels": ["number", "average"] if kind == "spot" else ["number", "min", "max", "average"],
    }


def validate_region(value: object, native_size: tuple[int, int]) -> dict:
    """
    Validate finite, bounded native coordinates and closed cubic Bezier topology.

    Bezier points are anchor/control/control triples. The next triple supplies
    the endpoint, and the final segment wraps to the first anchor.
    """
    appearance = {"line_color", "fill", "opacity", "z_index"}
    required = set(default_region()) - appearance
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or not set(value) <= set(default_region())
    ):
        raise ValueError("A measurement requires kind, radius, points, exclude and labels")
    line_color = value.get("line_color", "dynamic")
    if not isinstance(line_color, str) or (
        line_color != "dynamic" and re.fullmatch(r"#[0-9a-fA-F]{6}", line_color) is None
    ):
        raise ValueError("Line color must be dynamic or an RGB hex color (#RRGGBB)")
    fill = value.get("fill", False)
    if type(fill) is not bool:
        raise TypeError("Fill state must be a boolean")
    opacity = value.get("opacity", 1.0)
    z_index = value.get("z_index", 0)
    if type(z_index) is not int or not -1000 <= z_index <= 1000:
        raise ValueError("Measurement Z index must be an integer between -1000 and 1000")
    if (
        isinstance(opacity, bool)
        or not isinstance(opacity, (int, float))
        or not math.isfinite(opacity)
        or not 0 <= opacity <= 1
    ):
        raise ValueError("Measurement opacity must be between 0 and 1")
    kind, radius = value.get("kind"), value.get("radius")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("Choose a spot, rectangle, oval, polygon or Bezier area")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(radius)
        or not 0 <= radius <= max(native_size)
    ):
        raise ValueError("Spot radius must be a finite number of native pixels")
    if type(value.get("exclude")) is not bool:
        raise TypeError("Exclusion state must be a boolean")
    labels = value.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) != len({str(item) for item in labels})
        or any(not isinstance(item, str) or item not in LABEL_FIELDS for item in labels)
    ):
        raise ValueError("Choose unique measurement label fields, or none to hide the label")
    points = value.get("points")
    if not isinstance(points, list) or len(points) > 192:
        raise ValueError("Measurement geometry needs at most 192 points")
    counts = {
        "spot": len(points) == 0,
        "rectangle": len(points) == 2,
        "oval": len(points) == 2,
        "polygon": len(points) >= 3,
        "bezier": len(points) >= 6 and len(points) % 3 == 0,
    }
    if not counts[kind]:
        raise ValueError(
            "Rectangles/ovals need two corners; polygons three vertices; Beziers anchor/control/control triples"
        )
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Every geometry point must contain x and y")
        for coordinate, bound in zip(point, native_size, strict=True):
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(coordinate)
                or not 0 <= coordinate <= bound - 1
            ):
                raise ValueError("Geometry points must lie inside the native sensor grid")
        normalized.append([round(float(coordinate), 4) for coordinate in point])
    region = {
        "kind": kind,
        "radius": float(radius),
        "points": normalized,
        "exclude": value["exclude"],
        "labels": list(labels),
        "line_color": line_color.lower(),
        "fill": fill,
        "opacity": float(opacity),
        "z_index": z_index,
    }
    return region


def validate_center(value: object, size: tuple[int, int]) -> tuple[int, int]:
    """
    Validate a native pixel anchor separately from subpixel curve handles.
    """
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            type(n) is not int or not 0 <= n < bound for n, bound in zip(value, size, strict=True)
        )
    ):
        raise ValueError("Measurement center must lie inside the native sensor grid")
    return value[0], value[1]


def transform_region(region: dict, transform: Any) -> dict:
    """
    Transform all vertices and Bezier handles without changing radius or metadata.
    """
    transformed = {
        **region,
        "points": [[round(float(v), 4) for v in transform(x, y)] for x, y in region["points"]],
    }
    return transformed
