"""
Area boundaries and configurable edge-attached labels using the shared label placer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .paths import boundary

if TYPE_CHECKING:
    from ..desktop_app.spots import SampleSpots
    from ..render import RenderedThermalFrame

Rectangle = tuple[int, int, int, int]
LabelPlacer = Callable[
    [str, tuple[int, int], tuple[int, ...], list[Rectangle]],
    tuple[tuple[int, int], Rectangle],
]


@dataclass(frozen=True)
class MeasurementLabel:
    """
    Positioned text and its boundary attachment point in display coordinates.
    """

    text: str
    anchor: tuple[int, int]
    origin: tuple[int, int]
    rectangle: tuple[int, int, int, int]


def display_boundary(region: dict, center: tuple[int, int], scale: int) -> np.ndarray:
    """
    Convert native geometry to the renderer's scaled, oriented image.
    """
    points = np.rint(np.asarray(boundary(region, center)) * scale + scale // 2).astype(np.int32)
    return points


def label_layout(
    rendered: RenderedThermalFrame,
    spots: SampleSpots,
    scale: int,
    unit: str,
    placer: LabelPlacer,
) -> tuple[list[MeasurementLabel], list[tuple[int, int, int, int]]]:
    """
    Reserve points/shapes first, then place each enabled label outside occupied bounds.
    """
    readings = spots.readings(rendered.temperatures_celsius)
    outlines: dict[int, np.ndarray] = {}
    occupied: list[Rectangle] = []
    for number, center in spots.active:
        region = spots.region(number)
        points = display_boundary(region, center, scale)
        outlines[number] = points
        if region["kind"] == "spot" and region["radius"] == 0:
            x, y = points[0]
            occupied.append((int(x - 6), int(y - 6), int(x + 7), int(y + 7)))
        else:
            left, top = points.min(axis=0)
            right, bottom = points.max(axis=0)
            occupied.append((int(left - 3), int(top - 3), int(right + 4), int(bottom + 4)))
    labels: list[MeasurementLabel] = []
    for number, _center in spots.active:
        region = spots.region(number)
        fields = region["labels"]
        if not fields:
            continue
        if region["exclude"]:
            text = f"Exclusion {number}"
            if "name" in fields:
                text += f": {spots.name(number)}"
        else:
            reading = readings[number]
            parts = []
            if "number" in fields:
                parts.append(str(number) + ":")
            if "name" in fields:
                parts.append(spots.name(number))
            for field, value in (
                ("min", reading.minimum),
                ("max", reading.maximum),
                ("average", reading.average),
            ):
                if field not in fields:
                    continue
                displayed = float(value * 1.8 + 32 if unit == "F" else value)
                formatted = f"{displayed:.2f}" if np.isfinite(displayed) else "--"
                prefix = (
                    ""
                    if fields == ["number", "average"]
                    else {"min": "Min ", "max": "Max ", "average": "Avg "}[field]
                )
                parts.append(prefix + formatted + " " + unit)
            text = " ".join(parts)
        points = outlines[number]
        # Start above the outside edge. After placement, attach the leader to
        # whichever boundary point is nearest its text rectangle.
        highest = points[np.argmin(points[:, 1])]
        anchor = int(highest[0]), int(highest[1])
        origin, rectangle = placer(text, anchor, rendered.image.shape, occupied)
        midpoint = np.asarray(
            ((rectangle[0] + rectangle[2]) / 2, (rectangle[1] + rectangle[3]) / 2)
        )
        if len(points) == 4:
            # Rectangles need points along their edges, not just the four corners.
            candidates = np.concatenate(
                [
                    np.linspace(points[i], points[(i + 1) % len(points)], 32)
                    for i in range(len(points))
                ]
            )
        else:
            candidates = points
        nearest = candidates[np.argmin(np.sum((candidates - midpoint) ** 2, axis=1))]
        anchor = round(float(nearest[0])), round(float(nearest[1]))
        labels.append(MeasurementLabel(text, anchor, origin, rectangle))
        occupied.append(rectangle)
    return labels, occupied


def draw_boundaries(mask: np.ndarray, spots: SampleSpots, scale: int) -> None:
    """
    Union measurement outlines without touching radiometry or inverting overlaps twice.
    """
    for number, center in spots.active:
        draw_boundary(mask, spots.region(number), center, scale)


def draw_boundary(mask: np.ndarray, region: dict, center: tuple[int, int], scale: int) -> None:
    """
    Draw one measurement outline or a dashed exclusion border in display coordinates.
    """
    if region["kind"] == "spot" and region["radius"] == 0:
        cv2.drawMarker(
            mask, (center[0] * scale + scale // 2, center[1] * scale + scale // 2),
            255, cv2.MARKER_CROSS, 9, 1, cv2.LINE_8,
        )
        return
    points = display_boundary(region, center, scale)
    if region["exclude"]:
        draw_dashed_boundary(mask, points)
    else:
        cv2.polylines(mask, [points], True, 255, 1, cv2.LINE_AA)


def draw_dashed_boundary(mask: np.ndarray, points: np.ndarray) -> None:
    """
    Carry six-pixel dash/gap phases across vertices, including short curve edges.
    """
    phase: float = 0.0
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        delta = end - start
        length = float(np.linalg.norm(delta))
        distance: float = 0.0
        while distance < length - 1e-8:
            step: float = min(6 - phase % 6, length - distance)
            if phase < 6:
                first = np.rint(start + delta * distance / length).astype(int)
                last = np.rint(start + delta * (distance + step) / length).astype(int)
                cv2.line(
                    mask,
                    (int(first[0]), int(first[1])),
                    (int(last[0]), int(last[1])),
                    255,
                    1,
                    cv2.LINE_AA,
                )
            distance += step
            phase = (phase + step) % 12
