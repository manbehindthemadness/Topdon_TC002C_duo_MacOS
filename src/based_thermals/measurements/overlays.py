"""
Area boundaries and configurable edge-attached labels using the shared label placer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..desktop_app.typography import LABEL_FONT_SCALE
from .display_cache import geometry_key
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
    number: int = 0


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
    readings = spots.display_cache.readings(spots, rendered.temperatures_celsius)
    spots.display_cache.prune(spots.numbers)
    outlines: dict[int, np.ndarray] = {}
    occupied: list[Rectangle] = []
    for number, center in spots.active:
        region = spots.region(number)
        points = spots.display_cache.boundary(number, region, center, scale)
        outlines[number] = points
        if region["kind"] == "spot" and region["radius"] == 0:
            x, y = points[0]
            occupied.append((int(x - 6), int(y - 6), int(x + 7), int(y + 7)))
        else:
            left, top = points.min(axis=0)
            right, bottom = points.max(axis=0)
            occupied.append((int(left - 3), int(top - 3), int(right + 4), int(bottom + 4)))
    texts: list[tuple[int, str]] = []
    for number, _center in spots.active:
        region = spots.region(number)
        fields = region["labels"]
        if not fields:
            continue
        if region["exclude"]:
            if not {"number", "name"}.intersection(fields):
                continue
            text = "Exclusion"
            if "number" in fields:
                text += f" {number}"
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
        texts.append((number, text))
    key = (
        rendered.image.shape,
        scale,
        unit,
        placer,
        LABEL_FONT_SCALE,
        tuple(
            (number, geometry_key(spots.region(number), center)) for number, center in spots.active
        ),
        tuple(
            (number, spots.display_cache.text_bounds(text, LABEL_FONT_SCALE))
            for number, text in texts
        ),
    )
    if key == spots.display_cache.layout_key:
        assert spots.display_cache.layout is not None
        old_labels, old_occupied = spots.display_cache.layout
        updated = [
            replace(label, text=text)
            for label, (_number, text) in zip(old_labels, texts, strict=True)
        ]
        return positioned_labels(updated, list(old_occupied), spots, rendered.image.shape, scale)
    labels: list[MeasurementLabel] = []
    for number, text in texts:
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
        labels.append(MeasurementLabel(text, anchor, origin, rectangle, number))
        occupied.append(rectangle)
    spots.display_cache.layout_key = key
    spots.display_cache.layout = tuple(labels), tuple(occupied)
    spots.display_cache.layout_generation += 1
    return positioned_labels(labels, occupied, spots, rendered.image.shape, scale)


def positioned_labels(
    labels: list[MeasurementLabel],
    occupied: list[Rectangle],
    spots: SampleSpots,
    shape: tuple[int, ...],
    scale: int,
) -> tuple[list[MeasurementLabel], list[Rectangle]]:
    """
    Override only manually positioned labels without invalidating the automatic layout cache.
    """
    result = []
    shape_bounds = occupied[: len(occupied) - len(labels)]
    for label in labels:
        position = spots.display_cache.label_preview.get(
            label.number, spots.region(label.number).get("label_position")
        )
        if position is not None:
            key = spots.display_cache.layout_generation, tuple(position), label.rectangle, scale, shape
            cached = spots.display_cache.positioned_labels.get(label.number)
            if cached is not None and cached[0] == key:
                result.append(replace(cached[1], text=label.text))
                continue
            width = label.rectangle[2] - label.rectangle[0]
            above = label.origin[1] - label.rectangle[1]
            below = label.rectangle[3] - label.origin[1]
            x = max(0, min(max(0, shape[1] - width), round(position[0] * scale) + scale // 2))
            y = max(above, min(shape[0] - below, round(position[1] * scale) + scale // 2))
            rectangle = x, y - above, x + width, y + below
            center = spots.pixels[spots.numbers.index(label.number)]
            points = spots.display_cache.boundary(
                label.number, spots.region(label.number), center, scale
            )
            if len(points) == 4:
                points = np.concatenate(
                    [
                        np.linspace(points[i], points[(i + 1) % len(points)], 32)
                        for i in range(len(points))
                    ]
                )
            midpoint = np.array((x + width / 2, y + (below - above) / 2))
            nearest = points[np.argmin(np.sum((points - midpoint) ** 2, axis=1))]
            label = replace(
                label, origin=(x, y), rectangle=rectangle, anchor=(int(nearest[0]), int(nearest[1]))
            )
            spots.display_cache.positioned_labels[label.number] = key, label
        else:
            spots.display_cache.positioned_labels.pop(label.number, None)
        result.append(label)
    spots.display_cache.display_labels = result
    rectangles = shape_bounds + [label.rectangle for label in result]
    return result, rectangles


def draw_boundaries(mask: np.ndarray, spots: SampleSpots, scale: int) -> None:
    """
    Union measurement outlines without touching radiometry or inverting overlaps twice.
    """
    for number, center in spots.active:
        draw_boundary(mask, spots.region(number), center, scale)


def draw_boundary(
    mask: np.ndarray,
    region: dict,
    center: tuple[int, int],
    scale: int,
    *,
    points: np.ndarray | None = None,
    offset: tuple[int, int] = (0, 0),
) -> None:
    """
    Draw one measurement outline or a dashed exclusion border in display coordinates.
    """
    if region["kind"] == "spot" and region["radius"] == 0:
        cv2.drawMarker(
            mask,
            (
                center[0] * scale + scale // 2 - offset[0],
                center[1] * scale + scale // 2 - offset[1],
            ),
            255,
            cv2.MARKER_CROSS,
            9,
            1,
            cv2.LINE_8,
        )
        return
    points = (display_boundary(region, center, scale) if points is None else points) - offset
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
