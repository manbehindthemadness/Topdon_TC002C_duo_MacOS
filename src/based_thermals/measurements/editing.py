"""
Captured point and label gestures with transient geometry until release.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .display_cache import geometry_key
from .overlays import display_boundary

if TYPE_CHECKING:
    from ..desktop_app.spots import SampleSpots


@dataclass(frozen=True)
class EditTarget:
    """
    Identify a stable measurement and one vertex, label or whole marker.
    """

    number: int
    kind: str
    index: int = 0


class MeasurementEdit:
    """
    Keep selection independent of mouse capture and preserve committed sampling while dragging.
    """

    def __init__(self) -> None:
        """
        Start without a selection, captured gesture or geometry preview.
        """
        self.selected: EditTarget | None = None
        self.captured = False
        self.draft: dict | None = None
        self.original: dict | None = None
        self.start = (0.0, 0.0)
        self.offset = (0.0, 0.0)
        self.moved = False
        self.swallow_release = False
        self.size = (256, 192)
        self.label_origin = (0.0, 0.0)
        self.pending_nudges: list[str] = []
        self.generation: int | None = None
        self.preview_key: tuple | None = None
        self.preview_outline: np.ndarray | None = None

    def cancel(self) -> None:
        """
        Drop uncommitted previews and remember to consume a cancelled release.
        """
        self.swallow_release |= self.captured
        self.selected = None
        self.captured = False
        self.draft = self.original = None
        self.pending_nudges.clear()

    def reconcile(self, spots: SampleSpots) -> None:
        """
        Invalidate selection after deletion, disabling or external geometry edits.
        """
        target = self.selected
        if target is not None and (
            target.number not in spots.numbers
            or (self.generation is not None and self.generation != spots.generation)
            or target.number in spots.disabled
            or (self.original is not None and spots.region(target.number) != self.original)
            or (
                target.kind == "vertex"
                and target.index >= len(spots.region(target.number)["points"])
            )
        ):
            self.cancel()

    def handles(self, spots: SampleSpots, number: int) -> list[int]:
        """
        Expose anchors first and only the selected Bezier anchor's adjacent controls.
        """
        region = (
            self.draft
            if self.draft is not None
            and self.selected is not None
            and self.selected.number == number
            else spots.region(number)
        )
        count = len(region["points"])
        if region["kind"] != "bezier":
            return list(range(count))
        indices = list(range(0, count, 3))
        target = self.selected
        if target is not None and target.number == number and target.kind == "vertex":
            anchor = (
                target.index
                if target.index % 3 == 0
                else (target.index - 1 if target.index % 3 == 1 else (target.index + 1) % count)
            )
            indices.extend(((anchor - 1) % count, (anchor + 1) % count))
        return indices

    def hit(
        self, spots: SampleSpots, position: tuple[int, int], scale: int, radius: float
    ) -> EditTarget | None:
        """
        Test visible point handles and cached label rectangles once on mouse press.
        """
        hits = []
        for number, _center in spots.active:
            region = spots.region(number)
            for index in self.handles(spots, number):
                point = region["points"][index]
                distance = sum(
                    (position[axis] - (point[axis] * scale + scale // 2)) ** 2 for axis in (0, 1)
                )
                if distance <= radius**2:
                    hits.append((-region.get("z_index", 0), distance, number, index))
        if hits:
            _, _, number, index = min(hits)
            return EditTarget(number, "vertex", index)
        for label in sorted(
            spots.display_cache.display_labels,
            key=lambda item: spots.region(item.number).get("z_index", 0),
            reverse=True,
        ):
            left, top, right, bottom = label.rectangle
            if (
                label.number in spots.numbers
                and label.number not in spots.disabled
                and bool(spots.region(label.number)["labels"])
                and left <= position[0] <= right
                and top <= position[1] <= bottom
            ):
                return EditTarget(label.number, "label")
        return None

    def pointer(
        self,
        event: int,
        position: tuple[int, int] | None,
        flags: int,
        spots: SampleSpots,
        scale: int,
        radius: float,
    ) -> bool:
        """
        Capture points/labels, preview movement and commit a validated release once.
        """
        self.reconcile(spots)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.swallow_release = False
            if position is None:
                return False
            target = self.hit(spots, position, scale, radius)
            if target is None:
                return False
            self.selected = target
            self.generation = spots.generation
            self.captured = True
            self.moved = False
            original = deepcopy(spots.region(target.number))
            self.original = original
            self.draft = deepcopy(original)
            self.start = tuple((coordinate - scale // 2) / scale for coordinate in position)
            if target.kind == "label":
                label = next(
                    label
                    for label in spots.display_cache.display_labels
                    if label.number == target.number
                )
                point = tuple((coordinate - scale // 2) / scale for coordinate in label.origin)
                self.label_origin = point
            else:
                point = original["points"][target.index]
            self.offset = tuple(point[axis] - self.start[axis] for axis in (0, 1))
            return True
        if event == cv2.EVENT_LBUTTONUP and self.swallow_release:
            self.swallow_release = False
            return True
        if not self.captured:
            return False
        if event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON:
            self.cancel()
            return True
        if position is not None:
            self._preview(position, scale)
        if event == cv2.EVENT_LBUTTONUP:
            if self.moved and self.selected is not None and self.draft is not None:
                spots.configure_region(self.selected.number, self.draft, self.size)
            self.captured = False
            self.original = self.draft = None
        return True

    def _preview(self, position: tuple[int, int], scale: int) -> None:
        """
        Move one handle, carrying both tangents with a Bezier anchor.
        """
        target = self.selected
        if target is None or self.original is None:
            return
        current = tuple((coordinate - scale // 2) / scale for coordinate in position)
        if (
            not self.moved
            and sum((current[axis] - self.start[axis]) ** 2 for axis in (0, 1)) < (2 / scale) ** 2
        ):
            return
        self.moved = True
        draft = deepcopy(self.original)
        point = [
            max(0.0, min(self.size[axis] - 1, current[axis] + self.offset[axis])) for axis in (0, 1)
        ]
        if target.kind == "label":
            draft["label_position"] = point
        else:
            move_vertex(draft, target.index, point, self.size)
        self.draft = draft

    def nudge(self, spots: SampleSpots, direction: object) -> bool:
        """
        Move the selected target by one oriented native pixel without changing its identity.
        """
        self.reconcile(spots)
        target = self.selected
        delta = (
            {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.get(direction)
            if isinstance(direction, str)
            else None
        )
        if target is None or delta is None or self.captured:
            return False
        if target.kind == "shape":
            center = spots.pixels[spots.numbers.index(target.number)]
            pixel = tuple(
                max(0, min(self.size[axis] - 1, center[axis] + delta[axis])) for axis in (0, 1)
            )
            spots.move(target.number, pixel, size=self.size)
        else:
            region = deepcopy(spots.region(target.number))
            if target.kind == "label":
                label = next(
                    (
                        label
                        for label in spots.display_cache.display_labels
                        if label.number == target.number
                    ),
                    None,
                )
                if label is None:
                    return False
                point = region.get("label_position") or self.label_origin
            else:
                point = region["points"][target.index]
            point = [max(0, min(self.size[axis] - 1, point[axis] + delta[axis])) for axis in (0, 1)]
            if target.kind == "label":
                region["label_position"] = point
            else:
                move_vertex(region, target.index, point, self.size)
            spots.configure_region(target.number, region, self.size)
        return True

    def draw(self, image: np.ndarray, spots: SampleSpots, scale: int) -> np.ndarray:
        """
        Draw selected handles and a single preview outline without rasterizing sampling masks.
        """
        self.reconcile(spots)
        target = self.selected
        if target is None:
            return image
        if target.kind == "label":
            label = next(
                (
                    label
                    for label in spots.display_cache.display_labels
                    if label.number == target.number
                ),
                None,
            )
            if label is not None:
                left, top, right, bottom = label.rectangle
                cv2.rectangle(image, (left, top), (right, bottom), (0, 220, 255), 1, cv2.LINE_AA)
            return image
        region = self.draft or spots.region(target.number)
        center = spots.pixels[spots.numbers.index(target.number)]
        if self.draft is not None and self.moved:
            key = geometry_key(region, center), scale
            if key != self.preview_key:
                self.preview_outline = display_boundary(region, center, scale)
                self.preview_key = key
            assert self.preview_outline is not None
            cv2.polylines(image, [self.preview_outline], True, (0, 220, 255), 1, cv2.LINE_AA)
        points = region["points"]
        if not points:
            point = tuple(value * scale + scale // 2 for value in center)
            cv2.circle(image, point, 6, (0, 220, 255), 1, cv2.LINE_AA)
        for index in self.handles(spots, target.number):
            point = tuple(round(value * scale) + scale // 2 for value in points[index])
            is_control = region["kind"] == "bezier" and index % 3 != 0
            if is_control:
                anchor_index = index - 1 if index % 3 == 1 else (index + 1) % len(points)
                anchor = tuple(round(value * scale) + scale // 2 for value in points[anchor_index])
                cv2.line(image, anchor, point, (0, 220, 255), 1, cv2.LINE_AA)
            selected = target.kind == "vertex" and target.index == index
            cv2.circle(image, point, 4 if selected else 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(
                image, point, 4 if selected else 3, (0, 220, 255), 2 if selected else 1, cv2.LINE_AA
            )
        return image


def move_vertex(region: dict, index: int, point: list[float], size: tuple[int, int]) -> None:
    """
    Move one native point and preserve adjacent Bezier tangents inside the sensor bounds.
    """
    indices = [index]
    if region["kind"] == "bezier" and index % 3 == 0:
        indices.extend(((index - 1) % len(region["points"]), (index + 1) % len(region["points"])))
    delta = [point[axis] - region["points"][index][axis] for axis in (0, 1)]
    for axis in (0, 1):
        delta[axis] = max(
            -min(region["points"][i][axis] for i in indices),
            min(delta[axis], size[axis] - 1 - max(region["points"][i][axis] for i in indices)),
        )
    for i in indices:
        region["points"][i] = [round(region["points"][i][axis] + delta[axis], 4) for axis in (0, 1)]
