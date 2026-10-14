"""
Measuring-spot state, pointer events, and drag interaction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..camera_backends.duo.camera import (
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
)
from ..measurements.model import MeasurementRegions
from ..measurements.schema import transform_region
from ..spot_preferences import validate_spots
from .layout import image_position_at


@dataclass
class MousePicker:
    x: int | None = None
    y: int | None = None
    clicks: list[tuple[int, int]] = field(default_factory=list)
    context_clicks: list[tuple[int, int]] = field(default_factory=list)
    drag_events: list[tuple[int, int, int, int]] = field(default_factory=list)

    def callback(self, event: int, x: int, y: int, flags: int, _parameter: Any) -> None:
        """
        Callback.
        """
        if (
            sys.platform == "darwin"
            and flags & cv2.EVENT_FLAG_CTRLKEY
            and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP)
        ):
            # Cocoa may report Control-click as a left click rather than a
            # secondary click. Do not let it place or drag a measuring spot.
            if event == cv2.EVENT_LBUTTONUP:
                self.context_clicks.append((x, y))
            return
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            self.drag_events.append((event, x, y, flags))
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONUP:
            self.x, self.y = x, y
            self.clicks.append((x, y))

        if event == cv2.EVENT_RBUTTONUP:
            self.context_clicks.append((x, y))

    def consume_drag_events(self) -> Any:
        """
        Consume drag events.
        """
        events, self.drag_events = self.drag_events, []
        return events

    def discard_click(self, x: Any, y: Any) -> None:
        """
        Discard click.
        """
        if (x, y) in self.clicks:
            self.clicks.remove((x, y))

    def consume_context_clicks(self) -> Any:
        """
        Consume context clicks.
        """
        clicks, self.context_clicks = self.context_clicks, []
        return clicks

    def consume_clicks(self) -> list[tuple[int, int]]:
        """
        Consume clicks.
        """
        clicks, self.clicks = self.clicks, []
        return clicks


@dataclass
class SampleSpots(MeasurementRegions):
    placing: bool = False
    pixels: list[tuple[int, int]] = field(default_factory=list)
    generation: int = 0
    native_size: tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT)

    numbers: list[int] = field(default_factory=list, init=False)
    disabled: set[int] = field(default_factory=set, init=False)
    next_number: int = field(default=1, init=False)
    names: dict[int, str] = field(default_factory=dict, init=False)
    regions: dict[int, dict] = field(default_factory=dict, init=False)
    region_masks: dict[int, tuple[tuple, np.ndarray]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """
        Post init.
        """
        self.numbers = list(range(1, len(self.pixels) + 1))
        self.next_number = len(self.pixels) + 1

    @property
    def active(self) -> list[tuple[int, tuple[int, int]]]:
        """
        Return enabled measurement identities and their native pixel anchors.
        """
        return [
            (number, pixel)
            for number, pixel in zip(self.numbers, self.pixels, strict=True)
            if number not in self.disabled
        ]

    def toggle(self) -> None:
        """
        Toggle.
        """
        self.placing = not self.placing

    def add(self, pixel: tuple[int, int] | None) -> None:
        """
        Add.
        """
        if self.placing and pixel is not None and pixel not in self.pixels:
            self.pixels.append(pixel)
            self.numbers.append(self.next_number)
            self.next_number += 1

    def set_enabled(self, number: Any, enabled: Any) -> None:
        """
        Set enabled.
        """
        if type(number) is not int or number not in self.numbers or type(enabled) is not bool:
            raise ValueError("Choose an existing spot and an enabled state")
        if enabled:
            self.disabled.discard(number)
        else:
            self.disabled.add(number)

    def move(self, number: Any, pixel: Any, size: tuple[int, int] | None = None) -> None:
        """
        Translate the anchor and all area handles, clamping the complete shape.
        """
        if type(number) is not int or number not in self.numbers:
            raise ValueError("Choose an existing spot to move")
        index = self.numbers.index(number)
        previous = self.pixels[index]
        if number in self.regions:
            dx, dy = pixel[0] - previous[0], pixel[1] - previous[1]
            points = self.regions[number]["points"]
            if points:
                width, height = size or self.native_size
                dx = max(-min(p[0] for p in points), min(dx, width - 1 - max(p[0] for p in points)))
                dy = max(-min(p[1] for p in points), min(dy, height - 1 - max(p[1] for p in points)))
                dx = max(-previous[0], min(dx, width - 1 - previous[0]))
                dy = max(-previous[1], min(dy, height - 1 - previous[1]))
                pixel = (round(previous[0] + dx), round(previous[1] + dy))
            self.regions[number] = transform_region(
                self.regions[number], lambda x, y: (x + dx, y + dy)
            )
        self.pixels[index] = pixel

    def name(self, number: int) -> str:
        """
        Return a saved name or a shape-specific default for its stable number.
        """
        kind = self.region(number)["kind"].title()
        return self.names.get(number, f"{kind} {number}")

    def rename(self, number: Any, name: Any) -> None:
        """
        Rename.
        """
        if type(number) is not int or number not in self.numbers:
            raise ValueError("Choose an existing spot to name")
        if (
            not isinstance(name, str)
            or len(name) > 64
            or any(not char.isprintable() for char in name)
        ):
            raise ValueError("Region names must be a single line of up to 64 characters")
        name = name.strip()
        if name and name != f"Spot {number}":
            self.names[number] = name
        else:
            self.names.pop(number, None)

    def clear(self, number: int | None = None) -> None:
        """
        Clear.
        """
        if number is None:
            self.pixels.clear()
            self.numbers.clear()
            self.disabled.clear()
            self.names.clear()
            self.regions.clear()
            self.region_masks.clear()
            self.next_number = 1
            self.generation += 1
            return
        if type(number) is not int or number not in self.numbers:
            raise ValueError("Choose an existing spot to clear")
        index = self.numbers.index(number)
        self.pixels.pop(index)
        self.numbers.pop(index)
        self.disabled.discard(number)
        self.names.pop(number, None)
        self.regions.pop(number, None)
        self.region_masks.pop(number, None)

    def state(self) -> Any:
        """
        State.
        """
        return [
            {"number": number, "enabled": number not in self.disabled, "name": self.name(number)}
            for number in self.numbers
        ]

    def saved_state(self, rotation: Any = 0, horizontal: Any = False, vertical: Any = False) -> Any:
        """
        Saved state.
        """
        width, height = (
            self.native_size if rotation in (0, 180) else self.native_size[::-1]
        )
        pixels = [
            (width - 1 - x if horizontal else x, height - 1 - y if vertical else y)
            for x, y in self.pixels
        ]
        for _ in range((360 - rotation) % 360 // 90):
            pixels = [(height - 1 - y, x) for x, y in pixels]
            width, height = height, width
        region_width, region_height = (
            self.native_size if rotation in (0, 180) else self.native_size[::-1]
        )
        # Use the original oriented bounds; the anchor loop above has already unrotated them.
        regions = {number: transform_region(
            region, lambda x, y: (region_width - 1 - x if horizontal else x,
                                  region_height - 1 - y if vertical else y)
        ) for number, region in self.regions.items()}
        for _ in range((360 - rotation) % 360 // 90):
            regions = {number: transform_region(region, lambda x, y, h=region_height: (h - 1 - y, x))
                       for number, region in regions.items()}
            region_width, region_height = region_height, region_width
        return {
            "version": 2 if self.regions else 1,
            "placing": self.placing,
            "next_number": self.next_number,
            "items": [
                {
                    "number": number,
                    "x": x,
                    "y": y,
                    "name": self.name(number),
                    "enabled": number not in self.disabled,
                    **({"region": regions.get(number, self.region(number))} if self.regions else {}),
                }
                for number, (x, y) in zip(self.numbers, pixels, strict=True)
            ],
        }

    def restore_saved_state(
        self, saved: Any, rotation: Any = 0, horizontal: Any = False, vertical: Any = False
    ) -> None:
        """
        Restore saved state.
        """
        saved = validate_spots(saved, native_size=self.native_size)
        self.region_masks.clear()
        self.pixels = [(item["x"], item["y"]) for item in saved["items"]]
        self.numbers = [item["number"] for item in saved["items"]]
        self.names = {item["number"]: item["name"] for item in saved["items"] if item["name"]}
        self.regions = {item["number"]: item["region"] for item in saved["items"] if "region" in item}
        self.disabled = {item["number"] for item in saved["items"] if not item["enabled"]}
        self.next_number, self.placing = saved["next_number"], saved["placing"]
        width, height = self.native_size
        for _ in range(rotation // 90):
            self.rotate_clockwise(height)
            width, height = height, width
        self.mirror(width, height, horizontal, vertical)

    def rotate_clockwise(self, sensor_height: int) -> None:
        """
        Rotate clockwise.
        """
        self.pixels = [(sensor_height - 1 - y, x) for x, y in self.pixels]
        self.transform_regions(lambda x, y: (sensor_height - 1 - y, x))

    def mirror(self, width: int, height: int, horizontal: bool, vertical: bool) -> None:
        """
        Mirror.
        """
        self.pixels = [
            (width - 1 - x if horizontal else x, height - 1 - y if vertical else y)
            for x, y in self.pixels
        ]
        self.transform_regions(lambda x, y: (width - 1 - x if horizontal else x,
                                             height - 1 - y if vertical else y))


def spot_hit(
    position: Any,
    spots: Any,
    image_shape: Any,
    scale: Any,
    viewport_size: tuple[int, int] | None = None,
) -> tuple[float, int, tuple[int, int]] | None:
    """
    Prefer the highest Z layer, then the closest enabled anchor within that layer.
    """
    radius = 10 * image_shape[1] / viewport_size[0] if viewport_size else 10
    hits = []
    for number, (sx, sy) in spots.active:
        anchor = (sx * scale + scale // 2, sy * scale + scale // 2)
        distance = (anchor[0] - position[0]) ** 2 + (anchor[1] - position[1]) ** 2
        region = spots.region(number)
        inside_area = False
        if region["kind"] != "spot" or region["radius"] > 0:
            sensor_x, sensor_y = position[0] // scale, position[1] // scale
            shape = (image_shape[0] // scale, image_shape[1] // scale)
            if 0 <= sensor_x < shape[1] and 0 <= sensor_y < shape[0]:
                inside_area = bool(spots.measurement_mask(number, (sx, sy), shape)[sensor_y, sensor_x])
        if inside_area or (region["kind"] == "spot" and distance <= radius**2):
            hits.append((distance, number, anchor))
    hit = (
        min(hits, key=lambda item: (-spots.region(item[1]).get("z_index", 0), item[0], item[1]))
        if hits else None
    )
    return hit


class SpotDrag:
    """Move a marker while retaining its identity and suppressing release clicks."""

    def __init__(self) -> None:
        """
        Init.
        """
        self.number = None
        self.captured = False
        self.offset = (0, 0)

    def cancel(self) -> None:
        """
        Cancel.
        """
        self.number = None
        # A cancelled drag's release must still not place a new spot or click a button.

    def update(
        self,
        picker: Any,
        spots: Any,
        image_shape: Any,
        scale: Any,
        viewport_size: tuple[int, int] | None = None,
        toolbar_height: Any = 0,
        locked: Any = False,
    ) -> None:
        """
        Update.
        """
        if locked:
            self.cancel()
        for event, x, y, flags in picker.consume_drag_events():
            position = image_position_at(x, y, image_shape, viewport_size, toolbar_height)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.number = None
                self.captured = False
                if locked or position is None:
                    continue
                hit = spot_hit(position, spots, image_shape, scale, viewport_size)
                if hit is not None:
                    _distance, self.number, anchor = hit
                    self.offset = (anchor[0] - position[0], anchor[1] - position[1])
                    self.captured = True
            elif event == cv2.EVENT_MOUSEMOVE and self.captured:
                if not flags & cv2.EVENT_FLAG_LBUTTON:
                    self.number = None
                    self.captured = False
                elif not locked:
                    self._move(spots, position, image_shape, scale)
            elif event == cv2.EVENT_LBUTTONUP:
                if self.captured:
                    picker.discard_click(x, y)
                    if not locked:
                        self._move(spots, position, image_shape, scale)
                self.number = None
                self.captured = False

    def _move(self, spots: Any, position: Any, image_shape: Any, scale: Any) -> None:
        """
        Move.
        """
        if position is None or self.number not in spots.numbers or self.number in spots.disabled:
            return
        width, height = image_shape[1] // scale, image_shape[0] // scale
        pixel = (
            max(0, min(width - 1, (position[0] + self.offset[0]) // scale)),
            max(0, min(height - 1, (position[1] + self.offset[1]) // scale)),
        )
        spots.move(self.number, pixel, size=(width, height))
