"""
Measuring-spot state, pointer events, and drag interaction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import cv2

from ..camera import (
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
)
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
class SampleSpots:
    placing: bool = False
    pixels: list[tuple[int, int]] = field(default_factory=list)
    generation: int = 0

    numbers: list[int] = field(default_factory=list, init=False)
    disabled: set[int] = field(default_factory=set, init=False)
    next_number: int = field(default=1, init=False)
    names: dict[int, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """
        Post init.
        """
        self.numbers = list(range(1, len(self.pixels) + 1))
        self.next_number = len(self.pixels) + 1

    @property
    def active(self) -> Any:
        """
        Active.
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

    def move(self, number: Any, pixel: Any) -> None:
        """
        Move.
        """
        if type(number) is not int or number not in self.numbers:
            raise ValueError("Choose an existing spot to move")
        self.pixels[self.numbers.index(number)] = pixel

    def name(self, number: Any) -> Any:
        """
        Name.
        """
        return self.names.get(number, f"Spot {number}")

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
            (SENSOR_WIDTH, SENSOR_HEIGHT) if rotation in (0, 180) else (SENSOR_HEIGHT, SENSOR_WIDTH)
        )
        pixels = [
            (width - 1 - x if horizontal else x, height - 1 - y if vertical else y)
            for x, y in self.pixels
        ]
        for _ in range((360 - rotation) % 360 // 90):
            pixels = [(height - 1 - y, x) for x, y in pixels]
            width, height = height, width
        return {
            "version": 1,
            "placing": self.placing,
            "next_number": self.next_number,
            "items": [
                {
                    "number": number,
                    "x": x,
                    "y": y,
                    "name": self.name(number),
                    "enabled": number not in self.disabled,
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
        saved = validate_spots(saved)
        self.pixels = [(item["x"], item["y"]) for item in saved["items"]]
        self.numbers = [item["number"] for item in saved["items"]]
        self.names = {item["number"]: item["name"] for item in saved["items"] if item["name"]}
        self.disabled = {item["number"] for item in saved["items"] if not item["enabled"]}
        self.next_number, self.placing = saved["next_number"], saved["placing"]
        width, height = SENSOR_WIDTH, SENSOR_HEIGHT
        for _ in range(rotation // 90):
            self.rotate_clockwise(height)
            width, height = height, width
        self.mirror(width, height, horizontal, vertical)

    def rotate_clockwise(self, sensor_height: int) -> None:
        """
        Rotate clockwise.
        """
        self.pixels = [(sensor_height - 1 - y, x) for x, y in self.pixels]

    def mirror(self, width: int, height: int, horizontal: bool, vertical: bool) -> None:
        """
        Mirror.
        """
        self.pixels = [
            (width - 1 - x if horizontal else x, height - 1 - y if vertical else y)
            for x, y in self.pixels
        ]


def spot_hit(
    position: Any,
    spots: Any,
    image_shape: Any,
    scale: Any,
    viewport_size: tuple[int, int] | None = None,
) -> Any:
    """
    Find the closest enabled marker within ten displayed pixels.
    """
    radius = 10 * image_shape[1] / viewport_size[0] if viewport_size else 10
    hits = []
    for number, (sx, sy) in spots.active:
        anchor = (sx * scale + scale // 2, sy * scale + scale // 2)
        distance = (anchor[0] - position[0]) ** 2 + (anchor[1] - position[1]) ** 2
        if distance <= radius**2:
            hits.append((distance, number, anchor))
    return min(hits) if hits else None


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
        spots.move(self.number, pixel)
