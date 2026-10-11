"""
Transient native-coordinate gestures for drawing measurements on live video.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schema import KINDS, default_region, validate_region

Point = tuple[int, int]


@dataclass
class MeasurementDrawing:
    """
    Keep drafts outside saved measurements until a complete gesture is committed.
    """

    kind: str | None = None
    exclude: bool = False
    size: tuple[int, int] = (256, 192)
    orientation: tuple = ()
    anchors: list[Point] = field(default_factory=list)
    tangents: list[Point] = field(default_factory=list)
    pressed: Point | None = None
    cursor: Point | None = None
    released: bool = False
    swallow_release: bool = False

    def begin(
        self, kind: object, exclude: object, size: tuple[int, int], orientation: list
    ) -> None:
        """
        Start a fresh drawing session after the parent has checked its locks.
        """
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError("Choose a measurement shape to draw")
        if not isinstance(exclude, bool):
            raise TypeError("Exclusion state must be a boolean")
        self.cancel()
        self.swallow_release = False
        self.kind, self.exclude, self.size, self.orientation = (
            kind,
            exclude,
            size,
            tuple(orientation),
        )

    def cancel(self) -> None:
        """
        Discard temporary geometry without creating or modifying a measurement.
        """
        self.swallow_release = self.swallow_release or (
            self.pressed is not None and not self.released
        )
        self.kind = None
        self.anchors.clear()
        self.tangents.clear()
        self.pressed = self.cursor = None
        self.released = False

    def undo(self) -> None:
        """
        Remove the latest polygon/curve anchor or reset an unfinished drag.
        """
        if self.pressed is not None:
            self.pressed = None
            self.swallow_release = True
            return
        if self.anchors:
            self.anchors.pop()
            self.tangents.pop()
        self.pressed = None

    def pointer(self, kind: str, button: str, point: Point | None) -> None:
        """
        Update draft gestures, retaining released geometry until explicit completion.

        Releases outside the camera image discard the current gesture. Polygon
        vertices and Bezier anchors are capped at the persisted topology limit.
        """
        single_drag = self.kind in ("spot", "rectangle", "oval")
        if point is not None and not (single_drag and self.released and kind != "press"):
            self.cursor = point
        if button != "left":
            return
        if kind == "press":
            self.pressed = point
            self.released = False
        elif kind == "release":
            if point is None or self.pressed is None:
                self.pressed = None
                return
            if single_drag:
                self.released = True
                return
            limit = 64 if self.kind == "bezier" else 192
            if len(self.anchors) >= limit:
                self.pressed = None
                raise ValueError("Maximum number of vertices/anchors reached; finish or undo")
            anchor = self.pressed if self.kind == "bezier" else point
            tangent = (
                (point[0] - anchor[0], point[1] - anchor[1]) if self.kind == "bezier" else (0, 0)
            )
            self.anchors.append(anchor)
            self.tangents.append(tangent)
            self.pressed = None

    def geometry(self, *, preview: bool = False) -> tuple[Point, dict]:
        """
        Build a closed region, optionally including a rubber-band vertex/anchor.
        """
        if self.kind is None:
            raise ValueError("No measurement drawing is active")
        region = default_region(self.kind)
        if not preview and self.pressed is not None and not self.released:
            raise ValueError("Release the mouse button before finishing the drawing")
        region["exclude"] = self.exclude
        cursor = self.cursor
        if self.kind in ("spot", "rectangle", "oval"):
            if self.pressed is None or cursor is None:
                raise ValueError("Drag on the camera image to draw the shape")
            center = self.pressed
            if self.kind == "spot":
                region["radius"] = round(math.dist(center, cursor), 2)
                region["radius"] = min(region["radius"], max(self.size))
            else:
                region["points"] = [list(center), list(cursor)]
                center = ((center[0] + cursor[0]) // 2, (center[1] + cursor[1]) // 2)
        else:
            anchors, tangents = self.anchors.copy(), self.tangents.copy()
            if preview and cursor is not None:
                anchor = self.pressed or cursor
                anchors.append(anchor)
                tangents.append((cursor[0] - anchor[0], cursor[1] - anchor[1]))
            if not anchors:
                raise ValueError("Click on the camera image to place the first anchor")
            center = (
                round(sum(p[0] for p in anchors) / len(anchors)),
                round(sum(p[1] for p in anchors) / len(anchors)),
            )
            if self.kind == "polygon":
                region["points"] = [list(point) for point in anchors]
            else:
                for index, anchor in enumerate(anchors):
                    following = (index + 1) % len(anchors)
                    outgoing = self._control(anchor, tangents[index], 1)
                    incoming = self._control(anchors[following], tangents[following], -1)
                    region["points"].extend([list(anchor), outgoing, incoming])
        if not preview:
            region = validate_region(region, self.size)
        return center, region

    def _control(self, anchor: Point, tangent: Point, direction: int) -> list[int]:
        """
        Keep each cubic control handle within the native sensor bounds.
        """
        return [
            max(0, min(bound - 1, value + direction * delta))
            for value, delta, bound in zip(anchor, tangent, self.size, strict=True)
        ]

    def state(self) -> dict:
        """
        Expose only drawing controls and instructions to the Qt viewer.
        """
        hints = {
            "spot": "Drag center to radius or click for a pixel; Enter to finish",
            "rectangle": "Drag opposite corners; Enter to finish",
            "oval": "Drag opposite corners of the bounding box; Enter to finish",
            "polygon": "Click vertices; Enter or right-click to finish",
            "bezier": "Click/drag anchors and tangents; Enter or right-click to finish",
        }
        minimum = 2 if self.kind == "bezier" else 3
        drag_ready = (
            self.kind in ("spot", "rectangle", "oval")
            and self.released
            and self.pressed is not None
        )
        path_ready = (
            self.kind in ("polygon", "bezier")
            and len(self.anchors) >= minimum
            and self.pressed is None
        )
        return {
            "active": self.kind is not None,
            "kind": self.kind,
            "hint": hints.get(self.kind or "", ""),
            "exclude": self.exclude,
            "finish": drag_ready or path_ready,
            "undo": bool(self.anchors or self.pressed),
        }
