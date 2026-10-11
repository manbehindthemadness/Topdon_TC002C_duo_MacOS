"""
Guarded live-preview measurement creation and transient draft presentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..measurements.overlays import display_boundary, draw_dashed_boundary
from .overlays import draw_contrasting_overlay
from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class MeasurementDrawingController(SessionState):
    """
    Consume drawing events before existing spot dragging and calibration picking.
    """

    def begin_measurement_drawing(self: DesktopSession, command: dict) -> None:
        """
        Arm a live-video tool without changing placement mode or saved geometry.
        """
        state = self.spots_state()
        if not state.get("mapping_available", True):
            raise ValueError("Optical registration is required before drawing thermal measurements")
        if state["locked"]:
            raise ValueError("Stop logging or close calibration before drawing measurements")
        if command.get("orientation") != state["orientation"]:
            raise ValueError("The view changed; choose the drawing tool again")
        if self.viewer is None:
            raise ValueError("Drawing on preview requires the Qt viewer")
        size = state["sensor_size"][0], state["sensor_size"][1]
        self.measurement_drawing.begin(
            command.get("kind"), command.get("exclude", False), size, state["orientation"]
        )
        self.spot_drag.cancel()
        self.picker.consume_drag_events()
        self.picker.consume_clicks()
        self.picker.consume_context_clicks()
        self.notify("Drawing measurement: " + self.measurement_drawing.state()["hint"])

    def reconcile_measurement_drawing(self: DesktopSession) -> None:
        """
        Discard drafts when logging/calibration or a view transformation invalidates them.
        """
        draft = self.measurement_drawing
        if draft.kind is None:
            return
        state = self.spots_state()
        if (state["locked"] or not state.get("mapping_available", True)
                or tuple(state["orientation"]) != draft.orientation):
            draft.cancel()
            self.notify("Measurement drawing cancelled because settings or view changed")

    def measurement_drawing_action(self: DesktopSession, action: str) -> None:
        """
        Finish, undo or cancel an active draft through the parent lock boundary.
        """
        self.reconcile_measurement_drawing()
        draft = self.measurement_drawing
        if draft.kind is None:
            return
        if action == "draw_cancel":
            draft.cancel()
            self.notify("Measurement drawing cancelled")
        elif action == "draw_undo":
            draft.undo()
        elif action == "draw_finish":
            center, region = draft.geometry()
            number = self.spots.create_region(center, region, draft.size)
            draft.cancel()
            self.persist_settings()
            self.spots_panel.open(self.spots_state())
            self.notify(f"Measurement {number} created")

    def measurement_drawing_pointer(self: DesktopSession, command: dict) -> bool:
        """
        Map bitmap events only within camera pixels, excluding the adjoining graph pane.
        """
        was_active = self.measurement_drawing.kind is not None
        self.reconcile_measurement_drawing()
        draft = self.measurement_drawing
        if draft.swallow_release:
            if command["kind"] == "press":
                draft.swallow_release = False
            elif command["button"] == "left" and command["kind"] == "release":
                draft.swallow_release = False
                return True
            elif command.get("flags", 0) & cv2.EVENT_FLAG_LBUTTON:
                return True
        if not was_active:
            return False
        if draft.kind is None:
            return True
        x, y = command["x"], command["y"]
        width, height = draft.size
        scale = self.renderer.scale
        bitmap_width, bitmap_height = (
            self.graph_layout.camera_size
            if self.show_graph and self.graph_layout is not None
            else (width * scale, height * scale)
        )
        point = (
            (
                min(width - 1, x * width // bitmap_width),
                min(height - 1, y * height // bitmap_height),
            )
            if (0 <= x < bitmap_width and 0 <= y < bitmap_height)
            else None
        )
        kind, button = command["kind"], command["button"]
        if self.api.sys.platform == "darwin" and command.get("flags", 0) & cv2.EVENT_FLAG_CTRLKEY:
            button = "right"
        if button == "right" and kind == "release":
            if point is not None:
                self.measurement_drawing_action("draw_finish")
        else:
            draft.pointer(kind, button, point)
        return True

    def draw_measurement_draft(self: DesktopSession, image: np.ndarray) -> np.ndarray:
        """
        Composite a draft outline onto live presentation only, without sampling it.
        """
        self.reconcile_measurement_drawing()
        if self.measurement_drawing.kind is None:
            return image
        try:
            center, region = self.measurement_drawing.geometry(preview=True)
        except ValueError:
            return image
        mask = np.zeros(image.shape[:2], np.uint8)
        points = display_boundary(region, center, self.renderer.scale)
        if len(points):
            if region["exclude"]:
                draw_dashed_boundary(mask, points)
            else:
                cv2.polylines(mask, [points], True, 255, 1, cv2.LINE_AA)
        if region["kind"] == "spot" and region["radius"] == 0:
            cv2.drawMarker(
                mask,
                (
                    center[0] * self.renderer.scale + self.renderer.scale // 2,
                    center[1] * self.renderer.scale + self.renderer.scale // 2,
                ),
                255,
                cv2.MARKER_CROSS,
                9,
                1,
                cv2.LINE_8,
            )
        if region["kind"] == "bezier":
            draw_bezier_handles(mask, region["points"], self.renderer.scale)
        elif region["kind"] == "polygon":
            for point in region["points"]:
                cv2.circle(mask, preview_handle(point, self.renderer.scale), 3, 255, 1, cv2.LINE_AA)
        result = draw_contrasting_overlay(image, mask, np.zeros_like(mask))
        return result


def preview_handle(point: list, scale: int) -> tuple[int, int]:
    """
    Locate a vertex/control handle at its displayed native pixel center.
    """
    return (
        round(float(point[0]) * scale) + scale // 2,
        round(float(point[1]) * scale) + scale // 2,
    )


def draw_bezier_handles(mask: np.ndarray, points: list, scale: int) -> None:
    """
    Show both tangent guides and handles while a cubic measurement is being drawn.
    """
    for index in range(0, len(points), 3):
        anchor = preview_handle(points[index], scale)
        outgoing = preview_handle(points[index + 1], scale)
        incoming = preview_handle(points[index + 2], scale)
        following = preview_handle(points[(index + 3) % len(points)], scale)
        cv2.line(mask, anchor, outgoing, 160, 1, cv2.LINE_AA)
        cv2.line(mask, incoming, following, 160, 1, cv2.LINE_AA)
        cv2.circle(mask, anchor, 3, 255, 1, cv2.LINE_AA)
        cv2.circle(mask, outgoing, 2, 160, 1, cv2.LINE_AA)
        cv2.circle(mask, incoming, 2, 160, 1, cv2.LINE_AA)
