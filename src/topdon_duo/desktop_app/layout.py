"""
OpenCV toolbar and graph geometry with mouse-coordinate mapping.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..graphs import (
    graph_interval_rect,
    graph_log_button_rect,
)
from ..render import (
    TemperatureStats,
)
from .constants import (
    TOOLBAR_BUTTON_HEIGHT,
    TOOLBAR_FONT_SCALE,
    TOOLBAR_PADDING,
    WINDOW_NAME,
)


@dataclass
class ToolbarLayout:
    height: int
    buttons: dict[str, tuple[int, int, int, int]]


@dataclass(frozen=True)
class GraphWindowLayout:
    canvas_size: tuple[int, int]
    camera_size: tuple[int, int]

    @classmethod
    def fit(cls, camera_size: Any, window_size: Any = None) -> Any:
        """
        Fit.
        """
        source_width, source_height = camera_size
        width, height = window_size or (source_width * 2, source_height)
        width, height = max(2, width), max(1, height)
        scale = min((width // 2) / source_width, height / source_height)
        camera = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
        return cls((width, height), camera)

    @property
    def graph_size(self) -> Any:
        """
        Graph size.
        """
        return self.canvas_size[0] - self.camera_size[0], self.canvas_size[1]

    def camera_viewport(self, event_viewport: tuple[int, int] | None = None) -> Any:
        """
        Camera viewport.
        """
        width, height = event_viewport or self.canvas_size
        return (
            self.camera_size[0] * width / self.canvas_size[0],
            self.camera_size[1] * height / self.canvas_size[1],
        )

    def graph_control_at(
        self, x: Any, y: Any, rectangle: Any, event_viewport: tuple[int, int] | None = None
    ) -> Any:
        """
        Graph control at.
        """
        if event_viewport:
            if min(event_viewport) <= 0:
                return False
            x = x * self.canvas_size[0] / event_viewport[0]
            y = y * self.canvas_size[1] / event_viewport[1]
        x0, y0, x1, y1 = rectangle(self.graph_size[0])
        return x0 <= x - self.camera_size[0] <= x1 and y0 <= y <= y1

    def compose(self, camera: Any, graph: Any) -> Any:
        """
        Compose.
        """
        width, height = self.canvas_size
        canvas = np.zeros((height, width, 3), np.uint8)
        camera_width, camera_height = self.camera_size
        resized = cv2.resize(camera, self.camera_size, interpolation=cv2.INTER_LINEAR)
        canvas[:camera_height, :camera_width] = resized
        canvas[:, camera_width:] = graph
        return canvas


def toolbar_layout(width: int) -> ToolbarLayout:
    """
    Fit compact controls across a single row at the current image width.
    """
    controls = (
        ("rotate", "Rotate", 54),
        ("unit", "Units", 75),
        ("view", "Camera", 65),
        ("spots", "Add spots", 84),
        ("capture", "Capture", 64),
        ("graph", "Show graph", 90),
        ("help", "Help", 42),
        ("quit", "Quit", 38),
    )
    buttons: dict[str, tuple[int, int, int, int]] = {}
    available_width = width - (len(controls) + 1) * TOOLBAR_PADDING
    total_weight = sum(button_width for _action, _label, button_width in controls)
    weight = 0
    for index, (action, _label, button_width) in enumerate(controls):
        x0 = TOOLBAR_PADDING * (index + 1) + round(available_width * weight / total_weight)
        weight += button_width
        x1 = TOOLBAR_PADDING * (index + 1) + round(available_width * weight / total_weight)
        buttons[action] = (x0, TOOLBAR_PADDING, x1, TOOLBAR_PADDING + TOOLBAR_BUTTON_HEIGHT)
    return ToolbarLayout(
        height=TOOLBAR_PADDING * 2 + TOOLBAR_BUTTON_HEIGHT,
        buttons=buttons,
    )


def draw_toolbar(
    image: np.ndarray,
    ambient_celsius: float | None,
    temperature_unit: str,
    placing_spots: bool = False,
    recording_mode: str | None = None,
    pending_recording: str | None = None,
    status: str = "Ready",
    stats: TemperatureStats | None = None,
    show_graph: bool = False,
    graph_locked: bool = False,
    settings_locked: bool = False,
    spots_locked: bool = False,
) -> np.ndarray:
    """
    Draw toolbar.
    """
    # Retain these legacy arguments for callers; toolbar labels no longer use them.
    del ambient_celsius, status, stats
    layout = toolbar_layout(image.shape[1])
    canvas = np.zeros((image.shape[0] + layout.height, image.shape[1], 3), np.uint8)
    canvas[layout.height :] = image
    labels = {
        "rotate": "Rotate",
        "unit": f"Units: {temperature_unit}/{'in' if temperature_unit == 'F' else 'cm'}",
        "view": "Camera",
        "spots": "Add spots",
        "capture": "Capture",
        "graph": "Hide graph" if show_graph else "Show graph",
        "help": "Help",
        "quit": "Quit",
    }
    for action, (x0, y0, x1, y1) in layout.buttons.items():
        active = (
            (action == "graph" and show_graph)
            or action == "unit"
            or (action == "spots" and placing_spots)
        )
        active = active or (action == "capture" and bool(recording_mode or pending_recording))
        disabled = (
            (action == "graph" and graph_locked)
            or (settings_locked and action in ("rotate", "unit", "spots"))
            or (action == "spots" and spots_locked)
        )
        fill = (30, 32, 38) if disabled else (74, 92, 70) if active else (47, 51, 61)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
        label = labels[action]
        (text_width, text_height), _baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, TOOLBAR_FONT_SCALE, 1
        )
        font_scale = TOOLBAR_FONT_SCALE
        if text_width > x1 - x0 - TOOLBAR_PADDING * 2:
            font_scale *= max(1, x1 - x0 - TOOLBAR_PADDING * 2) / text_width
            (text_width, text_height), _baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
        cv2.putText(
            canvas,
            label,
            (
                x0 + (x1 - x0 - text_width) // 2,
                y0 + (y1 - y0 + text_height) // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (120, 125, 135) if disabled else (235, 238, 244),
            1,
            cv2.LINE_AA,
        )
    return canvas


def toolbar_action_at(
    x: int,
    y: int,
    image_width: int,
    viewport_size: tuple[int, int] | None = None,
    canvas_height: int | None = None,
) -> str | None:
    """
    Toolbar action at.
    """
    layout = toolbar_layout(image_width)
    if viewport_size and canvas_height:
        if viewport_size[0] <= 0 or viewport_size[1] <= 0:
            return None
        x = round(x * image_width / viewport_size[0])
        y = round(y * canvas_height / viewport_size[1])
    for action, (x0, y0, x1, y1) in layout.buttons.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return action
    return None


def graph_logging_button_at(
    x: Any, y: Any, image_width: Any, canvas_height: Any, viewport_size: Any = None
) -> Any:
    """
    Graph logging button at.
    """
    return graph_control_at(x, y, image_width, canvas_height, viewport_size, graph_log_button_rect)


def graph_interval_at(
    x: Any, y: Any, image_width: Any, canvas_height: Any, viewport_size: Any = None
) -> Any:
    """
    Graph interval at.
    """
    return graph_control_at(x, y, image_width, canvas_height, viewport_size, graph_interval_rect)


def graph_control_at(
    x: Any, y: Any, image_width: Any, canvas_height: Any, viewport_size: Any, rectangle: Any
) -> Any:
    """
    Graph control at.
    """
    if viewport_size:
        if viewport_size[0] <= 0 or viewport_size[1] <= 0:
            return False
        x = round(x * image_width / viewport_size[0])
        y = round(y * canvas_height / viewport_size[1])
    x0, y0, x1, y1 = rectangle(image_width)
    return x0 <= x - image_width <= x1 and y0 <= y <= y1


def mouse_viewport_size() -> tuple[int, int] | None:
    """
    Return viewport dimensions only when mouse events need image scaling.
    """
    # Linux Qt/GTK and macOS Cocoa callbacks report coordinates in the imshow image,
    # including its toolbar, even when the window is resized or letterboxed.
    # Scaling those coordinates again moves the sampler away from the cursor.
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        return None
    try:
        _left, _top, width, height = cv2.getWindowImageRect(WINDOW_NAME)
        return width, height
    except cv2.error:
        return None


def image_position_at(
    x: int,
    y: int,
    image_shape: tuple[int, ...],
    viewport_size: tuple[int, int] | None = None,
    toolbar_height: int = 0,
) -> tuple[int, int] | None:
    """
    Map a mouse event to an image pixel, excluding toolbar and outside clicks.
    """
    height, width = image_shape[:2]
    if viewport_size:
        if viewport_size[0] <= 0 or viewport_size[1] <= 0:
            return None
        x = round(x * width / viewport_size[0])
        y = round(y * (height + toolbar_height) / viewport_size[1])
    y -= toolbar_height
    if not (0 <= x < width and 0 <= y < height):
        return None
    return x, y
