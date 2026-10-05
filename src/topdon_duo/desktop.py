"""Native OpenCV desktop viewer with per-pixel inspection and radiometric saves."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import FRAME_RATE, SENSOR_HEIGHT, SENSOR_WIDTH, CameraError, TC002CDuoCamera
from .capture_panel import CapturePanel
from .graphs import GraphSnapshot, GraphWorker, draw_graph_logging_control, graph_log_button_rect
from .hardware_controls import HardwareControls
from .pointer import PointerMonitor
from .recording import VideoRecorder
from .render import (
    READOUT_HEIGHT,
    RenderedThermalFrame,
    TemperatureStats,
    ThermalRenderer,
    draw_temperature_readout,
)
from .settings_preferences import load_settings, save_settings
from .view_panel import ViewPanel
from .view_settings import VIEW_DEFAULTS
from .window_preferences import load_main_window_size, save_main_window_size
from .window_style import set_black_window_backgrounds, window_resize_size

LOG = logging.getLogger(__name__)
WINDOW_NAME = "TOPDON TC002C Duo"
TOOLBAR_ROW_HEIGHT = 30
TOOLBAR_BUTTON_HEIGHT = 24
TOOLBAR_PADDING = 4
TOOLBAR_STATUS_HEIGHT = 20
TOOLBAR_FONT_SCALE = 0.36
TIMELAPSE_DEFAULT_FPM = 60
TIMELAPSE_MAX_FPM = FRAME_RATE * 60
SAVE_DIALOG_SCRIPT = """
on run argv
    tell current application to activate
    set defaultName to item 1 of argv
    set defaultFolder to item 2 of argv
    if defaultFolder is "" then
        set chosenFile to choose file name with prompt "Save thermal capture" default name defaultName
    else
        set chosenFile to choose file name with prompt "Save thermal capture" default name defaultName default location (POSIX file defaultFolder)
    end if
    return POSIX path of chosenFile
end run
"""


@dataclass
class MousePicker:
    x: int | None = None
    y: int | None = None
    clicks: list[tuple[int, int]] = field(default_factory=list)

    def callback(self, event: int, x: int, y: int, flags: int, _parameter) -> None:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONUP:
            self.x, self.y = x, y
            self.clicks.append((x, y))

    def consume_clicks(self) -> list[tuple[int, int]]:
        clicks, self.clicks = self.clicks, []
        return clicks


@dataclass(frozen=True)
class ToolbarLayout:
    height: int
    buttons: dict[str, tuple[int, int, int, int]]


@dataclass
class SampleSpots:
    placing: bool = False
    pixels: list[tuple[int, int]] = field(default_factory=list)
    generation: int = 0

    def toggle(self) -> None:
        self.placing = not self.placing
        if not self.placing:
            self.pixels.clear()
            self.generation += 1

    def add(self, pixel: tuple[int, int] | None) -> None:
        if self.placing and pixel is not None and pixel not in self.pixels:
            self.pixels.append(pixel)

    def rotate_clockwise(self, sensor_height: int) -> None:
        self.pixels = [(sensor_height - 1 - y, x) for x, y in self.pixels]

    def mirror(self, width: int, height: int, horizontal: bool, vertical: bool) -> None:
        self.pixels = [
            (width - 1 - x if horizontal else x, height - 1 - y if vertical else y)
            for x, y in self.pixels
        ]


class MacSaveDialog:
    """Non-blocking macOS save panel so USB capture continues behind it."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    @property
    def is_open(self) -> bool:
        return self._process is not None

    def open(
        self, default_directory: Path | None = None, *, suffix: str = ".png", kind: str = ""
    ) -> bool:
        if self._process is not None:
            return False
        prefix = f"TC002C-Duo-{kind}-" if kind else "TC002C-Duo-"
        default_name = prefix + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + suffix
        directory = ""
        if default_directory is not None and default_directory.is_dir():
            directory = str(default_directory.resolve())
        command = self._command(default_name, directory)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True

    def _command(self, default_name: str, directory: str) -> list[str]:
        script = SAVE_DIALOG_SCRIPT
        if Path(default_name).suffix.lower() == ".mp4":
            script = script.replace("Save thermal capture", "Save thermal recording")
        elif Path(default_name).suffix.lower() == ".csv":
            script = script.replace("Save thermal capture", "Save temperature log")
        return [
            "/usr/bin/osascript",
            "-e",
            script,
            "--",
            default_name,
            directory,
        ]

    def _cancelled(self, returncode: int, stderr: str) -> bool:
        return "User canceled" in stderr

    def poll(self) -> tuple[bool, Path | None]:
        """Return (finished, selected path); cancellation yields (True, None)."""
        if self._process is None or self._process.poll() is None:
            return False, None
        process, self._process = self._process, None
        stdout, stderr = process.communicate()
        if process.returncode == 0 and stdout.strip():
            return True, Path(stdout.strip())
        if not self._cancelled(process.returncode, stderr):
            LOG.error("Save dialog failed: %s", stderr.strip() or process.returncode)
        return True, None

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None


class LinuxSaveDialog(MacSaveDialog):
    """Non-blocking GTK save panel supplied by Ubuntu's zenity package."""

    def _cancelled(self, returncode: int, stderr: str) -> bool:
        return returncode == 1

    def _command(self, default_name: str, directory: str) -> list[str]:
        video = Path(default_name).suffix.lower() == ".mp4"
        csv_log = Path(default_name).suffix.lower() == ".csv"
        return [
            "zenity",
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title=Save temperature log"
            if csv_log
            else "--title=Save thermal recording"
            if video
            else "--title=Save thermal capture",
            f"--filename={Path(directory) / default_name}",
            "--file-filter=CSV logs | *.csv"
            if csv_log
            else "--file-filter=MP4 videos | *.mp4"
            if video
            else "--file-filter=PNG images | *.png",
        ]


def toolbar_layout(width: int) -> ToolbarLayout:
    """Fit compact controls across a single row at the current image width."""
    controls = (
        ("rotate", "Rotate", 54),
        ("unit", "C / F", 50),
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
        height=TOOLBAR_PADDING * 2 + TOOLBAR_ROW_HEIGHT + TOOLBAR_STATUS_HEIGHT + READOUT_HEIGHT,
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
) -> np.ndarray:
    layout = toolbar_layout(image.shape[1])
    canvas = np.zeros((image.shape[0] + layout.height, image.shape[1], 3), np.uint8)
    canvas[layout.height :] = image
    if stats is not None:
        canvas[layout.height - READOUT_HEIGHT :] = draw_temperature_readout(
            image, stats, ambient_celsius, temperature_unit
        )
    labels = {
        "rotate": "Rotate",
        "unit": f"Unit: {temperature_unit}",
        "view": "Camera",
        "spots": "Clear spots" if placing_spots else "Add spots",
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
        disabled = (action == "graph" and graph_locked) or (
            settings_locked and action in ("rotate", "unit", "spots")
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
    while (
        status
        and cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] > image.shape[1] - 12
    ):
        status = status[:-1]
    cv2.putText(
        canvas,
        status,
        (6, layout.height - READOUT_HEIGHT - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (210, 215, 225),
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


def graph_logging_button_at(x, y, image_width, canvas_height, viewport_size=None):
    if viewport_size:
        if viewport_size[0] <= 0 or viewport_size[1] <= 0:
            return False
        x = round(x * image_width / viewport_size[0])
        y = round(y * canvas_height / viewport_size[1])
    x0, y0, x1, y1 = graph_log_button_rect(image_width)
    return x0 <= x - image_width <= x1 and y0 <= y <= y1


def _restore_user_ownership(paths: list[Path]) -> None:
    """Make sudo-created captures belong to the invoking desktop user."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    for path in paths:
        try:
            os.chown(path, int(uid), int(gid))
        except OSError:
            pass


def save_capture(
    rendered: RenderedThermalFrame,
    output_directory: Path,
    ambient_celsius: float | None,
    rotation: int,
    selected_pixel: tuple[int, int] | None = None,
    display_unit: str = "C",
    base_path: Path | None = None,
) -> list[Path]:
    """Save a viewable PNG plus lossless raw/Celsius data and JSON metadata."""
    directory_existed = output_directory.exists()
    output_directory.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now().astimezone()
    stamp = captured_at.strftime("%Y%m%d-%H%M%S-%f")
    if base_path is None:
        base = output_directory / f"TC002C-Duo-{stamp}"
    else:
        base = base_path
        if base.suffix.lower() in (".png", ".npz", ".json"):
            base = base.with_suffix("")
    png_path = base.with_suffix(".png")
    data_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")

    capture_image = draw_temperature_readout(
        rendered.image, rendered.stats, ambient_celsius, display_unit
    )
    if not cv2.imwrite(str(png_path), capture_image):
        raise OSError(f"could not write {png_path}")
    np.savez_compressed(
        data_path,
        raw_counts=rendered.raw_counts.astype(np.uint16),
        temperatures_celsius=rendered.temperatures_celsius.astype(np.float32),
        ambient_celsius=np.float32(ambient_celsius if ambient_celsius is not None else np.nan),
        raw_gain_divisor=np.float32(64.0),
        rotation_degrees=np.int16(rotation),
        mirror_horizontal=np.bool_(rendered.display_settings.get("mirror_horizontal", False)),
        mirror_vertical=np.bool_(rendered.display_settings.get("mirror_vertical", False)),
    )

    metadata: dict[str, object] = {
        "captured_at": captured_at.isoformat(),
        "camera": "TOPDON TC002C Duo",
        "sensor_shape": list(rendered.temperatures_celsius.shape),
        "ambient_celsius": ambient_celsius,
        "rotation_degrees": rotation,
        "display_unit": display_unit,
        "image_source": rendered.image_source,
        "display_settings": rendered.display_settings,
        "temperature_stats_celsius": rendered.stats.as_dict(),
        "radiometric_file": data_path.name,
        "image_file": png_path.name,
    }
    if selected_pixel is not None:
        x, y = selected_pixel
        metadata["selected_pixel"] = {
            "x": x,
            "y": y,
            "temperature_celsius": round(float(rendered.temperatures_celsius[y, x]), 3),
            "raw_count": int(rendered.raw_counts[y, x]),
        }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    paths = [png_path, data_path, json_path]
    if not directory_existed:
        paths.insert(0, output_directory)
    _restore_user_ownership(paths)
    return [png_path, data_path, json_path]


def mouse_viewport_size() -> tuple[int, int] | None:
    """Return viewport dimensions only when mouse events need image scaling."""
    # Linux Qt/GTK callbacks already report coordinates in the imshow image,
    # including its toolbar, even when the window is resized or letterboxed.
    # Scaling those coordinates again moves the sampler away from the cursor.
    if sys.platform.startswith("linux"):
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
    """Map a mouse event to an image pixel, excluding toolbar and outside clicks."""
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


def _draw_contrasting_overlay(
    image: np.ndarray, mask: np.ndarray, text_mask: np.ndarray
) -> np.ndarray:
    """Draw inverted crosshairs and stable white labels with a black outline."""
    result = image.copy()
    x, y, width, height = cv2.boundingRect(cv2.max(mask, text_mask))
    if width == 0 or height == 0:
        return result
    # Limit compositing to the overlay bounds, including the two-pixel text outline.
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(image.shape[1], x + width + 2), min(image.shape[0], y + height + 2)
    region = image[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    text_mask = text_mask[y0:y1, x0:x1]
    inverted = 255 - region
    # Inversion alone disappears on middle gray. A black/white halo also
    # separates the strokes from busy thermal detail without hiding a whole box.
    luminance = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    halo_color = np.where(luminance >= 128, 0, 255).astype(np.uint8)
    halo_mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    halo_alpha = (halo_mask.astype(np.float32) / 255.0)[..., None]
    outlined = region * (1.0 - halo_alpha) + halo_color[..., None] * halo_alpha
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    composited = outlined * (1.0 - alpha) + inverted * alpha
    # Keep each digit a consistent color even when it straddles hot/cold detail.
    text_outline = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    outline_alpha = (text_outline.astype(np.float32) / 255.0)[..., None]
    text_alpha = (text_mask.astype(np.float32) / 255.0)[..., None]
    composited *= 1.0 - outline_alpha
    result[y0:y1, x0:x1] = np.rint(composited * (1.0 - text_alpha) + 255.0 * text_alpha).astype(
        np.uint8
    )
    return result


def _place_temperature_label(
    text: str,
    anchor: tuple[int, int],
    image_shape: tuple[int, ...],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Try nearby positions first, reserving the text outline and a small gap."""
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    padding = 3
    box_width, box_height = width + padding * 2, height + baseline + padding * 2
    image_height, image_width = image_shape[:2]
    ax, ay = anchor
    best = None
    seen = set()
    row_step = box_height + 4
    for ring in range(max(image_height, image_width) // row_step + 1):
        above = ay - 8 - height - padding - ring * row_step
        below = ay + 8 + ring * row_step
        right = ax + 8 + ring * (box_width + 4)
        left = ax - 8 - box_width - ring * (box_width + 4)
        for x, y in (
            (right, above),
            (right, below),
            (left, above),
            (left, below),
            (ax - box_width // 2, above),
            (ax - box_width // 2, below),
            (right, ay - box_height // 2),
            (left, ay - box_height // 2),
        ):
            x = max(0, min(x, image_width - box_width))
            y = max(0, min(y, image_height - box_height))
            if (x, y) in seen:
                continue
            seen.add((x, y))
            rect = (x, y, x + box_width, y + box_height)
            overlap = sum(
                max(0, min(rect[2], other[2]) - max(x, other[0]))
                * max(0, min(rect[3], other[3]) - max(y, other[1]))
                for other in occupied
            )
            origin = (x + padding, y + height + padding)
            if overlap == 0:
                return origin, rect
            distance = (x + box_width / 2 - ax) ** 2 + (y + box_height / 2 - ay) ** 2
            score = (overlap, distance)
            if best is None or score < best[0]:
                best = (score, origin, rect)
    # When the image is completely crowded, choose the least obstructed position.
    assert best is not None
    return best[1], best[2]


def _spot_label_layout(
    rendered: RenderedThermalFrame, spots: SampleSpots, scale: int, temperature_unit: str
) -> tuple[
    list[tuple[str, tuple[int, int], tuple[int, int, int, int]]], list[tuple[int, int, int, int]]
]:
    anchors = [(x * scale + scale // 2, y * scale + scale // 2) for x, y in spots.pixels]
    occupied = [(x - 6, y - 6, x + 7, y + 7) for x, y in anchors]
    labels = []
    for (sensor_x, sensor_y), anchor in zip(spots.pixels, anchors):
        temperature = float(rendered.temperatures_celsius[sensor_y, sensor_x])
        if temperature_unit == "F":
            temperature = temperature * 9.0 / 5.0 + 32.0
        text = f"{temperature:.2f} {temperature_unit}"
        origin, rect = _place_temperature_label(text, anchor, rendered.image.shape, occupied)
        occupied.append(rect)
        labels.append((text, origin, rect))
    return labels, occupied


def _draw_label_leader(
    mask: np.ndarray, anchor: tuple[int, int], rect: tuple[int, int, int, int]
) -> None:
    """Connect every temperature label to its sampled pixel."""
    endpoint = (
        min(max(anchor[0], rect[0]), rect[2] - 1),
        min(max(anchor[1], rect[1]), rect[3] - 1),
    )
    cv2.line(mask, anchor, endpoint, 255, 1, cv2.LINE_AA)


def draw_sample_spots(
    image: np.ndarray,
    rendered: RenderedThermalFrame,
    spots: SampleSpots,
    scale: int,
    temperature_unit: str = "C",
) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    text_mask = np.zeros_like(mask)
    labels, _occupied = _spot_label_layout(rendered, spots, scale, temperature_unit)
    for (sensor_x, sensor_y), (text, origin, rect) in zip(spots.pixels, labels):
        anchor = (sensor_x * scale + scale // 2, sensor_y * scale + scale // 2)
        _draw_label_leader(mask, anchor, rect)
        cv2.putText(
            text_mask,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            255,
            1,
            cv2.LINE_AA,
        )
    # Union the markers so overlapping spots are inverted only once.
    for sensor_x, sensor_y in spots.pixels:
        cv2.drawMarker(
            mask,
            (sensor_x * scale + scale // 2, sensor_y * scale + scale // 2),
            255,
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_8,
        )
    return _draw_contrasting_overlay(image, mask, text_mask)


def draw_picker(
    rendered: RenderedThermalFrame,
    picker: MousePicker,
    scale: int,
    viewport_size: tuple[int, int] | None = None,
    toolbar_height: int = 0,
    temperature_unit: str = "C",
    pointer_over_image: bool | None = None,
    spots: SampleSpots | None = None,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    image = rendered.image.copy()
    if pointer_over_image is False or picker.x is None or picker.y is None:
        return image, None
    position = image_position_at(picker.x, picker.y, image.shape, viewport_size, toolbar_height)
    if position is None:
        return image, None
    image_x, image_y = position
    sensor_x = min(max(image_x // scale, 0), rendered.temperatures_celsius.shape[1] - 1)
    sensor_y = min(max(image_y // scale, 0), rendered.temperatures_celsius.shape[0] - 1)
    temperature = float(rendered.temperatures_celsius[sensor_y, sensor_x])
    if temperature_unit == "F":
        temperature = temperature * 9.0 / 5.0 + 32.0

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    text_mask = np.zeros_like(mask)
    cv2.drawMarker(
        mask,
        (image_x, image_y),
        255,
        markerType=cv2.MARKER_CROSS,
        markerSize=9,
        thickness=1,
        line_type=cv2.LINE_8,
    )
    text = f"({sensor_x}, {sensor_y}) {temperature:.2f} {temperature_unit}"
    _, occupied = _spot_label_layout(rendered, spots or SampleSpots(), scale, temperature_unit)
    occupied.append((image_x - 6, image_y - 6, image_x + 7, image_y + 7))
    origin, rect = _place_temperature_label(text, (image_x, image_y), image.shape, occupied)
    _draw_label_leader(mask, (image_x, image_y), rect)
    cv2.putText(
        text_mask,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        255,
        1,
        cv2.LINE_AA,
    )
    return _draw_contrasting_overlay(image, mask, text_mask), (sensor_x, sensor_y)


def draw_control_instructions(image: np.ndarray) -> np.ndarray:
    """Draw a translucent keyboard/mouse help panel over the image."""
    result = image.copy()
    panel_width = min(390, result.shape[1] - 20)
    panel_height = min(349, result.shape[0] - 20)
    x0, y0 = 10, result.shape[0] - panel_height - 10
    x1, y1 = x0 + panel_width, y0 + panel_height

    overlay = result.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (8, 10, 16), -1)
    cv2.addWeighted(overlay, 0.82, result, 0.18, 0, result)
    cv2.rectangle(result, (x0, y0), (x1, y1), (120, 130, 150), 1)

    lines = (
        ("Controls", (255, 255, 255)),
        ("Mouse move   Inspect pixel temperature", (210, 215, 225)),
        ("P / Add spots  Place spots; again clears", (210, 215, 225)),
        ("Camera Hardware ambient and image controls", (210, 215, 225)),
        ("S            Save image data", (210, 215, 225)),
        ("C            Open Capture controls", (210, 215, 225)),
        ("O            Rotate 90 degrees clockwise", (210, 215, 225)),
        ("F            Toggle Celsius / Fahrenheit", (210, 215, 225)),
        ("V            Open Camera controls", (210, 215, 225)),
        ("G            Show / hide graph area", (210, 215, 225)),
        ("L            Start / stop CSV logging", (210, 215, 225)),
        ("Space        Hide controls", (210, 215, 225)),
        ("Q / Esc      Quit", (210, 215, 225)),
    )
    for index, (line, color) in enumerate(lines):
        cv2.putText(
            result,
            line,
            (x0 + 14, y0 + 25 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48 if index else 0.58,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native TOPDON TC002C Duo viewer")
    parser.add_argument("--ambient", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=None)
    parser.add_argument("--scale", type=int, choices=range(1, 7), default=3)
    parser.add_argument(
        "--image-source",
        choices=("preview", "raw"),
        default=None,
        help="camera preview (default, when available) or raw thermal visualization",
    )
    parser.add_argument(
        "--timelapse-fpm",
        type=int,
        default=TIMELAPSE_DEFAULT_FPM,
        help="timelapse frames per minute (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="initial directory for the Save dialog",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.timelapse_fpm <= TIMELAPSE_MAX_FPM:
        parser.error(f"--timelapse-fpm must be between 1 and {TIMELAPSE_MAX_FPM}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    saved_settings = load_settings()
    remembered_hardware = saved_settings.get("hardware", {}).copy()
    advanced_auto = saved_settings.get("advanced_auto", True)
    show_graph = saved_settings.get("show_graph", False)
    requested_window_size = None
    saved_window_size = load_main_window_size()
    last_window_size = None
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    renderer = ThermalRenderer(
        scale=args.scale,
        ambient_celsius=None,
        rotation=args.rotate if args.rotate is not None else saved_settings.get("rotation", 0),
        image_source=args.image_source or "preview",
    )
    for name, value in saved_settings.get("display", {}).items():
        renderer.set_view_setting(name, value)
    if args.image_source is not None:
        renderer.set_view_setting("image_source", args.image_source)
    renderer.native_temperatures = True
    if args.ambient is not None:
        LOG.warning("--ambient is ignored; set hardware ambient temperature in Camera.")
    camera = TC002CDuoCamera()
    picker = MousePicker()
    spots = SampleSpots()
    pointer_monitor = PointerMonitor(WINDOW_NAME)
    save_dialog = LinuxSaveDialog() if sys.platform.startswith("linux") else MacSaveDialog()
    recorder = VideoRecorder()
    capture_panel = CapturePanel()
    view_panel = ViewPanel()
    hardware = HardwareControls(camera)
    graphs = GraphWorker()
    actual_image_source = renderer.image_source
    pending_save_kind: str | None = None
    capture_cursor = False
    timelapse_fpm = args.timelapse_fpm
    status_message = "Ready"
    status_until = 0.0
    status = "Ready"
    show_instructions = False
    last_selected: tuple[int, int] | None = None

    def persist_settings() -> None:
        try:
            save_settings(
                {
                    "display": {name: getattr(renderer, name) for name in VIEW_DEFAULTS},
                    "hardware": remembered_hardware,
                    "rotation": renderer.rotation,
                    "advanced_auto": advanced_auto,
                    "show_graph": show_graph,
                }
            )
        except OSError as exc:
            LOG.warning("Could not save Camera settings: %s", exc)

    def toggle_graph() -> None:
        nonlocal show_graph, requested_window_size, last_window_size
        if graphs.logging or pending_save_kind == "graph_log":
            notify("Stop graph logging before hiding graphs.")
            return
        size = window_resize_size(WINDOW_NAME) or last_window_size or requested_window_size
        if size is None:
            return
        width, height = size
        show_graph = not show_graph
        if not show_graph:
            graphs.pause()
        requested_window_size = (width * 2 if show_graph else max(1, round(width / 2)), height)
        cv2.resizeWindow(WINDOW_NAME, *requested_window_size)
        last_window_size = requested_window_size
        picker.x = picker.y = None
        persist_settings()

    def notify(message: str) -> None:
        nonlocal status_message, status_until
        status_message = message
        status_until = time.monotonic() + 8.0
        LOG.info("%s", message)

    def toggle_graph_logging() -> None:
        nonlocal pending_save_kind
        if graphs.logging:
            try:
                path = graphs.stop_logging()
                if path is not None:
                    _restore_user_ownership([path])
                    notify(f"Saved temperature log: {path.name}")
            except OSError as exc:
                notify(f"Could not finish temperature log: {exc}")
        elif pending_save_kind == "graph_log":
            save_dialog.close()
            pending_save_kind = None
            notify("Temperature logging cancelled")
        elif save_dialog.is_open:
            notify("Finish the current Save dialog before starting temperature logging.")
        elif show_graph:
            try:
                if save_dialog.open(args.output, suffix=".csv", kind="temperatures"):
                    pending_save_kind = "graph_log"
            except OSError as exc:
                notify(f"Could not choose a temperature log file: {exc}")

    def set_timelapse_fpm(value: int) -> None:
        nonlocal timelapse_fpm
        if not recorder.is_recording and pending_save_kind not in ("video", "timelapse"):
            timelapse_fpm = min(max(value, 1), TIMELAPSE_MAX_FPM)

    def stop_recording() -> None:
        try:
            path = recorder.stop()
        except (OSError, cv2.error) as exc:
            notify(f"Could not finalize recording: {exc}")
            return
        if path is not None:
            _restore_user_ownership([path])
            notify(f"Saved {path.name} ({recorder.frames_written} frames)")
            LOG.info("Recording saved to %s", path)

    def request_save(kind: str = "image") -> None:
        nonlocal pending_save_kind
        if kind in ("video", "timelapse"):
            if recorder.is_recording:
                if recorder.mode == kind:
                    stop_recording()
                return
            if pending_save_kind == kind:
                save_dialog.close()
                pending_save_kind = None
                notify("Recording cancelled")
                return
        if save_dialog.is_open:
            notify("Finish or cancel the open Save dialog first")
            return
        try:
            opened = (
                save_dialog.open(args.output)
                if kind == "image"
                else save_dialog.open(args.output, suffix=".mp4", kind=kind)
            )
            if opened:
                pending_save_kind = kind
        except OSError as exc:
            notify(f"Unable to open Save dialog: {exc}")

    def rotate_view() -> None:
        nonlocal last_selected
        if graphs.logging:
            notify("Stop temperature logging before changing settings.")
            return
        # Rotate the stored sensor coordinates with the image, preserving samples.
        sensor_height = SENSOR_HEIGHT if renderer.rotation in (0, 180) else SENSOR_WIDTH
        sensor_width = SENSOR_WIDTH if renderer.rotation in (0, 180) else SENSOR_HEIGHT
        spots.mirror(
            sensor_width, sensor_height, renderer.mirror_horizontal, renderer.mirror_vertical
        )
        spots.rotate_clockwise(sensor_height)
        spots.mirror(
            sensor_height, sensor_width, renderer.mirror_horizontal, renderer.mirror_vertical
        )
        renderer.rotate_clockwise()
        persist_settings()
        picker.x = picker.y = None
        last_selected = None

    def set_view_setting(name: str, value: object) -> None:
        nonlocal last_selected
        if graphs.logging:
            notify("Stop temperature logging before changing settings.")
            return
        previous = renderer.view_settings()
        renderer.set_view_setting(name, value)
        persist_settings()
        if name in ("mirror_horizontal", "mirror_vertical") and previous[name] != value:
            width, height = (
                (SENSOR_WIDTH, SENSOR_HEIGHT)
                if renderer.rotation in (0, 180)
                else (SENSOR_HEIGHT, SENSOR_WIDTH)
            )
            spots.mirror(width, height, name == "mirror_horizontal", name == "mirror_vertical")
            picker.x = picker.y = None
            last_selected = None

    def toggle_spots() -> None:
        if graphs.logging:
            notify("Stop temperature logging before changing measuring spots.")
            return
        spots.toggle()

    def toggle_temperature_unit() -> None:
        set_view_setting("temperature_unit", "F" if renderer.temperature_unit == "C" else "C")

    def view_state() -> dict:
        message = "Camera preview" if actual_image_source == "preview" else "Raw thermal image"
        if renderer.image_source == "preview" and actual_image_source == "raw":
            message = "Camera preview unavailable; showing the raw thermal image."
        if hardware.error:
            message = hardware.error
        else:
            message += " · Camera temperatures (approximate)"
        if renderer.upsampling != "off":
            if not renderer.enhancement_amount:
                message += " · Enhancement amount 0 (original image)"
            elif renderer.upsampler.error:
                message += f" · {renderer.upsampler.error}; showing unenhanced image"
            else:
                algorithm = "Anime4K09" if renderer.upsampling == "anime4k09" else "ACNet"
                message += f" · {algorithm} 2× · {renderer.upsampler.elapsed_ms:.0f} ms"
        if graphs.logging:
            message += " · Settings locked while logging"
        return {
            **renderer.view_settings(),
            "hardware": hardware.state(),
            "advanced_auto": advanced_auto,
            "settings_locked": graphs.logging,
            "status": message,
        }

    def open_view() -> None:
        try:
            try:
                hardware.load()
            except CameraError as exc:
                hardware.error = str(exc)
            view_panel.open(view_state())
        except OSError as exc:
            notify(f"Could not open Camera: {exc}")

    def capture_state() -> dict:
        return {
            "recording_mode": recorder.mode,
            "pending_recording": pending_save_kind
            if pending_save_kind in ("video", "timelapse")
            else None,
            "capture_cursor": capture_cursor,
            "frames_per_minute": timelapse_fpm,
            "max_fpm": TIMELAPSE_MAX_FPM,
            "status": status,
        }

    def open_capture() -> None:
        try:
            capture_panel.open(capture_state())
        except OSError as exc:
            notify(f"Could not open Capture: {exc}")

    print(
        "Mouse: inspect a pixel | "
        "p: add/clear spots | s: save image data | c: Capture controls | "
        "o: rotate | f: C/F | v: Camera controls | g: show/hide graph | l: log temperatures | Space: controls | q/Esc: quit"
    )
    try:
        camera.open()
        try:
            hardware.load()
            for name, value in remembered_hardware.items():
                try:
                    hardware.set(name, value, True)
                except (CameraError, ValueError, TypeError) as exc:
                    hardware.error = f"Could not restore saved {name}: {exc}"
                    LOG.warning("%s", hardware.error)
        except CameraError as exc:
            hardware.error = str(exc)
            LOG.warning("Could not read hardware settings: %s", exc)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, picker.callback)
        set_black_window_backgrounds(WINDOW_NAME)
        initial_window_size_set = False
        for frame in camera.frames():
            for command in view_panel.poll():
                if graphs.logging and command.get("action") in (
                    "setting",
                    "hardware",
                    "advanced_auto",
                    "restore_hardware",
                    "reset",
                ):
                    notify("Stop temperature logging before changing settings.")
                    continue
                try:
                    if command.get("action") == "setting":
                        set_view_setting(command["name"], command["value"])
                    elif command.get("action") == "hardware":
                        hardware.set(command["name"], command["value"], command["enabled"])
                        if command["enabled"]:
                            remembered_hardware[command["name"]] = hardware.state()[
                                command["name"]
                            ]["value"]
                        else:
                            remembered_hardware.pop(command["name"], None)
                        persist_settings()
                        hardware.error = ""
                        renderer._average_raw = None
                    elif command.get("action") == "advanced_auto":
                        if not isinstance(command["value"], bool):
                            raise ValueError("Advanced / Auto must be a boolean")
                        advanced_auto = command["value"]
                        persist_settings()
                    elif command.get("action") == "restore_hardware":
                        hardware.restore()
                        remembered_hardware.clear()
                        persist_settings()
                        hardware.error = ""
                        renderer._average_raw = None
                    elif command.get("action") == "reset":
                        for name, value in VIEW_DEFAULTS.items():
                            set_view_setting(name, value)
                    elif command.get("action") == "error":
                        notify(f"Camera window failed: {command.get('message', '')}")
                except (KeyError, ValueError, TypeError, CameraError) as exc:
                    hardware.error = f"Camera setting rejected: {exc}"
                    notify(hardware.error)
            for command in capture_panel.poll():
                action = command.get("action")
                if action == "error":
                    notify(f"Capture window failed: {command.get('message', '')}")
                elif action == "rate":
                    set_timelapse_fpm(int(command["value"]))
                elif action == "cursor":
                    capture_cursor = bool(command["value"])
                elif action == "image":
                    request_save()
                elif action in ("video", "timelapse"):
                    set_timelapse_fpm(int(command["frames_per_minute"]))
                    capture_cursor = bool(command["capture_cursor"])
                    request_save(action)
            renderer.camera_preview = hardware.preview_active
            renderer.camera_color = "palette" in hardware.enabled
            renderer.hardware_settings = hardware.state() if hardware.original else {}
            ambient = renderer.hardware_settings.get("ambient", {})
            renderer.ambient_celsius = ambient.get("value") if ambient.get("available") else None
            rendered = renderer.render_detailed(frame)
            actual_image_source = rendered.image_source
            layout = toolbar_layout(rendered.image.shape[1])
            if not initial_window_size_set:
                width, height = saved_window_size or (
                    rendered.image.shape[1] * (2 if show_graph else 1),
                    rendered.image.shape[0] + layout.height,
                )
                requested_window_size = (width, height)
                cv2.resizeWindow(WINDOW_NAME, width, height)
                initial_window_size_set = True
            viewport_size = mouse_viewport_size()
            if show_graph and viewport_size is not None:
                viewport_size = (viewport_size[0] / 2, viewport_size[1])
            display, selected = draw_picker(
                rendered,
                picker,
                renderer.scale,
                viewport_size=viewport_size,
                toolbar_height=layout.height,
                temperature_unit=renderer.temperature_unit,
                spots=spots,
                pointer_over_image=pointer_monitor.over_image(
                    rendered.image.shape[0], layout.height
                ),
            )
            if selected is not None:
                last_selected = selected

            logging_error = graphs.take_logging_error()
            if logging_error:
                notify(logging_error)
            save_finished, save_path = save_dialog.poll()
            if save_finished:
                kind, pending_save_kind = pending_save_kind, None
                if save_path is not None:
                    try:
                        if kind == "graph_log":
                            path = graphs.start_logging(save_path)
                            _restore_user_ownership([path])
                            notify(f"Logging temperatures to {path.name}")
                        elif kind in ("video", "timelapse"):
                            recorder.start(
                                save_path,
                                (
                                    rendered.image.shape[0] + READOUT_HEIGHT,
                                    rendered.image.shape[1],
                                    3,
                                ),
                                kind,
                                frames_per_minute=timelapse_fpm,
                            )
                            notify(f"Recording to {recorder.path.name}")
                        else:
                            saved = save_capture(
                                rendered,
                                save_path.parent,
                                renderer.ambient_celsius,
                                renderer.rotation,
                                last_selected,
                                renderer.temperature_unit,
                                base_path=save_path,
                            )
                            notify(f"Saved {saved[0].name}")
                    except (OSError, ValueError, cv2.error) as exc:
                        notify(f"Capture failed: {exc}")
                else:
                    notify("Save cancelled")
            # Only the thermal view and sample annotations go into the video.
            # The cursor sampler remains visible locally even when capture is off.
            if recorder.is_recording:
                recording_view = draw_sample_spots(
                    display if capture_cursor else rendered.image,
                    rendered,
                    spots,
                    renderer.scale,
                    renderer.temperature_unit,
                )
                try:
                    recorder.write(
                        draw_temperature_readout(
                            recording_view,
                            rendered.stats,
                            renderer.ambient_celsius,
                            renderer.temperature_unit,
                        )
                    )
                except (OSError, cv2.error) as exc:
                    stop_recording()
                    notify(f"Recording failed: {exc}")
            display = draw_sample_spots(
                display, rendered, spots, renderer.scale, renderer.temperature_unit
            )
            if show_instructions:
                display = draw_control_instructions(display)
            if recorder.is_recording:
                status = (
                    f"REC {recorder.mode} | {recorder.elapsed_seconds():.1f}s | "
                    f"{recorder.frames_written} frames"
                )
                if recorder.mode == "timelapse":
                    status += f" | {timelapse_fpm}/min"
            elif pending_save_kind:
                status = (
                    "Choose a filename for temperature logging..."
                    if pending_save_kind == "graph_log"
                    else f"Choose a filename for {pending_save_kind}..."
                )
            elif time.monotonic() < status_until:
                status = status_message
            else:
                status = "Ready"
            capture_panel.update(capture_state())
            display = draw_toolbar(
                display,
                renderer.ambient_celsius,
                renderer.temperature_unit,
                placing_spots=spots.placing,
                recording_mode=recorder.mode,
                pending_recording=pending_save_kind
                if pending_save_kind in ("video", "timelapse")
                else None,
                status=status,
                stats=rendered.stats,
                show_graph=show_graph,
                graph_locked=graphs.logging or pending_save_kind == "graph_log",
                settings_locked=graphs.logging,
            )

            # Capture spot values in this frame's orientation, before click actions can rotate
            # coordinates or add/clear spots. The next frame supplies any changed selection.
            graph_spots = (
                tuple(
                    ((spots.generation, index), float(rendered.temperatures_celsius[y, x]))
                    for index, (x, y) in enumerate(spots.pixels)
                )
                if show_graph
                else ()
            )
            quit_requested = False
            for click_x, click_y in picker.consume_clicks():
                if show_graph and graph_logging_button_at(
                    click_x, click_y, rendered.image.shape[1], display.shape[0], viewport_size
                ):
                    toggle_graph_logging()
                    continue
                action = toolbar_action_at(
                    click_x,
                    click_y,
                    rendered.image.shape[1],
                    viewport_size=viewport_size,
                    canvas_height=display.shape[0],
                )
                if action == "capture":
                    open_capture()
                elif action == "rotate":
                    rotate_view()
                elif action == "spots":
                    toggle_spots()
                elif action == "view":
                    open_view()
                elif action == "unit":
                    toggle_temperature_unit()
                elif action == "graph":
                    toggle_graph()
                elif action == "help":
                    show_instructions = not show_instructions
                elif action == "quit":
                    quit_requested = True
                elif action is None and spots.placing:
                    if graphs.logging:
                        notify("Stop temperature logging before changing measuring spots.")
                        continue
                    position = image_position_at(
                        click_x, click_y, rendered.image.shape, viewport_size, layout.height
                    )
                    spots.add(
                        (position[0] // renderer.scale, position[1] // renderer.scale)
                        if position is not None
                        else None
                    )
            if quit_requested:
                break
            view_panel.update(view_state())
            graph_size = (display.shape[1], display.shape[0])
            if show_graph:
                graphs.submit(
                    GraphSnapshot(
                        stats=(
                            rendered.stats.minimum,
                            rendered.stats.average,
                            rendered.stats.maximum,
                            rendered.stats.center,
                        ),
                        spots=graph_spots,
                        size=graph_size,
                        unit=renderer.temperature_unit,
                    )
                )
                display = np.concatenate((display, graphs.image(graph_size)), axis=1)
                draw_graph_logging_control(
                    display[:, graph_size[0] :], graphs.logging, pending_save_kind == "graph_log"
                )
            else:
                graphs.pause()
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            current_size = window_resize_size(WINDOW_NAME)
            if current_size is not None:
                last_window_size = current_size
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
                    break
            except cv2.error:
                if last_window_size is not None:
                    break
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                rotate_view()
            elif key == ord("p"):
                toggle_spots()
            elif key == ord(" "):
                show_instructions = not show_instructions
            elif key == ord("g"):
                toggle_graph()
            elif key == ord("l"):
                toggle_graph_logging()
            elif key == ord("f"):
                toggle_temperature_unit()
            elif key == ord("v"):
                open_view()
            elif key == ord("s"):
                request_save()
            elif key == ord("c"):
                open_capture()
    except CameraError as exc:
        LOG.error("Unable to run desktop viewer: %s", exc)
        return 2
    finally:
        persist_settings()
        if last_window_size is not None:
            try:
                save_main_window_size(last_window_size)
            except OSError as exc:
                LOG.warning("Could not save main window size: %s", exc)
        try:
            graphs.close()
        except OSError as exc:
            LOG.error("Could not finalize temperature log: %s", exc)
        stop_recording()
        capture_panel.close()
        view_panel.close()
        pointer_monitor.close()
        save_dialog.close()
        try:
            hardware.restore()
        except (CameraError, ValueError) as exc:
            LOG.error("Could not restore camera settings: %s", exc)
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
