"""Native OpenCV desktop viewer with per-pixel inspection and radiometric saves."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import (
    FRAME_RATE,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    CameraError,
    TC002CDuoCamera,
    raw_temperatures,
)
from .capture_panel import CapturePanel
from .dialog_preferences import load_dialog_directory, remember_dialog_directory
from .display_awake import DisplayAwake
from .distance_calibration import DistanceCalibrator
from .emissivity_calibration import EmissivityCalibrator
from .frame_pump import CameraFramePump
from .graph_panel import GraphPanel
from .graph_settings import GRAPH_DEFAULTS, validate_graph_settings
from .graphs import (
    GRAPH_INTERVAL,
    GraphIntervalEditor,
    GraphSnapshot,
    GraphWorker,
    draw_graph_logging_control,
    graph_config_rect,
    graph_interval_rect,
    graph_log_button_rect,
    graph_reset_rect,
)
from .hardware_controls import HARDWARE_CONTROLS, HardwareControls
from .pointer import PointerMonitor
from .recording import VideoRecorder
from .reflected_calibration import ReflectedCalibrator
from .render import (
    READOUT_HEIGHT,
    RenderedThermalFrame,
    TemperatureStats,
    ThermalRenderer,
    draw_temperature_readout,
)
from .settings_preferences import load_settings, save_settings
from .spot_preferences import validate_spots
from .spots_panel import SpotsPanel
from .view_panel import ViewPanel
from .view_settings import VIEW_DEFAULTS
from .viewer_diagnostics import ViewerDiagnostics
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
    context_clicks: list[tuple[int, int]] = field(default_factory=list)
    drag_events: list[tuple[int, int, int, int]] = field(default_factory=list)

    def callback(self, event: int, x: int, y: int, flags: int, _parameter) -> None:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            self.drag_events.append((event, x, y, flags))
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONUP:
            self.x, self.y = x, y
            self.clicks.append((x, y))

        if event == cv2.EVENT_RBUTTONUP:
            self.context_clicks.append((x, y))

    def consume_drag_events(self):
        events, self.drag_events = self.drag_events, []
        return events

    def discard_click(self, x, y):
        if (x, y) in self.clicks:
            self.clicks.remove((x, y))

    def consume_context_clicks(self):
        clicks, self.context_clicks = self.context_clicks, []
        return clicks

    def consume_clicks(self) -> list[tuple[int, int]]:
        clicks, self.clicks = self.clicks, []
        return clicks


@dataclass(frozen=True)
class ToolbarLayout:
    height: int
    buttons: dict[str, tuple[int, int, int, int]]


@dataclass(frozen=True)
class GraphWindowLayout:
    canvas_size: tuple[int, int]
    camera_size: tuple[int, int]

    @classmethod
    def fit(cls, camera_size, window_size=None):
        source_width, source_height = camera_size
        width, height = window_size or (source_width * 2, source_height)
        width, height = max(2, width), max(1, height)
        scale = min((width // 2) / source_width, height / source_height)
        camera = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
        return cls((width, height), camera)

    @property
    def graph_size(self):
        return self.canvas_size[0] - self.camera_size[0], self.canvas_size[1]

    def camera_viewport(self, event_viewport=None):
        width, height = event_viewport or self.canvas_size
        return (
            self.camera_size[0] * width / self.canvas_size[0],
            self.camera_size[1] * height / self.canvas_size[1],
        )

    def graph_control_at(self, x, y, rectangle, event_viewport=None):
        if event_viewport:
            if min(event_viewport) <= 0:
                return False
            x = x * self.canvas_size[0] / event_viewport[0]
            y = y * self.canvas_size[1] / event_viewport[1]
        x0, y0, x1, y1 = rectangle(self.graph_size[0])
        return x0 <= x - self.camera_size[0] <= x1 and y0 <= y <= y1

    def compose(self, camera, graph):
        width, height = self.canvas_size
        canvas = np.zeros((height, width, 3), np.uint8)
        camera_width, camera_height = self.camera_size
        resized = cv2.resize(camera, self.camera_size, interpolation=cv2.INTER_LINEAR)
        canvas[:camera_height, :camera_width] = resized
        canvas[:, camera_width:] = graph
        return canvas


@dataclass
class SampleSpots:
    placing: bool = False
    pixels: list[tuple[int, int]] = field(default_factory=list)
    generation: int = 0

    numbers: list[int] = field(default_factory=list, init=False)
    disabled: set[int] = field(default_factory=set, init=False)
    next_number: int = field(default=1, init=False)
    names: dict[int, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.numbers = list(range(1, len(self.pixels) + 1))
        self.next_number = len(self.pixels) + 1

    @property
    def active(self):
        return [
            (number, pixel)
            for number, pixel in zip(self.numbers, self.pixels, strict=True)
            if number not in self.disabled
        ]

    def toggle(self) -> None:
        self.placing = not self.placing

    def add(self, pixel: tuple[int, int] | None) -> None:
        if self.placing and pixel is not None and pixel not in self.pixels:
            self.pixels.append(pixel)
            self.numbers.append(self.next_number)
            self.next_number += 1

    def set_enabled(self, number, enabled):
        if type(number) is not int or number not in self.numbers or type(enabled) is not bool:
            raise ValueError("Choose an existing spot and an enabled state")
        if enabled:
            self.disabled.discard(number)
        else:
            self.disabled.add(number)

    def move(self, number, pixel):
        if type(number) is not int or number not in self.numbers:
            raise ValueError("Choose an existing spot to move")
        self.pixels[self.numbers.index(number)] = pixel

    def name(self, number):
        return self.names.get(number, f"Spot {number}")

    def rename(self, number, name):
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

    def clear(self, number=None):
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

    def state(self):
        return [
            {"number": number, "enabled": number not in self.disabled, "name": self.name(number)}
            for number in self.numbers
        ]

    def saved_state(self, rotation=0, horizontal=False, vertical=False):
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

    def restore_saved_state(self, saved, rotation=0, horizontal=False, vertical=False):
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
        self._directory_kind = "image"

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
        self._directory_kind = (
            kind
            if kind in ("video", "timelapse", "temperatures")
            else "video"
            if suffix.lower() == ".mp4"
            else "temperatures"
            if suffix.lower() == ".csv"
            else "image"
        )
        remembered = load_dialog_directory(self._directory_kind, default_directory)
        directory = str(remembered) if remembered else ""
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
            selected = Path(stdout.strip())
            remember_dialog_directory(self._directory_kind, selected)
            return True, selected
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
    spots_locked: bool = False,
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
    return graph_control_at(x, y, image_width, canvas_height, viewport_size, graph_log_button_rect)


def graph_interval_at(x, y, image_width, canvas_height, viewport_size=None):
    return graph_control_at(x, y, image_width, canvas_height, viewport_size, graph_interval_rect)


def graph_control_at(x, y, image_width, canvas_height, viewport_size, rectangle):
    if viewport_size:
        if viewport_size[0] <= 0 or viewport_size[1] <= 0:
            return False
        x = round(x * image_width / viewport_size[0])
        y = round(y * canvas_height / viewport_size[1])
    x0, y0, x1, y1 = rectangle(image_width)
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
    graph_image: np.ndarray | None = None,
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
    if graph_image is not None:
        capture_image = np.concatenate((capture_image, graph_image), axis=1)
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
        measurements_valid=np.bool_(rendered.measurements_valid),
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
        "measurements_valid": rendered.measurements_valid,
        "measurement_status": rendered.measurement_status,
        "radiometric_file": data_path.name,
        "image_file": png_path.name,
        "graphs_included": graph_image is not None,
    }
    if selected_pixel is not None:
        x, y = selected_pixel
        metadata["selected_pixel"] = {
            "x": x,
            "y": y,
            "temperature_celsius": (
                round(float(rendered.temperatures_celsius[y, x]), 3)
                if np.isfinite(rendered.temperatures_celsius[y, x])
                else None
            ),
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


def _spot_hit(position, spots, image_shape, scale, viewport_size=None):
    """Find the closest enabled marker within ten displayed pixels."""
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

    def __init__(self):
        self.number = None
        self.captured = False
        self.offset = (0, 0)

    def cancel(self):
        self.number = None
        # A cancelled drag's release must still not place a new spot or click a button.

    def update(
        self, picker, spots, image_shape, scale, viewport_size=None, toolbar_height=0, locked=False
    ):
        if locked:
            self.cancel()
        for event, x, y, flags in picker.consume_drag_events():
            position = image_position_at(x, y, image_shape, viewport_size, toolbar_height)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.number = None
                self.captured = False
                if locked or position is None:
                    continue
                hit = _spot_hit(position, spots, image_shape, scale, viewport_size)
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

    def _move(self, spots, position, image_shape, scale):
        if position is None or self.number not in spots.numbers or self.number in spots.disabled:
            return
        width, height = image_shape[1] // scale, image_shape[0] // scale
        pixel = (
            max(0, min(width - 1, (position[0] + self.offset[0]) // scale)),
            max(0, min(height - 1, (position[1] + self.offset[1]) // scale)),
        )
        spots.move(self.number, pixel)


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
    active = spots.active
    anchors = [(x * scale + scale // 2, y * scale + scale // 2) for _number, (x, y) in active]
    occupied = [(x - 6, y - 6, x + 7, y + 7) for x, y in anchors]
    labels = []
    for (number, (sensor_x, sensor_y)), anchor in zip(active, anchors, strict=True):
        temperature = float(rendered.temperatures_celsius[sensor_y, sensor_x])
        if temperature_unit == "F":
            temperature = temperature * 9.0 / 5.0 + 32.0
        reading = f"{temperature:.2f}" if np.isfinite(temperature) else "--"
        text = f"{number}: {reading} {temperature_unit}"
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
    active = spots.active
    for (_number, (sensor_x, sensor_y)), (text, origin, rect) in zip(active, labels, strict=True):
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
    for _number, (sensor_x, sensor_y) in active:
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


def draw_distance_selection(image, calibration, scale):
    if not calibration.corners:
        return image
    mask = np.zeros(image.shape[:2], np.uint8)
    text_mask = np.zeros_like(mask)
    points = np.rint(np.asarray(calibration.corners) * scale).astype(np.int32)
    if len(points) > 1:
        cv2.polylines(mask, [points], len(points) == 4, 255, 1, cv2.LINE_AA)
    for index, (x, y) in enumerate(points):
        cv2.drawMarker(mask, (int(x), int(y)), 255, cv2.MARKER_CROSS, 11, 1)
        cv2.putText(
            text_mask,
            str(index + 1),
            (int(x) + 6, max(12, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            255,
            1,
            cv2.LINE_AA,
        )
    return _draw_contrasting_overlay(image, mask, text_mask)


def draw_reflector_target(image, calibration, scale, unit):
    mask = np.zeros(image.shape[:2], np.uint8)
    text_mask = np.zeros_like(mask)
    center = (
        (image.shape[1] // scale // 2) * scale + scale // 2,
        (image.shape[0] // scale // 2) * scale + scale // 2,
    )
    radius = calibration.RADIUS * scale
    cv2.circle(mask, center, radius, 255, 1, cv2.LINE_AA)
    label = "Reflector"
    if calibration.measured_celsius is not None:
        value = calibration.measured_celsius
        if unit == "F":
            value = value * 1.8 + 32
        label += f" {value:.1f} {unit}"
    cv2.putText(
        text_mask,
        label,
        (max(2, center[0] - 55), max(15, center[1] - radius - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        255,
        1,
        cv2.LINE_AA,
    )
    return _draw_contrasting_overlay(image, mask, text_mask)


def draw_emissivity_point(image, calibration, scale, unit):
    if calibration.point is None:
        return image
    x, y = (coordinate * scale + scale // 2 for coordinate in calibration.point)
    mask = np.zeros(image.shape[:2], np.uint8)
    text_mask = np.zeros_like(mask)
    cv2.drawMarker(mask, (x, y), 255, cv2.MARKER_CROSS, 11, 1)
    temperature = calibration.measured_celsius
    label = "Reference"
    if temperature is not None:
        display = temperature * 1.8 + 32 if unit == "F" else temperature
        label = f"Ref {display:.2f} {unit}"
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = max(2, min(x + 10, image.shape[1] - width - 2))
    label_y = y - 10 if y > height + 12 else y + height + 12
    label_y = min(label_y, image.shape[0] - baseline - 2)
    cv2.putText(
        text_mask, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA
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
    dragging_spot: bool = False,
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
    if dragging_spot or (
        spots is not None
        and _spot_hit(position, spots, image.shape, scale, viewport_size) is not None
    ):
        return _draw_contrasting_overlay(image, mask, text_mask), (sensor_x, sensor_y)
    reading = f"{temperature:.2f}" if np.isfinite(temperature) else "--"
    text = f"({sensor_x}, {sensor_y}) {reading} {temperature_unit}"
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
        ("P / Add spots  Toggle spot placement", (210, 215, 225)),
        ("Camera Hardware ambient and image controls", (210, 215, 225)),
        ("S            Save image data", (210, 215, 225)),
        ("C            Open Capture controls", (210, 215, 225)),
        ("O            Rotate 90 degrees clockwise", (210, 215, 225)),
        ("F            Toggle metric / imperial", (210, 215, 225)),
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
        default=None,
        help="timelapse frames per minute (default: remembered rate, or 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="initial directory for the Save dialog",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--diagnostics", type=Path, help="Write UI timings and stall thread stacks")
    parser.add_argument(
        "--usb-queue-depth", type=int, choices=(0, *range(2, 129)),
        default=32 if sys.platform.startswith("linux") else 0,
        help="Queued USB requests (2-128; 0: synchronous; default: 32 on Linux, 0 elsewhere)",
    )
    args = parser.parse_args(argv)
    if args.timelapse_fpm is not None and not 1 <= args.timelapse_fpm <= TIMELAPSE_MAX_FPM:
        parser.error(f"--timelapse-fpm must be between 1 and {TIMELAPSE_MAX_FPM}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    saved_settings = load_settings()
    remembered_hardware = saved_settings.get("hardware", {}).copy()
    advanced_auto = saved_settings.get("advanced_auto", True)
    auto_calibrate = saved_settings.get("auto_calibrate", False)
    calibration_available = False
    startup_calibration_pending = True
    show_graph = saved_settings.get("show_graph", False)
    graph_interval = saved_settings.get("graph_interval", GRAPH_INTERVAL)
    graph_interval_editor = GraphIntervalEditor(graph_interval)
    graph_settings = saved_settings.get("graph_settings", GRAPH_DEFAULTS.copy())
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
    camera.usb_queue_depth = args.usb_queue_depth
    display_awake = DisplayAwake()
    picker = MousePicker()
    spots = SampleSpots()
    if "spots" in saved_settings:
        spots.restore_saved_state(
            saved_settings["spots"],
            renderer.rotation,
            renderer.mirror_horizontal,
            renderer.mirror_vertical,
        )
    last_saved_spots = spots.saved_state(
        renderer.rotation, renderer.mirror_horizontal, renderer.mirror_vertical
    )
    ambient_input_celsius = saved_settings.get("ambient_input_celsius")
    spot_drag = SpotDrag()
    distance_calibration = DistanceCalibrator(saved_settings.get("distance_calibration"))
    pointer_monitor = PointerMonitor(WINDOW_NAME)
    save_dialog = LinuxSaveDialog() if sys.platform.startswith("linux") else MacSaveDialog()
    recorder = VideoRecorder()
    capture_panel = CapturePanel()
    view_panel = ViewPanel()
    graph_panel = GraphPanel()
    spots_panel = SpotsPanel()
    hardware = HardwareControls(camera)
    emissivity_calibration = EmissivityCalibrator(
        hardware, saved_settings.get("emissivity_calibration")
    )
    reflected_calibration = ReflectedCalibrator(
        hardware, saved_settings.get("reflected_calibration")
    )
    graphs = GraphWorker()
    graphs.set_interval(graph_interval)
    graphs.configure(graph_settings)
    actual_image_source = renderer.image_source
    pending_save_kind: str | None = None
    capture_cursor = saved_settings.get("capture_cursor", False)
    capture_graphs = saved_settings.get("capture_graphs", False)
    recording_graphs = False
    timelapse_fpm = (
        args.timelapse_fpm
        if args.timelapse_fpm is not None
        else saved_settings.get("timelapse_fpm", TIMELAPSE_DEFAULT_FPM)
    )
    status_message = "Ready"
    status_until = 0.0
    status = "Ready"
    show_instructions = False
    last_selected: tuple[int, int] | None = None

    def persist_settings() -> None:
        nonlocal last_saved_spots
        last_saved_spots = spots.saved_state(
            renderer.rotation, renderer.mirror_horizontal, renderer.mirror_vertical
        )
        try:
            save_settings(
                {
                    "display": {name: getattr(renderer, name) for name in VIEW_DEFAULTS},
                    "hardware": remembered_hardware,
                    "ambient_input_celsius": ambient_input_celsius,
                    "spots": last_saved_spots,
                    "rotation": renderer.rotation,
                    "advanced_auto": advanced_auto,
                    "auto_calibrate": auto_calibrate,
                    "show_graph": show_graph,
                    "graph_interval": graph_interval,
                    "graph_settings": graph_settings,
                    "capture_cursor": capture_cursor,
                    "capture_graphs": capture_graphs,
                    "timelapse_fpm": timelapse_fpm,
                    "distance_calibration": (
                        distance_calibration.reference.as_dict()
                        if distance_calibration.reference
                        else None
                    ),
                    "emissivity_calibration": emissivity_calibration.reference,
                    "reflected_calibration": reflected_calibration.reference,
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
        spot_drag.cancel()
        show_graph = not show_graph
        if not show_graph:
            graph_interval_editor.text = None
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
        if reflected_calibration.active:
            notify("Close reflected-temperature calibration before starting logging.")
            return
        if emissivity_calibration.active:
            notify("Close emissivity calibration before starting temperature logging.")
            return
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
                    spot_drag.cancel()
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
        if graphs.logging or emissivity_calibration.running or reflected_calibration.running:
            notify("Stop logging or calibration measurement before changing settings.")
            return
        if emissivity_calibration.active:
            emissivity_calibration.cancel()
        if reflected_calibration.active:
            reflected_calibration.cancel()
        spot_drag.cancel()
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
        if distance_calibration.corners or distance_calibration.selecting:
            distance_calibration.cancel()
        persist_settings()
        picker.x = picker.y = None
        last_selected = None

    def set_view_setting(name: str, value: object) -> None:
        nonlocal last_selected
        if graphs.logging or emissivity_calibration.running or reflected_calibration.running:
            notify("Stop logging or calibration measurement before changing settings.")
            return
        previous = renderer.view_settings()
        if name == "palette_source" and value == "camera":
            palette = hardware.state().get("palette", {})
            if not palette.get("available", False):
                raise ValueError("Camera palette control is unavailable")
            hardware.set("palette", palette["value"], True)
            remembered_hardware["palette"] = hardware.state()["palette"]["value"]
        renderer.set_view_setting(name, value)
        if name == "color_palette":
            renderer.set_view_setting("palette_source", "app")
        persist_settings()
        if name in ("mirror_horizontal", "mirror_vertical") and previous[name] != value:
            spot_drag.cancel()
            if emissivity_calibration.active:
                emissivity_calibration.cancel()
            if reflected_calibration.active:
                reflected_calibration.cancel()
            width, height = (
                (SENSOR_WIDTH, SENSOR_HEIGHT)
                if renderer.rotation in (0, 180)
                else (SENSOR_HEIGHT, SENSOR_WIDTH)
            )
            spots.mirror(width, height, name == "mirror_horizontal", name == "mirror_vertical")
            if distance_calibration.corners or distance_calibration.selecting:
                distance_calibration.cancel()
            picker.x = picker.y = None
            last_selected = None

    def spots_state():
        return {
            "spots": spots.state(),
            "placing": spots.placing,
            "calibration_available": calibration_available and not renderer.measurement_status,
            "locked": bool(
                graphs.logging
                or pending_save_kind == "graph_log"
                or emissivity_calibration.active
                or reflected_calibration.active
                or distance_calibration.selecting
            ),
        }

    def toggle_spots() -> None:
        if spots_state()["locked"]:
            notify("Stop logging or close calibration before changing measuring spots.")
            return
        spots.toggle()

    def toggle_temperature_unit() -> None:
        set_view_setting("temperature_unit", "F" if renderer.temperature_unit == "C" else "C")

    def view_state() -> dict:
        message = "Camera preview" if actual_image_source == "preview" else "Raw thermal image"
        if renderer.image_source == "preview" and actual_image_source == "raw":
            message = (
                "App colors from raw thermal data"
                if renderer.palette_source == "app" and renderer.camera_color
                else "Camera preview unavailable; showing the raw thermal image."
            )
        if hardware.error:
            message = hardware.error
        else:
            message += " · Camera temperatures (approximate)"
        if renderer.measurement_status:
            message += f" · {renderer.measurement_status}"
        if renderer.upsampling != "off":
            if not renderer.enhancement_amount:
                message += " · Enhancement amount 0 (original image)"
            elif renderer.upsampler.error:
                message += f" · {renderer.upsampler.error}; showing unenhanced image"
            else:
                algorithm = (
                    "TIDY denoise"
                    if renderer.upsampling == "tidy"
                    else "DnCNN denoise"
                    if renderer.upsampling == "dncnn-gray-blind"
                    else "Anime4K09 2×"
                    if renderer.upsampling == "anime4k09"
                    else "ACNet 2×"
                )
                message += f" · {algorithm} · {renderer.upsampler.elapsed_ms:.0f} ms"
        if graphs.logging:
            message += " · Settings locked while logging"
        elif reflected_calibration.running:
            message += " · Settings locked while measuring reflected temperature"
        elif emissivity_calibration.running:
            message += " · Settings locked while fitting emissivity"
        ui_hardware = hardware.state()
        if ambient_input_celsius is not None and "ambient" in ui_hardware:
            ui_hardware = {
                **ui_hardware,
                "ambient": {**ui_hardware["ambient"], "value": ambient_input_celsius},
            }
        return {
            **renderer.view_settings(),
            "hardware": ui_hardware,
            "color_source": (
                "camera"
                if actual_image_source == "preview"
                and renderer.camera_preview
                and renderer.camera_color
                else "app"
            ),
            "advanced_auto": advanced_auto,
            "auto_calibrate": auto_calibrate,
            "settings_locked": graphs.logging
            or emissivity_calibration.running
            or reflected_calibration.running,
            "distance_calibration": distance_calibration.state(),
            "emissivity_calibration": emissivity_calibration.state(),
            "reflected_calibration": reflected_calibration.state(),
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
            "capture_graphs": capture_graphs,
            "frames_per_minute": timelapse_fpm,
            "max_fpm": TIMELAPSE_MAX_FPM,
            "status": status,
        }

    def graph_config_state():
        return {
            "settings": graph_settings,
            "locked": graphs.logging or pending_save_kind == "graph_log",
        }

    def open_capture() -> None:
        try:
            capture_panel.open(capture_state())
        except OSError as exc:
            notify(f"Could not open Capture: {exc}")

    print(
        "Mouse: inspect a pixel | "
        "p: toggle spot placement | s: save image data | c: Capture controls | "
        "o: rotate | f: metric/imperial | v: Camera controls | g: show/hide graph | l: log temperatures | Space: controls | q/Esc: quit"
    )

    def initialize_hardware() -> bool:
        nonlocal calibration_available
        try:
            hardware.load()
            hardware.error = ""
            for name, value in remembered_hardware.items():
                try:
                    hardware.set(name, value, True)
                except (CameraError, ValueError, TypeError) as exc:
                    hardware.error = f"Could not restore saved {name}: {exc}"
                    LOG.warning("%s", hardware.error)
        except CameraError as exc:
            hardware.error = str(exc)
            LOG.warning("Could not read hardware settings: %s", exc)
            return False
        try:
            hardware.set_auto_calibrate(auto_calibrate)
            calibration_available = True
        except CameraError as exc:
            hardware.error = f"Could not set Auto calibrate: {exc}"
            LOG.warning("%s", hardware.error)
        return True

    try:
        diagnostics = ViewerDiagnostics(args.diagnostics)
    except OSError as exc:
        LOG.error("Could not open diagnostic log: %s", exc)
        return 2
    frame_pump = None
    try:
        diagnostics.stage("camera_open")
        camera.stream_observer = diagnostics.stream if args.diagnostics else None
        camera.rejected_frame_observer = diagnostics.rejected_frame if args.diagnostics else None
        camera.open()
        diagnostics.stage("hardware_setup")
        display_awake.start()
        hardware_setup_pending = not initialize_hardware()
        hardware_retry_deadline = None
        next_hardware_retry = 0.0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, picker.callback)
        set_black_window_backgrounds(WINDOW_NAME)
        initial_window_size_set = False
        frame_pump = CameraFramePump(camera)
        last_frame = None
        last_frame_at = None
        for frame in diagnostics.frames(frame_pump):
            fresh_frame = frame is not None
            if fresh_frame:
                last_frame = frame
                last_frame_at = time.monotonic()
            elif last_frame is None:
                waiting = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(
                    waiting,
                    "Waiting for camera...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (230, 230, 230),
                    1,
                )
                cv2.imshow(WINDOW_NAME, waiting)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            else:
                frame = last_frame
            if hardware_setup_pending:
                now = time.monotonic()
                if hardware_retry_deadline is None:
                    hardware_retry_deadline = now + 30
                if now >= next_hardware_retry:
                    diagnostics.stage("hardware_setup_retry")
                    hardware_setup_pending = not initialize_hardware()
                    next_hardware_retry = now + 1
                    if not hardware_setup_pending:
                        notify("Camera settings loaded after startup")
                    elif now >= hardware_retry_deadline:
                        hardware_setup_pending = False
                        notify(
                            "Camera settings unavailable after startup retries; reconnect camera."
                        )
                    diagnostics.stage("controls")
            display_error = display_awake.check()
            if display_error:
                notify(display_error)
            for command in spots_panel.poll():
                if command.get("action") == "error":
                    notify(f"Spot menu failed: {command.get('message', '')}")
                    continue
                if spots_state()["locked"]:
                    notify("Stop logging or close calibration before changing measuring spots.")
                    continue
                try:
                    action = command["action"]
                    if action == "calibrate_now":
                        if not spots_state()["calibration_available"]:
                            raise ValueError(
                                "Camera calibration is unavailable or already in progress"
                            )
                        hardware.calibrate_now()
                        renderer._recover_measurements = True
                        notify("Calibration requested")
                    elif action == "placing":
                        if type(command["enabled"]) is not bool:
                            raise ValueError("Placement state must be a boolean")
                        spots.placing = command["enabled"]
                    elif action == "enable":
                        spots.set_enabled(command["spot"], command["enabled"])
                    elif action == "rename":
                        spots.rename(command["spot"], command["name"])
                    elif action == "clear":
                        spots.clear(command["spot"])
                    elif action == "clear_all":
                        spots.clear()
                    else:
                        raise ValueError("Unknown spot menu operation")
                except (CameraError, KeyError, ValueError, TypeError) as exc:
                    notify(f"Spot change rejected: {exc}")
            for command in graph_panel.poll():
                try:
                    if command.get("action") == "error":
                        notify(f"Graph configuration failed: {command.get('message', '')}")
                    elif command.get("action") == "settings":
                        if graph_config_state()["locked"]:
                            raise ValueError("Stop logging before changing graph settings")
                        settings = validate_graph_settings(command["settings"])
                        graphs.configure(settings)
                        graph_settings = settings
                        persist_settings()
                except (ValueError, TypeError, KeyError) as exc:
                    notify(f"Graph settings rejected: {exc}")
            for command in view_panel.poll():
                if (
                    graphs.logging
                    or emissivity_calibration.running
                    or reflected_calibration.running
                ) and command.get("action") in (
                    "setting",
                    "hardware",
                    "advanced_auto",
                    "auto_calibrate",
                    "restore_hardware",
                    "reset",
                    "distance_calibration",
                ):
                    notify("Stop logging or calibration measurement before changing settings.")
                    continue
                if (graphs.logging or reflected_calibration.running) and command.get(
                    "action"
                ) == "emissivity_calibration":
                    notify("Stop temperature logging before emissivity calibration.")
                    continue
                try:
                    if reflected_calibration.active and command.get("action") in (
                        "hardware",
                        "distance_calibration",
                        "emissivity_calibration",
                        "restore_hardware",
                        "reset",
                    ):
                        reflected_calibration.cancel()
                    if command.get("action") == "setting":
                        set_view_setting(command["name"], command["value"])
                    elif command.get("action") == "auto_calibrate":
                        if emissivity_calibration.active:
                            emissivity_calibration.cancel()
                        if reflected_calibration.active:
                            reflected_calibration.cancel()
                        hardware.set_auto_calibrate(command["value"])
                        auto_calibrate = command["value"]
                        calibration_available = True
                        persist_settings()
                        hardware.error = ""
                    elif command.get("action") == "hardware":
                        if emissivity_calibration.active:
                            emissivity_calibration.cancel()
                        value = command["value"]
                        if command["name"] == "ambient":
                            spec = HARDWARE_CONTROLS["ambient"]
                            if (
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or not spec.minimum <= value <= spec.maximum
                            ):
                                raise ValueError(
                                    "Ambient temperature is outside the supported range"
                                )
                            value = round(
                                spec.minimum
                                + round((value - spec.minimum) / spec.step) * spec.step,
                                2,
                            )
                        hardware.set(command["name"], value, command["enabled"])
                        if command["name"] == "ambient":
                            ambient_input_celsius = command["value"] if command["enabled"] else None
                        if command["name"] == "palette" and command["enabled"]:
                            renderer.set_view_setting("palette_source", "camera")
                        if command["enabled"]:
                            remembered_hardware[command["name"]] = hardware.state()[
                                command["name"]
                            ]["value"]
                        else:
                            remembered_hardware.pop(command["name"], None)
                        persist_settings()
                        hardware.error = ""
                        renderer._average_raw = None
                    elif command.get("action") == "distance_calibration":
                        if emissivity_calibration.active:
                            emissivity_calibration.cancel()
                        operation = command["operation"]
                        if operation in ("select", "measure"):
                            distance_calibration.begin(
                                "reference" if operation == "select" else "measure"
                            )
                            notify(distance_calibration.message)
                        elif operation == "cancel":
                            distance_calibration.cancel()
                        elif operation == "save":
                            distance_calibration.save_reference(
                                command["side_m"], command["distance_m"]
                            )
                            persist_settings()
                            notify(distance_calibration.message)
                        elif operation == "apply":
                            estimated = distance_calibration.estimated_m
                            if estimated is None or not 0.3 <= estimated <= 99:
                                raise ValueError(
                                    "Measure a valid square within the camera's distance range first"
                                )
                            value = round(estimated, 2)
                            hardware.set("distance", value, True)
                            distance_calibration.corners = []
                            remembered_hardware["distance"] = hardware.state()["distance"]["value"]
                            renderer._average_raw = None
                            hardware.error = ""
                            persist_settings()
                            notify("Estimated distance applied to the camera")
                        elif operation == "clear":
                            distance_calibration.clear()
                            persist_settings()
                        else:
                            raise ValueError("Unknown distance calibration operation")
                    elif command.get("action") == "reflected_calibration":
                        operation = command["operation"]
                        if (
                            graphs.logging
                            or emissivity_calibration.running
                            or pending_save_kind == "graph_log"
                        ):
                            raise ValueError(
                                "Stop logging or emissivity fitting before reflected-temperature calibration"
                            )
                        if operation == "cancel":
                            reflected_calibration.cancel()
                        elif operation == "select":
                            emissivity_calibration.cancel()
                            distance_calibration.cancel()
                            reflected_calibration.begin()
                            picker.x = picker.y = None
                        elif operation == "measure":
                            reflected_calibration.start(time.monotonic())
                        elif operation == "apply":
                            remembered_hardware["reflected"] = reflected_calibration.apply()
                            persist_settings()
                        else:
                            raise ValueError("Unknown reflected-temperature calibration operation")
                        renderer._average_raw = None
                        notify(reflected_calibration.message)
                    elif command.get("action") == "emissivity_calibration":
                        operation = command["operation"]
                        if operation == "cancel":
                            emissivity_calibration.cancel()
                        elif operation == "select":
                            if pending_save_kind == "graph_log":
                                raise ValueError("Finish the pending logging dialog first")
                            distance_calibration.cancel()
                            emissivity_calibration.begin()
                            picker.x = picker.y = None
                        elif operation == "fit":
                            if pending_save_kind == "graph_log":
                                raise ValueError("Finish the pending logging dialog first")
                            emissivity_calibration.start(command["known_celsius"], time.monotonic())
                        elif operation == "apply":
                            remembered_hardware["emissivity"] = emissivity_calibration.apply()
                            persist_settings()
                        else:
                            raise ValueError("Unknown emissivity calibration operation")
                        renderer._average_raw = None
                        hardware.error = ""
                        notify(emissivity_calibration.message)
                    elif command.get("action") == "advanced_auto":
                        if not isinstance(command["value"], bool):
                            raise ValueError("Advanced / Auto must be a boolean")
                        advanced_auto = command["value"]
                        persist_settings()
                    elif command.get("action") == "restore_hardware":
                        if emissivity_calibration.active:
                            emissivity_calibration.cancel()
                        hardware.restore()
                        remembered_hardware.clear()
                        ambient_input_celsius = None
                        persist_settings()
                        hardware.error = ""
                        renderer._average_raw = None
                    elif command.get("action") == "reset":
                        for name, value in VIEW_DEFAULTS.items():
                            if name == "palette_source":
                                renderer.set_view_setting(name, value)
                            else:
                                set_view_setting(name, value)
                        persist_settings()
                    elif command.get("action") == "error":
                        notify(f"Camera window failed: {command.get('message', '')}")
                except (KeyError, ValueError, TypeError, CameraError) as exc:
                    hardware.error = f"Camera setting rejected: {exc}"
                    notify(hardware.error)
            for command in capture_panel.poll():
                previous_capture_preferences = (capture_cursor, capture_graphs, timelapse_fpm)
                action = command.get("action")
                if action == "error":
                    notify(f"Capture window failed: {command.get('message', '')}")
                elif action == "rate":
                    set_timelapse_fpm(int(command["value"]))
                elif action == "graphs":
                    if not recorder.is_recording and pending_save_kind not in (
                        "video",
                        "timelapse",
                    ):
                        capture_graphs = bool(command["value"])
                elif action == "cursor":
                    capture_cursor = bool(command["value"])
                elif action == "image":
                    request_save()
                elif action in ("video", "timelapse"):
                    set_timelapse_fpm(int(command["frames_per_minute"]))
                    capture_cursor = bool(command["capture_cursor"])
                    if not recorder.is_recording and pending_save_kind not in (
                        "video",
                        "timelapse",
                    ):
                        capture_graphs = bool(command.get("capture_graphs", capture_graphs))
                    request_save(action)
                if (capture_cursor, capture_graphs, timelapse_fpm) != previous_capture_preferences:
                    persist_settings()
            renderer.camera_preview = hardware.preview_active
            renderer.camera_color = "palette" in hardware.enabled
            renderer.hardware_settings = hardware.state() if hardware.original else {}
            ambient = renderer.hardware_settings.get("ambient", {})
            renderer.ambient_celsius = ambient.get("value") if ambient.get("available") else None
            diagnostics.stage("render")
            rendered = renderer.render_detailed(frame, update_measurements=fresh_frame)
            if not fresh_frame and time.monotonic() - last_frame_at >= 0.5:
                rendered = replace(
                    rendered,
                    measurements_valid=False,
                    measurement_status="Waiting for camera frame; readings held",
                )
                renderer.measurement_status = rendered.measurement_status
                renderer._recover_measurements = True
            diagnostics.measurements(rendered.measurement_status)
            diagnostics.stage("calibration_and_layout")
            if (
                startup_calibration_pending
                and calibration_available
                and rendered.measurements_valid
            ):
                startup_calibration_pending = False
                try:
                    hardware.calibrate_now()
                    renderer._recover_measurements = True
                    notify("Startup calibration requested")
                except CameraError as exc:
                    hardware.error = f"Could not run startup calibration: {exc}"
                    LOG.warning("%s", hardware.error)
                    notify(hardware.error)
            actual_image_source = rendered.image_source
            try:
                if emissivity_calibration.active and rendered.measurements_valid:
                    previous_reference = emissivity_calibration.reference
                    emissivity_calibration.update(
                        raw_temperatures(rendered.raw_counts, offset=50), time.monotonic()
                    )
                    if previous_reference != emissivity_calibration.reference:
                        persist_settings()
            except (CameraError, ValueError) as exc:
                notify(f"Emissivity calibration failed: {exc}")
                try:
                    emissivity_calibration.cancel()
                except CameraError as restore_exc:
                    hardware.error = f"Could not restore emissivity: {restore_exc}"
            try:
                previous_reference = reflected_calibration.reference
                if rendered.measurements_valid:
                    reflected_calibration.update(
                        raw_temperatures(rendered.raw_counts, offset=50), time.monotonic()
                    )
                if previous_reference != reflected_calibration.reference:
                    persist_settings()
                    renderer._average_raw = None
                    notify(reflected_calibration.message)
            except (CameraError, ValueError) as exc:
                hardware.error = f"Reflected-temperature calibration failed: {exc}"
                notify(hardware.error)
                try:
                    reflected_calibration.cancel()
                except CameraError as restore_exc:
                    hardware.error = (
                        f"Could not restore reflector measurement settings: {restore_exc}"
                    )
            if (
                rendered.measurements_valid
                and not graphs.logging
                and distance_calibration.update(rendered.temperatures_celsius)
            ):
                notify(distance_calibration.message)
            layout = toolbar_layout(rendered.image.shape[1])
            if not initial_window_size_set:
                width, height = saved_window_size or (
                    rendered.image.shape[1] * (2 if show_graph else 1),
                    rendered.image.shape[0] + layout.height,
                )
                requested_window_size = (width, height)
                cv2.resizeWindow(WINDOW_NAME, width, height)
                initial_window_size_set = True
            event_viewport = mouse_viewport_size()
            current_size = window_resize_size(WINDOW_NAME)
            if current_size is not None:
                last_window_size = current_size
            # A popup covering the window center can temporarily prevent native
            # size detection. Keep the canvas and mouse mapping at the last size.
            graph_layout = (
                GraphWindowLayout.fit(
                    (rendered.image.shape[1], rendered.image.shape[0] + layout.height),
                    last_window_size,
                )
                if show_graph
                else None
            )
            viewport_size = (
                graph_layout.camera_viewport(event_viewport) if graph_layout else event_viewport
            )
            spot_drag.update(
                picker,
                spots,
                rendered.image.shape,
                renderer.scale,
                viewport_size,
                layout.height,
                locked=spots_state()["locked"],
            )
            pointer_toolbar_height = layout.height
            pointer_image_height = rendered.image.shape[0]
            if graph_layout:
                pointer_toolbar_height = round(
                    layout.height
                    * graph_layout.camera_size[1]
                    / (rendered.image.shape[0] + layout.height)
                )
                pointer_image_height = graph_layout.canvas_size[1] - pointer_toolbar_height
            display, selected = (
                (rendered.image, None)
                if emissivity_calibration.active or reflected_calibration.active
                else draw_picker(
                    rendered,
                    picker,
                    renderer.scale,
                    viewport_size=viewport_size,
                    toolbar_height=layout.height,
                    temperature_unit=renderer.temperature_unit,
                    spots=spots,
                    dragging_spot=spot_drag.number is not None,
                    pointer_over_image=pointer_monitor.over_image(
                        pointer_image_height, pointer_toolbar_height
                    ),
                )
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
                            emissivity_calibration.cancel()
                            path = graphs.start_logging(save_path)
                            spot_drag.cancel()
                            if distance_calibration.selecting:
                                distance_calibration.cancel()
                            _restore_user_ownership([path])
                            notify(f"Logging temperatures to {path.name}")
                        elif kind in ("video", "timelapse"):
                            recording_graphs = capture_graphs and show_graph
                            recorder.start(
                                save_path,
                                (
                                    rendered.image.shape[0] + READOUT_HEIGHT,
                                    rendered.image.shape[1] * (2 if recording_graphs else 1),
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
                                graph_image=(
                                    graphs.image(
                                        (
                                            rendered.image.shape[1],
                                            rendered.image.shape[0] + READOUT_HEIGHT,
                                        ),
                                        resize=True,
                                    )
                                    if capture_graphs and show_graph
                                    else None
                                ),
                            )
                            notify(f"Saved {saved[0].name}")
                    except (OSError, ValueError, cv2.error) as exc:
                        notify(f"Capture failed: {exc}")
                else:
                    notify("Save cancelled")
            # Captures omit application controls; the graph pane is optional.
            # The cursor sampler remains visible locally even when capture is off.
            if recorder.is_recording:
                recording_view = (
                    rendered.image
                    if emissivity_calibration.active or reflected_calibration.active
                    else draw_sample_spots(
                        display if capture_cursor else rendered.image,
                        rendered,
                        spots,
                        renderer.scale,
                        renderer.temperature_unit,
                    )
                )
                try:
                    recording_image = draw_temperature_readout(
                        recording_view,
                        rendered.stats,
                        renderer.ambient_celsius,
                        renderer.temperature_unit,
                    )
                    if recording_graphs:
                        graph_image = (
                            graphs.image(
                                (recording_image.shape[1], recording_image.shape[0]), resize=True
                            )
                            if show_graph
                            else np.zeros_like(recording_image)
                        )
                        recording_image = np.concatenate((recording_image, graph_image), axis=1)
                    recorder.write(recording_image)
                except (OSError, cv2.error) as exc:
                    stop_recording()
                    notify(f"Recording failed: {exc}")
            if rendered.measurement_status:
                status = rendered.measurement_status
            elif reflected_calibration.active:
                display = draw_reflector_target(
                    display, reflected_calibration, renderer.scale, renderer.temperature_unit
                )
            elif not emissivity_calibration.active:
                display = draw_sample_spots(
                    display, rendered, spots, renderer.scale, renderer.temperature_unit
                )
                display = draw_distance_selection(display, distance_calibration, renderer.scale)
            else:
                display = draw_emissivity_point(
                    display, emissivity_calibration, renderer.scale, renderer.temperature_unit
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
            elif reflected_calibration.active:
                status = reflected_calibration.message
            elif emissivity_calibration.active:
                status = emissivity_calibration.message
            elif distance_calibration.selecting:
                status = distance_calibration.message
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
                settings_locked=graphs.logging
                or emissivity_calibration.running
                or reflected_calibration.running,
                spots_locked=spots_state()["locked"],
            )

            # Capture spot values in this frame's orientation, before click actions can rotate
            # coordinates or add/clear spots. The next frame supplies any changed selection.
            graph_spots = (
                tuple(
                    ((spots.generation, number - 1), float(rendered.temperatures_celsius[y, x]))
                    for number, (x, y) in spots.active
                )
                if show_graph
                else ()
            )
            quit_requested = False
            for click_x, click_y in picker.consume_context_clicks():
                if (
                    image_position_at(
                        click_x, click_y, rendered.image.shape, viewport_size, layout.height
                    )
                    is not None
                ):
                    try:
                        spots_panel.open(spots_state())
                    except OSError as exc:
                        notify(f"Could not open spot menu: {exc}")
            for click_x, click_y in picker.consume_clicks():
                if (
                    show_graph
                    and graph_layout
                    and graph_layout.graph_control_at(
                        click_x, click_y, graph_reset_rect, event_viewport
                    )
                ):
                    graph_interval_editor.text = None
                    if graph_config_state()["locked"]:
                        notify("Stop graph logging before clearing chart data.")
                    else:
                        graphs.clear_history()
                        notify("Chart data cleared")
                    continue
                if (
                    show_graph
                    and graph_layout
                    and graph_layout.graph_control_at(
                        click_x, click_y, graph_config_rect, event_viewport
                    )
                ):
                    graph_interval_editor.text = None
                    try:
                        graph_panel.open(graph_config_state())
                    except OSError as exc:
                        notify(f"Could not open graph configuration: {exc}")
                    continue
                if (
                    show_graph
                    and graph_layout
                    and graph_layout.graph_control_at(
                        click_x, click_y, graph_interval_rect, event_viewport
                    )
                ):
                    if graphs.logging or pending_save_kind == "graph_log":
                        notify("Stop graph logging before changing the update interval.")
                    else:
                        graph_interval_editor.begin()
                        notify("Update interval in seconds: Enter to apply, Esc to cancel.")
                    continue
                graph_interval_editor.text = None
                if (
                    show_graph
                    and graph_layout
                    and graph_layout.graph_control_at(
                        click_x, click_y, graph_log_button_rect, event_viewport
                    )
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
                elif action is None and reflected_calibration.active:
                    continue
                elif action is None and emissivity_calibration.active:
                    if emissivity_calibration.selecting and rendered.measurements_valid:
                        position = image_position_at(
                            click_x, click_y, rendered.image.shape, viewport_size, layout.height
                        )
                        if position is not None:
                            pixel = (position[0] // renderer.scale, position[1] // renderer.scale)
                            emissivity_calibration.select_point(
                                pixel,
                                rendered.temperatures_celsius.shape[::-1],
                                float(
                                    raw_temperatures(
                                        rendered.raw_counts[pixel[1], pixel[0]], offset=50
                                    )
                                ),
                            )
                            notify(emissivity_calibration.message)
                elif action is None and distance_calibration.selecting:
                    continue
                elif action is None and spots.placing:
                    if spots_state()["locked"]:
                        notify("Stop logging or close calibration before changing measuring spots.")
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
            if (
                spots.saved_state(
                    renderer.rotation, renderer.mirror_horizontal, renderer.mirror_vertical
                )
                != last_saved_spots
            ):
                persist_settings()
            view_panel.update(view_state())
            spots_panel.update(spots_state())
            graph_panel.update(graph_config_state())
            if show_graph:
                diagnostics.stage("graphs")
                if graph_layout is None:
                    graph_layout = GraphWindowLayout.fit(
                        (display.shape[1], display.shape[0]),
                        window_resize_size(WINDOW_NAME) or last_window_size,
                    )
                graph_size = graph_layout.graph_size
                graphs.submit(
                    GraphSnapshot(
                        stats=(
                            rendered.stats.minimum,
                            rendered.stats.average,
                            rendered.stats.maximum,
                            rendered.stats.center,
                        ),
                        spots=graph_spots,
                        spot_names=tuple(
                            ((spots.generation, number - 1), spots.name(number))
                            for number, _point in spots.active
                        ),
                        size=graph_size,
                        unit=renderer.temperature_unit,
                        measurements_valid=rendered.measurements_valid,
                    )
                )
                graph_image = graphs.image(graph_size, resize=True).copy()
                draw_graph_logging_control(
                    graph_image,
                    graphs.logging,
                    pending_save_kind == "graph_log",
                    interval=graph_interval,
                    edit_text=graph_interval_editor.text,
                )
                display = graph_layout.compose(display, graph_image)
            else:
                graphs.pause()
            diagnostics.stage("imshow")
            cv2.imshow(WINDOW_NAME, display)
            diagnostics.stage("waitKey")
            key = cv2.waitKey(1) & 0xFF
            diagnostics.stage("window_geometry")
            current_size = window_resize_size(WINDOW_NAME)
            if current_size is not None:
                last_window_size = current_size
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
                    break
            except cv2.error:
                if last_window_size is not None:
                    break
            if graph_interval_editor.text is not None:
                try:
                    updated_interval = graph_interval_editor.key(key)
                    if updated_interval is not None:
                        if graphs.logging or pending_save_kind == "graph_log":
                            notify("Stop graph logging before changing the update interval.")
                        else:
                            graphs.set_interval(updated_interval)
                            graph_interval = updated_interval
                            graph_interval_editor.value = updated_interval
                            persist_settings()
                            notify(f"Graph update interval: {graph_interval:g} seconds.")
                except ValueError:
                    notify("Enter an update interval between 0.1 and 60 seconds.")
                continue
            diagnostics.stage("keyboard_controls")
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
        diagnostics.stage("shutdown")
        try:
            reflected_calibration.cancel()
        except CameraError as exc:
            LOG.error("Could not restore reflector measurement settings: %s", exc)
        try:
            emissivity_calibration.cancel()
        except CameraError as exc:
            LOG.error("Could not restore pre-calibration emissivity: %s", exc)
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
        graph_panel.close()
        spots_panel.close()
        pointer_monitor.close()
        save_dialog.close()
        try:
            hardware.restore()
        except (CameraError, ValueError) as exc:
            LOG.error("Could not restore camera settings: %s", exc)
        try:
            hardware.restore_auto_calibrate()
        except CameraError as exc:
            LOG.error("Could not restore automatic camera calibration: %s", exc)
        display_awake.close()
        if frame_pump is not None:
            frame_pump.close()
        camera.close()
        cv2.destroyAllWindows()
        diagnostics.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
