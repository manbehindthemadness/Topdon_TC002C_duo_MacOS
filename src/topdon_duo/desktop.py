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
from .pointer import PointerMonitor
from .recording import VideoRecorder
from .render import (
    READOUT_HEIGHT,
    RenderedThermalFrame,
    TemperatureStats,
    ThermalRenderer,
    draw_temperature_readout,
)
from .window_style import set_black_window_backgrounds

LOG = logging.getLogger(__name__)
WINDOW_NAME = "TOPDON TC002C Duo"
AMBIENT_TRACKBAR = "Ambient x0.1 C"
AMBIENT_MIN_C = -50.0
AMBIENT_MAX_C = 100.0
AMBIENT_STEP_C = 0.1
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


def clamp_ambient(value: float) -> float:
    return round(min(max(value, AMBIENT_MIN_C), AMBIENT_MAX_C), 1)


def ambient_to_trackbar(value: float) -> int:
    return round((clamp_ambient(value) - AMBIENT_MIN_C) / AMBIENT_STEP_C)


def trackbar_to_ambient(position: int) -> float:
    return clamp_ambient(AMBIENT_MIN_C + position * AMBIENT_STEP_C)


@dataclass
class MousePicker:
    x: int | None = None
    y: int | None = None
    ambient_steps: int = 0
    clicks: list[tuple[int, int]] = field(default_factory=list)

    def callback(self, event: int, x: int, y: int, flags: int, _parameter) -> None:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONUP:
            self.x, self.y = x, y
            self.clicks.append((x, y))
        elif event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            delta = (flags >> 16) & 0xFFFF
            if delta >= 0x8000:
                delta -= 0x10000
            if delta:
                self.ambient_steps += 1 if delta > 0 else -1

    def consume_ambient_steps(self) -> int:
        steps, self.ambient_steps = self.ambient_steps, 0
        return steps

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

    def toggle(self) -> None:
        self.placing = not self.placing
        if not self.placing:
            self.pixels.clear()

    def add(self, pixel: tuple[int, int] | None) -> None:
        if self.placing and pixel is not None and pixel not in self.pixels:
            self.pixels.append(pixel)

    def rotate_clockwise(self, sensor_height: int) -> None:
        self.pixels = [(sensor_height - 1 - y, x) for x, y in self.pixels]


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
        return [
            "zenity",
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title=Save thermal recording" if video else "--title=Save thermal capture",
            f"--filename={Path(directory) / default_name}",
            "--file-filter=MP4 videos | *.mp4" if video else "--file-filter=PNG images | *.png",
        ]


def toolbar_layout(width: int) -> ToolbarLayout:
    """Fit compact controls across a single row at the current image width."""
    controls = (
        ("ambient_down", "Ambient -", 70),
        ("ambient_up", "Ambient +", 70),
        ("rotate", "Rotate", 54),
        ("unit", "C / F", 50),
        ("spots", "Add spots", 84),
        ("capture", "Capture", 64),
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
    ambient_celsius: float,
    temperature_unit: str,
    placing_spots: bool = False,
    recording_mode: str | None = None,
    pending_recording: str | None = None,
    status: str = "Ready",
    stats: TemperatureStats | None = None,
) -> np.ndarray:
    layout = toolbar_layout(image.shape[1])
    canvas = np.zeros((image.shape[0] + layout.height, image.shape[1], 3), np.uint8)
    canvas[layout.height :] = image
    if stats is not None:
        canvas[layout.height - READOUT_HEIGHT :] = draw_temperature_readout(
            image, stats, ambient_celsius, temperature_unit
        )
    ambient_display = ambient_celsius
    if temperature_unit == "F":
        ambient_display = ambient_celsius * 9.0 / 5.0 + 32.0
    labels = {
        "ambient_down": f"- {ambient_display:.1f}{temperature_unit}",
        "ambient_up": f"{ambient_display:.1f}{temperature_unit} +",
        "rotate": "Rotate",
        "unit": f"Unit: {temperature_unit}",
        "spots": "Clear spots" if placing_spots else "Add spots",
        "capture": "Capture",
        "help": "Help",
        "quit": "Quit",
    }
    for action, (x0, y0, x1, y1) in layout.buttons.items():
        active = action == "unit" or (action == "spots" and placing_spots)
        active = active or (action == "capture" and bool(recording_mode or pending_recording))
        fill = (74, 92, 70) if active else (47, 51, 61)
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
            (235, 238, 244),
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
    ambient_celsius: float,
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
        ambient_celsius=np.float32(ambient_celsius),
        raw_gain_divisor=np.float32(64.0),
        rotation_degrees=np.int16(rotation),
    )

    metadata: dict[str, object] = {
        "captured_at": captured_at.isoformat(),
        "camera": "TOPDON TC002C Duo",
        "sensor_shape": list(rendered.temperatures_celsius.shape),
        "ambient_celsius": ambient_celsius,
        "rotation_degrees": rotation,
        "display_unit": display_unit,
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


def draw_sample_spots(
    image: np.ndarray,
    rendered: RenderedThermalFrame,
    spots: SampleSpots,
    scale: int,
    temperature_unit: str = "C",
) -> np.ndarray:
    image = image.copy()
    color = (80, 255, 80)
    for sensor_x, sensor_y in spots.pixels:
        image_x = sensor_x * scale + scale // 2
        image_y = sensor_y * scale + scale // 2
        temperature = float(rendered.temperatures_celsius[sensor_y, sensor_x])
        if temperature_unit == "F":
            temperature = temperature * 9.0 / 5.0 + 32.0
        text = f"{temperature:.2f} {temperature_unit}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        text_x = max(2, min(image_x + 8, image.shape[1] - text_width - 2))
        text_y = image_y - 8
        if text_y < text_height + 2:
            text_y = image_y + text_height + 8
        text_y = min(text_y, image.shape[0] - baseline - 2)
        for thickness, text_color in ((3, (0, 0, 0)), (1, color)):
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
    # Draw markers last so nearby labels cannot hide the sampled pixels.
    for sensor_x, sensor_y in spots.pixels:
        cv2.drawMarker(
            image,
            (sensor_x * scale + scale // 2, sensor_y * scale + scale // 2),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_8,
        )
    return image


def draw_picker(
    rendered: RenderedThermalFrame,
    picker: MousePicker,
    scale: int,
    viewport_size: tuple[int, int] | None = None,
    toolbar_height: int = 0,
    temperature_unit: str = "C",
    pointer_over_image: bool | None = None,
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

    cv2.drawMarker(
        image,
        (image_x, image_y),
        (80, 255, 80),
        markerType=cv2.MARKER_CROSS,
        markerSize=9,
        thickness=1,
        line_type=cv2.LINE_8,
    )
    text = f"({sensor_x}, {sensor_y}) {temperature:.2f} {temperature_unit}"
    text_x = min(image_x + 12, max(5, image.shape[1] - 190))
    text_y = max(20, image_y - 12)
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 255, 80),
        1,
        cv2.LINE_AA,
    )
    return image, (sensor_x, sensor_y)


def draw_control_instructions(image: np.ndarray) -> np.ndarray:
    """Draw a translucent keyboard/mouse help panel over the image."""
    result = image.copy()
    panel_width = min(390, result.shape[1] - 20)
    panel_height = min(299, result.shape[0] - 20)
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
        ("Wheel/slider Adjust ambient by 0.1 C", (210, 215, 225)),
        ("[ / ]        Ambient down / up", (210, 215, 225)),
        ("S            Save image data", (210, 215, 225)),
        ("C            Open Capture controls", (210, 215, 225)),
        ("O            Rotate 90 degrees clockwise", (210, 215, 225)),
        ("F            Toggle Celsius / Fahrenheit", (210, 215, 225)),
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
    parser.add_argument("--ambient", type=float, default=22.0)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--scale", type=int, choices=range(1, 7), default=3)
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    renderer = ThermalRenderer(
        scale=args.scale,
        ambient_celsius=args.ambient,
        rotation=args.rotate,
    )
    camera = TC002CDuoCamera()
    picker = MousePicker()
    spots = SampleSpots()
    pointer_monitor = PointerMonitor(WINDOW_NAME)
    save_dialog = LinuxSaveDialog() if sys.platform.startswith("linux") else MacSaveDialog()
    recorder = VideoRecorder()
    capture_panel = CapturePanel()
    pending_save_kind: str | None = None
    capture_cursor = False
    timelapse_fpm = args.timelapse_fpm
    status_message = "Ready"
    status_until = 0.0
    status = "Ready"
    show_instructions = False
    last_selected: tuple[int, int] | None = None

    def set_ambient(value: float, *, update_trackbar: bool = True) -> None:
        renderer.ambient_celsius = clamp_ambient(value)
        if update_trackbar:
            cv2.setTrackbarPos(
                AMBIENT_TRACKBAR,
                WINDOW_NAME,
                ambient_to_trackbar(renderer.ambient_celsius),
            )

    def on_ambient_trackbar(position: int) -> None:
        set_ambient(trackbar_to_ambient(position), update_trackbar=False)

    def notify(message: str) -> None:
        nonlocal status_message, status_until
        status_message = message
        status_until = time.monotonic() + 8.0
        LOG.info("%s", message)

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
        # Rotate the stored sensor coordinates with the image, preserving samples.
        sensor_height = SENSOR_HEIGHT if renderer.rotation in (0, 180) else SENSOR_WIDTH
        spots.rotate_clockwise(sensor_height)
        renderer.rotate_clockwise()
        picker.x = picker.y = None
        last_selected = None

    def capture_state() -> dict:
        return {
            "recording_mode": recorder.mode,
            "pending_recording": pending_save_kind if pending_save_kind != "image" else None,
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
        "Mouse: inspect a pixel | wheel/slider or [/]: ambient +/- 0.1 C | "
        "p: add/clear spots | s: save image data | c: Capture controls | "
        "o: rotate | f: C/F | Space: controls | q/Esc: quit"
    )
    try:
        camera.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, picker.callback)
        cv2.createTrackbar(
            AMBIENT_TRACKBAR,
            WINDOW_NAME,
            ambient_to_trackbar(renderer.ambient_celsius),
            ambient_to_trackbar(AMBIENT_MAX_C),
            on_ambient_trackbar,
        )
        set_black_window_backgrounds(WINDOW_NAME)
        initial_window_size_set = False
        for frame in camera.frames():
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
            ambient_steps = picker.consume_ambient_steps()
            if ambient_steps:
                set_ambient(renderer.ambient_celsius + ambient_steps * AMBIENT_STEP_C)
            rendered = renderer.render_detailed(frame)
            layout = toolbar_layout(rendered.image.shape[1])
            if not initial_window_size_set:
                cv2.resizeWindow(
                    WINDOW_NAME,
                    rendered.image.shape[1],
                    rendered.image.shape[0] + layout.height,
                )
                initial_window_size_set = True
            viewport_size = mouse_viewport_size()
            display, selected = draw_picker(
                rendered,
                picker,
                renderer.scale,
                viewport_size=viewport_size,
                toolbar_height=layout.height,
                temperature_unit=renderer.temperature_unit,
                pointer_over_image=pointer_monitor.over_image(
                    rendered.image.shape[0], layout.height
                ),
            )
            if selected is not None:
                last_selected = selected

            save_finished, save_path = save_dialog.poll()
            if save_finished:
                kind, pending_save_kind = pending_save_kind, None
                if save_path is not None:
                    try:
                        if kind in ("video", "timelapse"):
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
                status = f"Choose a filename for {pending_save_kind}..."
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
                pending_recording=pending_save_kind if pending_save_kind != "image" else None,
                status=status,
                stats=rendered.stats,
            )

            quit_requested = False
            for click_x, click_y in picker.consume_clicks():
                action = toolbar_action_at(
                    click_x,
                    click_y,
                    rendered.image.shape[1],
                    viewport_size=viewport_size,
                    canvas_height=display.shape[0],
                )
                if action == "ambient_down":
                    set_ambient(renderer.ambient_celsius - AMBIENT_STEP_C)
                elif action == "ambient_up":
                    set_ambient(renderer.ambient_celsius + AMBIENT_STEP_C)
                elif action == "capture":
                    open_capture()
                elif action == "rotate":
                    rotate_view()
                elif action == "spots":
                    spots.toggle()
                elif action == "unit":
                    renderer.toggle_temperature_unit()
                elif action == "help":
                    show_instructions = not show_instructions
                elif action == "quit":
                    quit_requested = True
                elif action is None and spots.placing:
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
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                rotate_view()
            elif key == ord("p"):
                spots.toggle()
            elif key == ord(" "):
                show_instructions = not show_instructions
            elif key == ord("["):
                set_ambient(renderer.ambient_celsius - AMBIENT_STEP_C)
            elif key == ord("]"):
                set_ambient(renderer.ambient_celsius + AMBIENT_STEP_C)
            elif key == ord("f"):
                renderer.toggle_temperature_unit()
            elif key == ord("s"):
                request_save()
            elif key == ord("c"):
                open_capture()
    except CameraError as exc:
        LOG.error("Unable to run desktop viewer: %s", exc)
        return 2
    finally:
        stop_recording()
        capture_panel.close()
        pointer_monitor.close()
        save_dialog.close()
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
