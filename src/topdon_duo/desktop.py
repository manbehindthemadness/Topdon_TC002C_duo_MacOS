"""Native OpenCV desktop viewer with per-pixel inspection and radiometric saves."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraError, TC002CDuoCamera
from .render import RenderedThermalFrame, ThermalRenderer

LOG = logging.getLogger(__name__)
WINDOW_NAME = "TOPDON TC002C Duo"
AMBIENT_TRACKBAR = "Ambient x0.1 C"
AMBIENT_MIN_C = -50.0
AMBIENT_MAX_C = 100.0
AMBIENT_STEP_C = 0.1
TOOLBAR_ROW_HEIGHT = 42
TOOLBAR_PADDING = 6
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


class MacSaveDialog:
    """Non-blocking macOS save panel so USB capture continues behind it."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    @property
    def is_open(self) -> bool:
        return self._process is not None

    def open(self, default_directory: Path | None = None) -> bool:
        if self._process is not None:
            return False
        default_name = datetime.now().astimezone().strftime(
            "TC002C-Duo-%Y%m%d-%H%M%S.png"
        )
        directory = ""
        if default_directory is not None and default_directory.is_dir():
            directory = str(default_directory.resolve())
        self._process = subprocess.Popen(
            [
                "/usr/bin/osascript",
                "-e",
                SAVE_DIALOG_SCRIPT,
                "--",
                default_name,
                directory,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True

    def poll(self) -> tuple[bool, Path | None]:
        """Return (finished, selected path); cancellation yields (True, None)."""
        if self._process is None or self._process.poll() is None:
            return False, None
        process, self._process = self._process, None
        stdout, stderr = process.communicate()
        if process.returncode == 0 and stdout.strip():
            return True, Path(stdout.strip())
        if "User canceled" not in stderr:
            LOG.error("macOS save dialog failed: %s", stderr.strip() or process.returncode)
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


def toolbar_layout(width: int) -> ToolbarLayout:
    """Lay out compact controls, wrapping when the camera is rotated or small."""
    controls = (
        ("ambient_down", "Ambient -", 82),
        ("ambient_up", "Ambient +", 82),
        ("save", "Save", 58),
        ("rotate", "Rotate", 68),
        ("unit", "C / F", 58),
        ("help", "Help", 56),
        ("quit", "Quit", 52),
    )
    buttons: dict[str, tuple[int, int, int, int]] = {}
    x = TOOLBAR_PADDING
    row = 0
    for action, _label, button_width in controls:
        if x > TOOLBAR_PADDING and x + button_width > width - TOOLBAR_PADDING:
            row += 1
            x = TOOLBAR_PADDING
        y = TOOLBAR_PADDING + row * TOOLBAR_ROW_HEIGHT
        buttons[action] = (x, y, x + button_width, y + 30)
        x += button_width + TOOLBAR_PADDING
    return ToolbarLayout(
        height=TOOLBAR_PADDING * 2 + (row + 1) * TOOLBAR_ROW_HEIGHT,
        buttons=buttons,
    )


def draw_toolbar(
    image: np.ndarray,
    ambient_celsius: float,
    temperature_unit: str,
) -> np.ndarray:
    layout = toolbar_layout(image.shape[1])
    canvas = np.zeros((image.shape[0] + layout.height, image.shape[1], 3), np.uint8)
    canvas[: layout.height] = (24, 27, 34)
    canvas[layout.height :] = image
    ambient_display = ambient_celsius
    if temperature_unit == "F":
        ambient_display = ambient_celsius * 9.0 / 5.0 + 32.0
    labels = {
        "ambient_down": f"- {ambient_display:.1f}{temperature_unit}",
        "ambient_up": f"{ambient_display:.1f}{temperature_unit} +",
        "save": "Save",
        "rotate": "Rotate",
        "unit": f"Unit: {temperature_unit}",
        "help": "Help",
        "quit": "Quit",
    }
    for action, (x0, y0, x1, y1) in layout.buttons.items():
        active = action == "unit"
        fill = (74, 92, 70) if active else (47, 51, 61)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
        label = labels[action]
        (text_width, text_height), _baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )
        cv2.putText(
            canvas,
            label,
            (x0 + (x1 - x0 - text_width) // 2, y0 + (y1 - y0 + text_height) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 238, 244),
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

    if not cv2.imwrite(str(png_path), rendered.image):
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


def draw_picker(
    rendered: RenderedThermalFrame,
    picker: MousePicker,
    scale: int,
    viewport_size: tuple[int, int] | None = None,
    toolbar_height: int = 0,
    temperature_unit: str = "C",
) -> tuple[np.ndarray, tuple[int, int] | None]:
    image = rendered.image.copy()
    if picker.x is None or picker.y is None:
        return image, None
    image_x, image_y = picker.x, picker.y
    if viewport_size and viewport_size[0] > 0 and viewport_size[1] > 0:
        image_x = round(image_x * image.shape[1] / viewport_size[0])
        canvas_height = image.shape[0] + toolbar_height
        image_y = round(image_y * canvas_height / viewport_size[1])
    image_y -= toolbar_height
    if image_y < 0 or image_y >= image.shape[0]:
        return image, None
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
        markerSize=20,
        thickness=2,
    )
    text = f"({sensor_x}, {sensor_y}) {temperature:.2f} {temperature_unit}"
    text_x = min(image_x + 12, max(5, image.shape[1] - 190))
    text_y = max(52, image_y - 12)
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
    panel_height = min(249, result.shape[0] - 20)
    x0, y0 = 10, result.shape[0] - panel_height - 10
    x1, y1 = x0 + panel_width, y0 + panel_height

    overlay = result.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (8, 10, 16), -1)
    cv2.addWeighted(overlay, 0.82, result, 0.18, 0, result)
    cv2.rectangle(result, (x0, y0), (x1, y1), (120, 130, 150), 1)

    lines = (
        ("Controls", (255, 255, 255)),
        ("Mouse move   Inspect pixel temperature", (210, 215, 225)),
        ("Wheel/slider Adjust ambient by 0.1 C", (210, 215, 225)),
        ("[ / ]        Ambient down / up", (210, 215, 225)),
        ("S            Save PNG + radiometric data", (210, 215, 225)),
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
        "--output",
        type=Path,
        default=None,
        help="initial directory for the macOS Save dialog",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


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
    save_dialog = MacSaveDialog()
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

    def request_save() -> None:
        try:
            if not save_dialog.open(args.output):
                LOG.info("A Save dialog is already open")
        except OSError as exc:
            LOG.error("Unable to open the macOS Save dialog: %s", exc)

    print(
        "Mouse: inspect a pixel | wheel/slider or [/]: ambient +/- 0.1 C | "
        "s: save | o: rotate | f: C/F | Space: controls | q/Esc: quit"
    )
    try:
        camera.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, picker.callback)
        cv2.createTrackbar(
            AMBIENT_TRACKBAR,
            WINDOW_NAME,
            ambient_to_trackbar(renderer.ambient_celsius),
            ambient_to_trackbar(AMBIENT_MAX_C),
            on_ambient_trackbar,
        )
        initial_window_size_set = False
        for frame in camera.frames():
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
            try:
                _left, _top, viewport_width, viewport_height = cv2.getWindowImageRect(
                    WINDOW_NAME
                )
                viewport_size = (viewport_width, viewport_height)
            except cv2.error:
                viewport_size = None
            display, selected = draw_picker(
                rendered,
                picker,
                renderer.scale,
                viewport_size=viewport_size,
                toolbar_height=layout.height,
                temperature_unit=renderer.temperature_unit,
            )
            if selected is not None:
                last_selected = selected

            save_finished, save_path = save_dialog.poll()
            if save_finished and save_path is not None:
                saved = save_capture(
                    rendered,
                    save_path.parent,
                    renderer.ambient_celsius,
                    renderer.rotation,
                    last_selected,
                    renderer.temperature_unit,
                    base_path=save_path,
                )
                print("Saved " + ", ".join(str(path) for path in saved))
            if show_instructions:
                display = draw_control_instructions(display)
            display = draw_toolbar(
                display,
                renderer.ambient_celsius,
                renderer.temperature_unit,
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
                elif action == "save":
                    request_save()
                elif action == "rotate":
                    renderer.rotate_clockwise()
                    picker.x = picker.y = None
                    last_selected = None
                elif action == "unit":
                    renderer.toggle_temperature_unit()
                elif action == "help":
                    show_instructions = not show_instructions
                elif action == "quit":
                    quit_requested = True
            if quit_requested:
                break
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                renderer.rotate_clockwise()
                picker.x = picker.y = None
                last_selected = None
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
    except CameraError as exc:
        LOG.error("Unable to run desktop viewer: %s", exc)
        return 2
    finally:
        save_dialog.close()
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
