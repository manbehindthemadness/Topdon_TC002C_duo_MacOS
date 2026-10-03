"""Native OpenCV desktop viewer with per-pixel inspection and radiometric saves."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraError, TC002CDuoCamera
from .render import RenderedThermalFrame, ThermalRenderer

LOG = logging.getLogger(__name__)
WINDOW_NAME = "TOPDON TC002C Duo"


@dataclass
class MousePicker:
    x: int | None = None
    y: int | None = None

    def callback(self, event: int, x: int, y: int, _flags: int, _parameter) -> None:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.x, self.y = x, y


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
) -> list[Path]:
    """Save a viewable PNG plus lossless raw/Celsius data and JSON metadata."""
    output_directory.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now().astimezone()
    stamp = captured_at.strftime("%Y%m%d-%H%M%S-%f")
    base = output_directory / f"TC002C-Duo-{stamp}"
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

    paths = [output_directory, png_path, data_path, json_path]
    _restore_user_ownership(paths)
    return paths[1:]


def draw_picker(
    rendered: RenderedThermalFrame, picker: MousePicker, scale: int
) -> tuple[np.ndarray, tuple[int, int] | None]:
    image = rendered.image.copy()
    if picker.x is None or picker.y is None:
        return image, None
    sensor_x = min(max(picker.x // scale, 0), rendered.temperatures_celsius.shape[1] - 1)
    sensor_y = min(max(picker.y // scale, 0), rendered.temperatures_celsius.shape[0] - 1)
    temperature = float(rendered.temperatures_celsius[sensor_y, sensor_x])

    cv2.drawMarker(
        image,
        (picker.x, picker.y),
        (80, 255, 80),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )
    text = f"({sensor_x}, {sensor_y}) {temperature:.2f} C"
    text_x = min(picker.x + 12, max(5, image.shape[1] - 190))
    text_y = max(52, picker.y - 12)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native TOPDON TC002C Duo viewer")
    parser.add_argument("--ambient", type=float, default=22.0)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--scale", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--output", type=Path, default=Path("captures"))
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

    print("Mouse: inspect a pixel | s: save PNG + radiometric data | o: rotate | q/Esc: quit")
    try:
        camera.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, picker.callback)
        for frame in camera.frames():
            rendered = renderer.render_detailed(frame)
            display, selected = draw_picker(rendered, picker, renderer.scale)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                renderer.rotate_clockwise()
                picker.x = picker.y = None
            elif key == ord("s"):
                saved = save_capture(
                    rendered,
                    args.output,
                    renderer.ambient_celsius,
                    renderer.rotation,
                    selected,
                )
                print("Saved " + ", ".join(str(path) for path in saved))
    except CameraError as exc:
        LOG.error("Unable to run desktop viewer: %s", exc)
        return 2
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
