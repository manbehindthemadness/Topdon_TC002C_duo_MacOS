"""
Radiometric image files and ownership restoration.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..render import (
    RenderedThermalFrame,
    draw_temperature_readout,
)


def restore_user_ownership(paths: list[Path]) -> None:
    """
    Make sudo-created captures belong to the invoking desktop user.
    """
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
    """
    Save a viewable PNG plus lossless raw/Celsius data and JSON metadata.
    """
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
    restore_user_ownership(paths)
    return [png_path, data_path, json_path]
