"""Validated desktop preferences, independent of Qt and camera baseline snapshots."""

import json
import math

from .camera import FRAME_RATE
from .distance_calibration import DistanceReference
from .emissivity_calibration import validate_reference
from .graph_settings import validate_graph_settings
from .graphs import validate_graph_interval
from .hardware_controls import BLOCK_LENGTHS, HARDWARE_CONTROLS, PROCESSING_PRESETS
from .reflected_calibration import validate_reference as validate_reflected_reference
from .spot_preferences import validate_spots
from .view_settings import validate_view_setting
from .window_preferences import _path


def load_settings() -> dict:
    try:
        saved = json.loads(_path().with_name("settings.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    result = {"display": {}, "hardware": {}}
    for section in ("display", "hardware"):
        values = saved.get(section, {})
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            try:
                if section == "display":
                    validate_view_setting(name, value)
                else:
                    spec = HARDWARE_CONTROLS[name]
                    spec.apply(bytearray(BLOCK_LENGTHS[spec.selector, spec.command]), value)
            except (KeyError, TypeError, ValueError):
                continue
            result[section][name] = value
    ambient = saved.get("ambient_input_celsius")
    if (
        not isinstance(ambient, bool)
        and isinstance(ambient, (int, float))
        and math.isfinite(ambient)
        and -50 <= ambient <= 100
    ):
        result["ambient_input_celsius"] = ambient
    try:
        result["spots"] = validate_spots(saved.get("spots"))
    except (TypeError, ValueError):
        pass
    try:
        result["graph_settings"] = validate_graph_settings(saved.get("graph_settings"))
    except (TypeError, ValueError):
        pass
    rate = saved.get("timelapse_fpm")
    if type(rate) is int and 1 <= rate <= FRAME_RATE * 60:
        result["timelapse_fpm"] = rate
    rotation = saved.get("rotation")
    if type(rotation) is int and rotation in (0, 90, 180, 270):
        result["rotation"] = rotation
    for name in (
        "advanced_auto",
        "show_graph",
        "auto_calibrate",
        "fixed_range",
        "capture_cursor",
        "capture_graphs",
    ):
        if isinstance(saved.get(name), bool):
            result[name] = saved[name]
    preset = saved.get("processing_preset")
    if isinstance(preset, str) and preset in PROCESSING_PRESETS:
        result["processing_preset"] = preset
    gamma = saved.get("camera_gamma")
    if type(gamma) is int and 0 <= gamma <= 100:
        result["camera_gamma"] = gamma
    boost = saved.get("camera_boost")
    if type(boost) is bool:  # Migrate the old Off/On checkbox (On used mode 3).
        result["camera_boost"] = 3 if boost else 0
    elif type(boost) is int and boost in (0, 1, 2, 3):
        result["camera_boost"] = boost
    try:
        result["graph_interval"] = validate_graph_interval(saved.get("graph_interval"))
    except (TypeError, ValueError):
        pass
    try:
        result["distance_calibration"] = DistanceReference.from_dict(
            saved.get("distance_calibration")
        ).as_dict()
    except (TypeError, ValueError):
        pass
    try:
        result["emissivity_calibration"] = validate_reference(saved.get("emissivity_calibration"))
    except (TypeError, ValueError):
        pass
    try:
        result["reflected_calibration"] = validate_reflected_reference(
            saved.get("reflected_calibration")
        )
    except (TypeError, ValueError):
        pass
    return result


def save_settings(settings: dict) -> None:
    path = _path().with_name("settings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
