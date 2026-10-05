"""Validated desktop preferences, independent of Qt and camera baseline snapshots."""

import json

from .distance_calibration import DistanceReference
from .hardware_controls import BLOCK_LENGTHS, HARDWARE_CONTROLS
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
    rotation = saved.get("rotation")
    if type(rotation) is int and rotation in (0, 90, 180, 270):
        result["rotation"] = rotation
    for name in ("advanced_auto", "show_graph"):
        if isinstance(saved.get(name), bool):
            result[name] = saved[name]
    try:
        result["distance_calibration"] = DistanceReference.from_dict(
            saved.get("distance_calibration")
        ).as_dict()
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
