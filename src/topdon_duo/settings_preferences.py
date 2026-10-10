"""Validated desktop preferences, independent of Qt and camera baseline snapshots."""

import json
import math
from typing import Any, cast

from .camera import FRAME_RATE
from .camera_backends import CameraProfile, ControlSpec
from .camera_backends.preferences import DEVICE_KEYS, settings_path
from .distance_calibration import DistanceReference
from .emissivity_calibration import validate_reference
from .graph_settings import validate_graph_settings
from .graphs import validate_graph_interval
from .hardware_controls import BLOCK_LENGTHS, HARDWARE_CONTROLS, PROCESSING_PRESETS
from .pipeline import legacy_calibration_values, validate_pipeline
from .preference_io import save_json
from .reflected_calibration import validate_reference as validate_reflected_reference
from .spot_preferences import validate_spots
from .view_settings import VIEW_DEFAULTS, validate_view_setting
from .window_preferences import _path


def load_settings(
    profile: CameraProfile | None = None, specs: dict[str, ControlSpec] | None = None,
) -> dict[str, Any]:
    """
    Restore validated preferences and migrate removed pipeline correction controls.
    """
    try:
        saved = json.loads(settings_path(_path(), profile).read_text())
    except (OSError, ValueError):
        if profile is None or profile.id == "duo":
            return {}
        saved = {}
    if not isinstance(saved, dict):
        return {}
    if profile is not None and profile.id != "duo":
        try:
            shared = json.loads(_path().with_name("settings.json").read_text())
        except (OSError, ValueError):
            shared = {}
        if isinstance(shared, dict):
            saved = {**{key: value for key, value in shared.items() if key not in DEVICE_KEYS},
                     **saved}
    result: dict[str, Any] = {"display": {}, "hardware": {}}
    for section in ("display", "hardware"):
        values = saved.get(section, {})
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            try:
                if section == "display":
                    validate_view_setting(name, value)
                else:
                    if profile is not None and profile.id != "duo":
                        if specs is not None and name in specs and specs[name].supported:
                            specs[name].validate(value)
                        elif (type(value) not in (str, bool, int, float)
                              or isinstance(value, (int, float)) and not math.isfinite(value)):
                            continue
                    else:
                        spec = HARDWARE_CONTROLS[name]
                        spec.apply(bytearray(BLOCK_LENGTHS[spec.selector, spec.command]), value)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            result[section][name] = value
    display = result["display"]
    spot_hardware = saved.get("spot_hardware")
    if specs is not None and isinstance(spot_hardware, dict):
        result["spot_hardware"] = {}
        for spot_id, values in spot_hardware.items():
            if not isinstance(spot_id, str) or not isinstance(values, dict):
                continue
            accepted = {}
            for name, value in values.items():
                spec = specs.get(name)
                if spec is not None and spec.scope != "spot":
                    continue
                try:
                    if spec is not None and spec.supported:
                        spec.validate(value)
                    elif (type(value) not in (str, bool, int, float)
                          or isinstance(value, (int, float)) and not math.isfinite(value)):
                        continue
                except (TypeError, ValueError):
                    continue
                accepted[name] = value
            result["spot_hardware"][spot_id] = accepted
    if "raw_anime4k" in display:
        enabled = display.pop("raw_anime4k")
        display.setdefault("raw_upsampling", "anime4k09" if enabled else "off")
    if display.get("image_source") == "analyze":
        display["image_source"] = "raw"
        display["analyze_mode"] = True
    if display.get("raw_temperature_low", VIEW_DEFAULTS["raw_temperature_low"]) >= display.get(
        "raw_temperature_high", VIEW_DEFAULTS["raw_temperature_high"]
    ):
        display.pop("raw_temperature_low", None)
        display.pop("raw_temperature_high", None)
    ambient = saved.get("ambient_input_celsius")
    if (
        not isinstance(ambient, bool)
        and isinstance(ambient, (int, float))
        and -50 <= ambient <= 100
        and math.isfinite(ambient)
    ):
        result["ambient_input_celsius"] = ambient
    try:
        result["spots"] = validate_spots(
            saved.get("spots"), native_size=profile.native_size if profile else None,
        )
    except (TypeError, ValueError):
        pass
    try:
        result["graph_settings"] = validate_graph_settings(saved.get("graph_settings"))
    except (TypeError, ValueError, OverflowError):
        pass
    rate = saved.get("timelapse_fpm")
    frame_rate = profile.frame_rate if profile else FRAME_RATE
    if type(rate) is int and 1 <= cast(int, rate) <= frame_rate * 60:
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
    if type(gamma) is int and 0 <= cast(int, gamma) <= 100:
        result["camera_gamma"] = gamma
    boost = saved.get("camera_boost")
    if type(boost) is bool:  # Migrate the old Off/On checkbox (On used mode 3).
        result["camera_boost"] = 3 if boost else 0
    elif type(boost) is int and boost in (0, 1, 2, 3):
        result["camera_boost"] = boost
    try:
        result["graph_interval"] = validate_graph_interval(saved.get("graph_interval"))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        result["distance_calibration"] = DistanceReference.from_dict(
            saved.get("distance_calibration")
        ).as_dict()
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        result["emissivity_calibration"] = validate_reference(saved.get("emissivity_calibration"))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        result["reflected_calibration"] = validate_reflected_reference(
            saved.get("reflected_calibration")
        )
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        result["pipeline"] = validate_pipeline(
            saved.get("pipeline"), hardware_profile=profile.id if profile else "duo",
        )
        legacy_values = legacy_calibration_values(saved["pipeline"]) if (
            profile is None or profile.id == "duo"
        ) else {}
        for name, value in legacy_values.items():
            if name == "humidity":
                result["hardware"].setdefault(name, value)
            else:
                result["hardware"][name] = value
    except (TypeError, ValueError, OverflowError):
        pass
    return result


def save_settings(settings: dict[str, Any], profile: CameraProfile | None = None) -> None:
    """
    Atomically persist settings without sharing temporary files with other saves.
    """
    path = settings_path(_path(), profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, settings, indent=2)
    if profile is not None and profile.id != "duo":
        shared_path = _path().with_name("settings.json")
        try:
            shared = json.loads(shared_path.read_text())
        except (OSError, ValueError):
            shared = {}
        if not isinstance(shared, dict):
            shared = {}
        shared.update({key: value for key, value in settings.items() if key not in DEVICE_KEYS})
        save_json(shared_path, shared, indent=2)
