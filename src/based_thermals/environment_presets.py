"""
User-only presets combining environmental corrections and saved calibration references.
"""

import json
from pathlib import Path
from typing import Any

from .camera_backends import CameraProfile, ControlSpec
from .camera_backends.preferences import settings_path
from .distance_calibration import DistanceReference
from .emissivity_calibration import validate_reference as validate_emissivity
from .preference_io import save_json
from .reflected_calibration import validate_reference as validate_reflected
from .window_preferences import _path


def measurement_specs(specs: dict[str, ControlSpec]) -> dict[str, ControlSpec]:
    """
    Select supported whole-camera corrections independently of backend field names.
    """
    selected = {name: spec for name, spec in specs.items()
                if spec.supported and spec.effect == "measurement" and spec.scope == "device"}
    return selected


def validate_name(value: object) -> str:
    """
    Require a nonempty, bounded preset name suitable for display in a selector.
    """
    if not isinstance(value, str):
        raise TypeError("Enter a preset name")
    name = value.strip()
    if not name or len(name) > 100 or any(ord(char) < 32 for char in name):
        raise ValueError("Preset names must contain 1–100 characters without control characters")
    return name


def validate_preset(
    value: object, profile: CameraProfile, specs: dict[str, ControlSpec],
) -> dict[str, Any]:
    """
    Validate a complete compatible snapshot without touching hardware or user preferences.
    """
    if (not isinstance(value, dict) or type(value.get("version")) is not int
            or value["version"] != 1 or value.get("camera") != profile.settings_key):
        raise ValueError("Environment preset is incompatible with this camera profile")
    hardware = value.get("hardware")
    selected = measurement_specs(specs)
    if not selected or not isinstance(hardware, dict) or set(hardware) != set(selected):
        raise ValueError("Environment preset must contain all supported environmental settings")
    for name, spec in selected.items():
        spec.validate(hardware[name])
    references = value.get("calibrations")
    if not isinstance(references, dict) or set(references) != {"distance", "emissivity", "reflected"}:
        raise ValueError("Environment preset must contain its saved calibration references")
    validated = {}
    for name, validator in (
        ("distance", lambda distance_data: DistanceReference.from_dict(distance_data).as_dict()),
        ("emissivity", validate_emissivity), ("reflected", validate_reflected),
    ):
        reference = references[name]
        validated[name] = validator(reference) if reference is not None else None
    result = {"version": 1, "camera": profile.settings_key,
              "hardware": dict(hardware), "calibrations": validated}
    return result


def presets_path(profile: CameraProfile) -> Path:
    """
    Keep user presets beside the matching camera profile's preferences.
    """
    settings = settings_path(_path(), profile)
    path = (settings.with_name("environment-presets.json") if profile.id == "duo"
            else settings.with_suffix(".environment-presets.json"))
    return path


def load_presets(profile: CameraProfile, specs: dict[str, ControlSpec]) -> dict[str, dict[str, Any]]:
    """
    Load user-created presets only, ignoring malformed or incompatible saved entries.
    """
    try:
        saved = json.loads(presets_path(profile).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    presets = {}
    for name, document in saved.items():
        try:
            valid_name = validate_name(name)
            presets[valid_name] = validate_preset(document, profile, specs)
        except (ValueError, TypeError, OverflowError):
            continue
    return presets


def save_presets(
    presets: dict[str, dict[str, Any]], profile: CameraProfile, specs: dict[str, ControlSpec],
) -> None:
    """
    Atomically replace the validated user collection, with no shipped defaults or tombstones.
    """
    validated = {validate_name(name): validate_preset(document, profile, specs)
                 for name, document in presets.items()}
    save_json(presets_path(profile), validated, indent=2)
