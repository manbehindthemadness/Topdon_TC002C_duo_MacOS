"""
Validate declarative custom controls and configuration documents.
"""

import math
import re
from typing import Any

from .bundle import read_json_object
from .colors import color_value
from .devices import validate_device


def validate_model_choices(models: Any) -> None:
    """
    Check optional model names and filename/checksum hints without reading any weights.
    """
    if not isinstance(models, list) or not 1 <= len(models) <= 128:
        raise ValueError("Custom model choices must contain 1 to 128 definitions")
    names: set[str] = set()
    for model in models:
        if (not isinstance(model, dict) or set(model) - {"name", "sha256", "filenames"}
                or "name" not in model or not ({"sha256", "filenames"} & set(model))):
            raise ValueError("Custom model choice requires name and sha256 or filenames")
        name = model["name"]
        if not isinstance(name, str) or not name.strip() or len(name) > 200 or name in names:
            raise ValueError("Custom model choices require unique nonempty names")
        names.add(name)
        if "sha256" in model:
            digest = model["sha256"]
            if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("Custom model choice needs a lowercase SHA-256")
        if "filenames" in model:
            filenames = model["filenames"]
            if (not isinstance(filenames, list) or not 1 <= len(filenames) <= 32
                    or any(not isinstance(name, str) or not 1 <= len(name) <= 255
                           or "/" in name or "\\" in name or not name.lower().endswith(".onnx")
                           for name in filenames)):
                raise ValueError("Custom model choice filenames must be ONNX basenames")


def validate_controls(controls: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Check bounded control definitions and their current configuration values.
    """
    if not isinstance(controls, list) or len(controls) > 64:
        raise ValueError("Custom controls must be a list of at most 64 definitions")
    keys: set[str] = set()
    for control in controls:
        if not isinstance(control, dict):
            raise ValueError("Custom control must be an object")  # noqa: TRY004 - uniform schema errors
        key, label, kind = (control.get(field) for field in ("key", "label", "type"))
        if not isinstance(key, str) or not key or len(key) > 200 or key in keys:
            raise ValueError("Custom controls require unique nonempty keys")
        keys.add(key)
        if not isinstance(label, str) or not label or len(label) > 200:
            raise ValueError(f"Invalid custom control label: {key}")
        if kind not in ("number", "integer", "boolean", "choice", "text", "color", "model", "device"):
            raise ValueError(f"Unsupported custom control type: {key}")
        fields = {"key", "label", "type"}
        if kind in ("number", "integer"):
            fields |= {"min", "max", "step"}
        elif kind == "choice":
            fields.add("options")
        elif kind == "model" and "models" in control:
            fields.add("models")
            validate_model_choices(control["models"])
        if set(control) != fields or key not in values:
            raise ValueError(f"Custom control fields/default missing: {key}")
        value = values[key]
        if kind in ("number", "integer"):
            numbers = [control[field] for field in ("min", "max", "step")] + [value]
            if any(isinstance(n, bool) or not isinstance(n, (int, float))
                   or abs(n) > 1_000_000_000 or not math.isfinite(n) for n in numbers):
                raise ValueError(f"Custom numeric control requires finite bounded numbers: {key}")
            low, high, step, current = numbers
            if kind == "integer" and any(type(n) is not int for n in numbers):
                raise ValueError(f"Custom integer control requires integers: {key}")
            if not low < high or not 1e-9 <= step <= high - low or not low <= current <= high:
                raise ValueError(f"Invalid custom control range/value: {key}")
        elif kind == "boolean" and type(value) is not bool:
            raise ValueError(f"Custom checkbox requires a boolean: {key}")
        elif kind == "color":
            color_value(value)
        elif kind == "device":
            validate_device(value)
        elif kind in ("choice", "text", "model"):
            if not isinstance(value, str):
                raise ValueError(f"Custom text/choice requires a string: {key}")
            if kind == "choice":
                options = control["options"]
                if (not isinstance(options, list) or not 1 <= len(options) <= 128
                        or any(not isinstance(option, str) for option in options)
                        or len(set(options)) != len(options) or value not in options):
                    raise ValueError(f"Invalid custom choice options/value: {key}")
    return controls


def read_controls(text: str, values: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Decode bounded serialized control definitions with the standard finite JSON rules.
    """
    document = read_json_object('{"controls":' + text + '}')
    return validate_controls(document["controls"], values)


def configuration(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Read plain settings or a defaults/controls JSON document.
    """
    document = read_json_object(text)
    if "defaults" not in document and "controls" not in document:
        return document, []
    if set(document) != {"defaults", "controls"} or not isinstance(document["defaults"], dict):
        raise ValueError("Custom control document needs defaults and controls")
    values = document["defaults"]
    controls = validate_controls(document["controls"], values)
    return values, controls
