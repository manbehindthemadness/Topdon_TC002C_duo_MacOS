"""
Validate the example's editable JSON settings before inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from topdon_duo.custom_nodes.colors import color_value
from topdon_duo.custom_nodes.devices import validate_device


def fraction(config: dict[str, Any], name: str, default: float, minimum: float = 0.0) -> float:
    """
    Read a finite bounded fraction, rejecting booleans and oversized integers.
    """
    value = config.get(name, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not minimum <= value <= 1):
        raise ValueError(f"{name} must be a finite number from {minimum} to 1")
    return float(value)


def boolean(config: dict[str, Any], name: str, default: bool) -> bool:
    """
    Read an explicit on/off setting without coercing arbitrary JSON values.
    """
    value = config.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false")
    return value


def integer(config: dict[str, Any], name: str, default: int, maximum: int) -> int:
    """
    Read a positive integer, rejecting booleans and fractional values.
    """
    value = config.get(name, default)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    """
    Validated model, inference and overlay settings.
    """

    model_path: str
    providers: tuple[str, ...]
    threshold: float
    max_detections: int
    thickness: int
    show_scores: bool
    nesting: str
    border_color: str
    fill_color: str
    fill_opacity: float
    filter_scores: bool
    score_maximum: float
    nested_coverage: float
    suppress_duplicates: bool
    duplicate_iou: float
    device: dict[str, str] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Settings:
        """
        Validate an optional local-model override and bounded display settings.
        """
        model = config.get("model_path", "")
        if not isinstance(model, str):
            raise TypeError("model_path must be a string")
        resolved = ""
        if model.strip():
            path = Path(model).expanduser()
            if not path.is_absolute():
                raise ValueError("model_path must be absolute or begin with ~")
            if not path.is_file():
                raise ValueError(f"Model not found: {path}")
            resolved = str(path)
        device = validate_device(config["device"]) if "device" in config else None
        providers = (
            config.get("providers", ["CPUExecutionProvider"])
            if device is None else ["CPUExecutionProvider"]
        )
        if (
            not isinstance(providers, list)
            or not providers
            or any(not isinstance(provider, str) or not provider for provider in providers)
        ):
            raise ValueError("providers must be a nonempty list of ONNX Runtime provider names")
        threshold = fraction(config, "score_threshold", 0.2)
        score_maximum = fraction(config, "score_maximum", 1.0)
        filter_scores = boolean(config, "filter_scores", True)
        if filter_scores and threshold > score_maximum:
            raise ValueError("score_threshold must not exceed score_maximum")
        show_scores = boolean(config, "show_scores", False)
        nesting = config.get("nesting", "Outside-in")
        if nesting not in ("Inside-out", "Outside-in"):
            raise ValueError("nesting must be Inside-out or Outside-in")
        opacity = fraction(config, "fill_opacity", 0.0)
        settings = cls(
            resolved,
            tuple(providers),
            float(threshold),
            integer(config, "max_detections", 25, 100),
            integer(config, "line_thickness", 1, 8),
            show_scores,
            nesting,
            color_value(config.get("border_color", "#00ff00")),
            color_value(config.get("fill_color", "#00ff00")),
            float(opacity),
            filter_scores,
            score_maximum,
            fraction(config, "nested_coverage", 0.85, 0.5),
            boolean(config, "suppress_duplicates", True),
            fraction(config, "duplicate_iou", 0.5, 0.01),
            device,
        )
        return settings
