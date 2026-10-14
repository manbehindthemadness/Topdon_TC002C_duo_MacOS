"""Validated graph presentation and history preferences."""

import math

GRAPH_DEFAULTS = {
    "range_mode": "fixed",
    "range_seconds": 600.0,
    "compression": True,
    "history_points": 2048,
    "averaging_samples": 5,
    "interpolation_samples": 0,
}


def validate_graph_settings(values: object) -> dict:
    """
    Validate persisted chart settings, supplying defaults for older preferences.
    """
    if not isinstance(values, dict):
        raise TypeError("Graph settings must be an object")
    settings = {**GRAPH_DEFAULTS, **values}
    if set(settings) != set(GRAPH_DEFAULTS) or settings["range_mode"] not in ("session", "fixed"):
        raise ValueError("Choose entire session or a fixed time range")
    seconds = settings["range_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or not 6 <= seconds <= 604800
    ):
        raise ValueError("Chart range must be between 0.1 and 10,080 minutes")
    if type(settings["compression"]) is not bool:
        raise ValueError("Compression must be enabled or disabled")
    if (
        type(settings["history_points"]) is not int
        or not 256 <= settings["history_points"] <= 65536
    ):
        raise ValueError("History must contain between 256 and 65,536 points per chart")
    for name, minimum, maximum in (
        ("averaging_samples", 1, 256),
        ("interpolation_samples", 0, 32),
    ):
        if type(settings[name]) is not int or not minimum <= settings[name] <= maximum:
            raise ValueError(f"{name.replace('_', ' ').title()} must be {minimum}–{maximum}")
    return settings
