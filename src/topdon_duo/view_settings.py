"""Shared display options for the renderer and its separate Qt popup."""

import math

TEMPERATURE_UNITS = {"C": "Metric (°C, cm)", "F": "Imperial (°F, in)"}
DISTANCE_METERS_PER_UNIT = {"C": 0.01, "F": 0.0254}
IMAGE_SOURCES = {
    "preview": "Camera preview",
    "raw": "Raw thermal image",
    "analyze": "Analyze mode — fixed raw thermal",
}
PALETTE_SOURCES = {"app": "App colors", "camera": "Camera colors (preview)"}
IMAGE_FILTERS = {
    "none": "None",
    "bilateral": "Edge-preserving denoise",
    "median": "Median denoise",
    "gaussian": "Smooth",
    "sharpen": "Sharpen",
}
UPSCALING_MODES = {
    "off": "Off",
    "anime4k09": "Anime4K09 2× (CPU only) — phone algorithm",
    "acnet-legacy-hdn0": "ACNet 2× — no denoise",
    "acnet-legacy-hdn1": "ACNet 2× — light denoise",
    "acnet-legacy-hdn2": "ACNet 2× — medium denoise",
    "acnet-legacy-hdn3": "ACNet 2× — strong denoise",
}
ENHANCEMENT_INPUTS = {
    "native": "Native sensor size",
    "preview": "Full preview",
}
ANALYZE_UPSCALING_MODES = {
    "bicubic": "Sensor interpolation 2× — no model",
    **UPSCALING_MODES,
}
COLOR_PALETTES = {
    "inferno": "Inferno",
    "magma": "Magma",
    "plasma": "Plasma",
    "turbo": "Turbo",
    "jet": "Rainbow",
    "hot": "Iron / hot",
    "white_hot": "White hot",
    "black_hot": "Black hot",
}
VIEW_DEFAULTS = {
    "temperature_unit": "C",
    "image_source": "preview",
    "analyze_mode": False,
    "mirror_horizontal": False,
    "mirror_vertical": False,
    "image_filter": "none",
    "upsampling": "off",
    "enhancement_amount": 1.0,
    "enhancement_input": "native",
    "anime4k_passes": 3,
    "antialiasing": True,
    "color_palette": "inferno",
    "palette_source": "camera",
    "raw_temperature_low": 15.0,
    "raw_temperature_high": 45.0,
    "raw_sharpen_amount": 0.0,
    "raw_upsampling": "bicubic",
    "raw_anime4k_passes": 3,
    "raw_palette": "white_hot",
}


def validate_view_setting(name: str, value: object) -> None:
    options = {
        "temperature_unit": TEMPERATURE_UNITS,
        "image_source": IMAGE_SOURCES,
        "image_filter": IMAGE_FILTERS,
        "upsampling": UPSCALING_MODES,
        "enhancement_input": ENHANCEMENT_INPUTS,
        "color_palette": COLOR_PALETTES,
        "palette_source": PALETTE_SOURCES,
        "raw_palette": COLOR_PALETTES,
        "raw_upsampling": ANALYZE_UPSCALING_MODES,
    }
    if name in options:
        if not isinstance(value, str) or value not in options[name]:
            raise ValueError(f"Invalid {name}: {value!r}")
    elif name in ("mirror_horizontal", "mirror_vertical", "antialiasing", "raw_anime4k", "analyze_mode"):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    elif name in ("enhancement_amount", "anime4k_passes", "raw_anime4k_passes", "raw_sharpen_amount",
                  "raw_temperature_low", "raw_temperature_high"):
        limits = ((-20, 550) if name.startswith("raw_temperature_") else
                  (1, 5) if name in ("anime4k_passes", "raw_anime4k_passes") else (0, 1))
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not limits[0] <= value <= limits[1]
        ):
            raise ValueError(f"Invalid {name}: {value!r}")
        if name in ("anime4k_passes", "raw_anime4k_passes") and int(value) != value:
            raise ValueError("Anime4K09 passes must be a whole number")
    else:
        raise ValueError(f"Unknown display setting: {name}")
