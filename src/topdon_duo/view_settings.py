"""Shared display options for the renderer and its separate Qt popup."""

IMAGE_SOURCES = {"preview": "Camera preview", "raw": "Raw thermal image"}
IMAGE_FILTERS = {
    "none": "None",
    "bilateral": "Edge-preserving denoise",
    "median": "Median denoise",
    "gaussian": "Smooth",
    "sharpen": "Sharpen",
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
    "image_source": "preview",
    "mirror_horizontal": False,
    "mirror_vertical": False,
    "image_filter": "none",
    "antialiasing": True,
    "color_palette": "inferno",
}


def validate_view_setting(name: str, value: object) -> None:
    options = {
        "image_source": IMAGE_SOURCES,
        "image_filter": IMAGE_FILTERS,
        "color_palette": COLOR_PALETTES,
    }
    if name in options:
        if not isinstance(value, str) or value not in options[name]:
            raise ValueError(f"Invalid {name}: {value!r}")
    elif name in ("mirror_horizontal", "mirror_vertical", "antialiasing"):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    else:
        raise ValueError(f"Unknown display setting: {name}")
