"""
Portable color values for declarative custom-node controls and drawing.
"""

import re
from typing import Any


def color_value(value: Any) -> str:
    """
    Normalize a literal RGB hex color or the measurement-style dynamic mode.
    """
    if value == "dynamic":
        return value
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("Color must be #RRGGBB or dynamic")
    return value.lower()


def color_bgr(value: str) -> tuple[int, int, int]:
    """
    Convert a validated fixed RGB hex color into OpenCV channel order.
    """
    color = color_value(value)
    if color == "dynamic":
        raise ValueError("Dynamic colors require image compositing")
    red, green, blue = (int(color[offset:offset + 2], 16) for offset in (1, 3, 5))
    return blue, green, red
