"""
Display-only image primitives and preview encoding.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from ..enhancement_limits import MAX_PIXELS

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def check_size(width: int, height: int) -> None:
    """
    Check size.
    """
    if width * height > MAX_PIXELS:
        raise ValueError("Pipeline exceeds the 4 megapixel image limit; reduce scales/passes")


def resize(
    image: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Resize.
    """
    check_size(*size)
    return cv2.resize(image, size, interpolation=interpolation)


def bytes_image(image: np.ndarray) -> np.ndarray:
    """
    Bytes image.
    """
    return np.clip(image, 0, 255).round().astype(np.uint8)


def luminance(image: np.ndarray) -> np.ndarray:
    """
    Luminance.
    """
    color = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2YCrCb)
    return color[..., 0]


def map_luminance(image: np.ndarray, operation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Map luminance.
    """
    color = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2YCrCb)
    color[..., 0] = operation(color[..., 0])
    return np.clip(cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR), 0, 255)


def colorize(gray: np.ndarray, palette: str) -> np.ndarray:
    """
    Colorize.
    """
    gray = bytes_image(gray)
    if palette.startswith("camera_"):
        value = int(palette.split("_")[1])
        aliases = {
            1: "white_hot",
            2: "black_hot",
            10: "plasma",
            11: "jet",
            12: "hot",
            13: "magma",
            14: "inferno",
            15: "hot",
            16: "turbo",
            17: "plasma",
            18: "white_hot",
            19: "jet",
            20: "hot",
            21: "summer",
            22: "ocean",
        }
        if value not in aliases:
            raise ValueError(
                "Unknown baseline camera palette; select a supported Camera colors palette"
            )
        palette = aliases[value]
    if palette in ("white_hot", "black_hot"):
        return cv2.cvtColor(255 - gray if palette == "black_hot" else gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{palette.upper()}"))


def combine_images(
    base: np.ndarray, incoming: np.ndarray, params: dict[str, Any], mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Blend display pixels; the current node fixes size and coordinate orientation.
    """
    if incoming.shape[:2] != base.shape[:2]:
        incoming = resize(
            incoming, (base.shape[1], base.shape[0]), INTERPOLATIONS[params["interpolation"]]
        )
    base, incoming = base.astype(np.float32), incoming.astype(np.float32)
    mode = params["mode"]
    if mode == "opacity":
        result = incoming
    elif mode == "weighted":
        result = cv2.addWeighted(
            base, params["base_weight"], incoming, params["input_weight"], params["offset"]
        )
    elif mode == "add":
        result = cv2.add(base, incoming)
    elif mode == "subtract":
        result = cv2.subtract(base, incoming)
    elif mode == "difference":
        result = cv2.absdiff(base, incoming)
    elif mode == "multiply":
        result = cv2.multiply(base, incoming, scale=1 / 255)
    elif mode == "screen":
        result = 255 - cv2.multiply(255 - base, 255 - incoming, scale=1 / 255)
    elif mode == "overlay":
        result = np.where(
            base <= 127.5,
            2 * base * incoming / 255,
            255 - 2 * (255 - base) * (255 - incoming) / 255,
        )
    elif mode == "lighten":
        result = cv2.max(base, incoming)
    elif mode == "darken":
        result = cv2.min(base, incoming)
    elif mode in ("and", "or", "xor"):
        result = getattr(cv2, "bitwise_" + mode)(bytes_image(base), bytes_image(incoming))
    elif mode == "mask":
        binary = luminance(incoming) >= params["threshold"]
        if params["invert"]:
            binary = ~binary
        result = base * binary[..., None]
    else:
        raise ValueError("Unknown combine mode")
    result = np.clip(result, 0, 255).astype(np.float32)
    if mask is None:
        return np.clip(
            cv2.addWeighted(base, 1 - params["opacity"], result, params["opacity"], 0), 0, 255
        )
    if mask.shape[:2] != base.shape[:2]:
        mask = resize(mask, (base.shape[1], base.shape[0]), INTERPOLATIONS[params["interpolation"]])
    gray = luminance(mask) if mask.ndim == 3 else mask.astype(np.float32)
    if params["mask_kind"] == "threshold":
        coverage = (gray >= params["mask_threshold"]).astype(np.float32)
    else:
        coverage = np.clip(gray / 255, 0, 1)
    if params["mask_invert"]:
        coverage = 1 - coverage
    opacity = coverage[..., None] * params["opacity"]
    return np.clip(base * (1 - opacity) + result * opacity, 0, 255)


def encode_thumbnail(image: np.ndarray) -> str:
    """
    Encode thumbnail.
    """
    height, width = image.shape[:2]
    factor = min(320 / width, 240 / height, 1)
    thumbnail = bytes_image(
        resize(
            image, (max(1, round(width * factor)), max(1, round(height * factor))), cv2.INTER_AREA
        )
    )
    success, encoded = cv2.imencode(".png", thumbnail, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return base64.b64encode(encoded).decode("ascii") if success else ""
