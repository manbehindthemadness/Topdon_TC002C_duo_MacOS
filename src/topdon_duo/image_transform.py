"""
Display-only normalized projective transforms and deferred annotation geometry.
"""

from typing import Any

import cv2
import numpy as np

from .custom_nodes.labels import DisplayLabel
from .enhancement_limits import MAX_PIXELS

CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")
UNIT_CORNERS = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), np.float32)


def has_warp(params: dict[str, Any]) -> bool:
    """
    Distinguish spatial warps from working resolution changes and axis-aligned mirrors.
    """
    scaled_canvas = params["resize_mode"] == "canvas" and (
        params["width"] != 100 or params["height"] != 100
    )
    return scaled_canvas or any(params[key] != 0 for key in (
        "position_x", "position_y", "skew_x", "skew_y",
        *(f"{corner}_{axis}" for corner in CORNERS for axis in ("x", "y"))
    ))


def transform_matrix(
    params: dict[str, Any], size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Compose mirror, upper-left scale, shear, distortion and canvas position.

    Position uses the output canvas size; omit size for shape-independent validation.
    """
    destination = UNIT_CORNERS.copy()
    for index, corner in enumerate(CORNERS):
        destination[index] += (params[f"{corner}_x"] / 100, params[f"{corner}_y"] / 100)
    contour = destination.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour, oriented=True) <= 0.01:
        raise ValueError("Transform corners must form a non-crossing, non-collapsed quadrilateral")
    perspective = cv2.getPerspectiveTransform(UNIT_CORNERS, destination)
    sx, sy = params["skew_x"] / 100, params["skew_y"] / 100
    shear = np.array(((1, sx, -sx / 2), (sy, 1, -sy / 2), (0, 0, 1)))
    mirror = np.array((
        (-1 if params["horizontal"] else 1, 0, int(params["horizontal"])),
        (0, -1 if params["vertical"] else 1, int(params["vertical"])),
        (0, 0, 1),
    ))
    zx = params["width"] / 100 if params["resize_mode"] == "canvas" else 1
    zy = params["height"] / 100 if params["resize_mode"] == "canvas" else 1
    zoom = np.array(((zx, 0, 0), (0, zy, 0), (0, 0, 1)))
    matrix = perspective @ shear @ zoom @ mirror
    if size is not None:
        width, height = size
        dx, dy = params["position_x"], params["position_y"]
        if params["position_unit"] == "percent":
            dx, dy = dx * width / 100, dy * height / 100
        translation = np.array(((1, 0, dx / max(1, width - 1)),
                                (0, 1, dy / max(1, height - 1)), (0, 0, 1)))
        matrix = translation @ matrix
    homogeneous = np.column_stack((UNIT_CORNERS, np.ones(4))) @ matrix.T
    if not np.isfinite(matrix).all() or np.any(homogeneous[:, 2] <= 1e-6):
        raise ValueError("Transform perspective crosses the image horizon")
    return matrix


def apply_transform(
    image: np.ndarray, params: dict[str, Any], labels: list[DisplayLabel],
) -> tuple[np.ndarray, list[DisplayLabel]]:
    """
    Resize then warp display pixels; retain only annotations whose anchors remain visible.
    """
    height, width = image.shape[:2]
    size = (max(1, round(width * params["width"] / 100)),
            max(1, round(height * params["height"] / 100)))
    if params["resize_mode"] == "canvas":
        size = width, height
    if size[0] * size[1] > MAX_PIXELS:
        raise ValueError("Pipeline exceeds the 4 megapixel image limit; reduce Transform size")
    matrix = transform_matrix(params, size)
    methods = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
               "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
    method = methods[params["method"]]
    output = cv2.resize(image, size, interpolation=method) if size != (width, height) else image
    if params["horizontal"]:
        output = cv2.flip(output, 1)
    if params["vertical"]:
        output = cv2.flip(output, 0)
    if has_warp(params):
        # Mirrors have already been applied exactly, avoiding resampling old Mirror nodes.
        warp_params = {**params, "horizontal": False, "vertical": False}
        normalizer = np.diag((max(1, size[0] - 1), max(1, size[1] - 1), 1))
        pixel_matrix = normalizer @ transform_matrix(warp_params, size) @ np.linalg.inv(normalizer)
        output = cv2.warpPerspective(output, pixel_matrix, size, flags=method,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    transformed = []
    for label in labels:
        anchor = matrix @ np.array((label.x, label.y, 1))
        x, y = anchor[:2] / anchor[2]
        if 0 <= x <= 1 and 0 <= y <= 1:
            transformed.append(DisplayLabel(label.text, float(x), float(y)))
    return output, transformed
