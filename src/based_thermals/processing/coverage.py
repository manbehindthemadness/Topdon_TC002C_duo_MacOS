"""
Track branch footprints independently of display colors for regional replacement.
"""

from typing import Any

import cv2
import numpy as np

from ..custom_nodes.labels import DisplayLabel
from ..image_transform import apply_transform, has_warp
from .images import combine_strength, resize


def resize_coverage(coverage: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """
    Match a branch's current working resolution; None represents a full canvas.
    """
    if coverage is not None and coverage.shape != shape:
        coverage = resize(coverage, (shape[1], shape[0]), cv2.INTER_NEAREST)
    return coverage


def transform_coverage(
    coverage: np.ndarray | None, shape: tuple[int, int], params: dict[str, Any],
) -> np.ndarray | None:
    """
    Move a binary footprint through the exact display transform, retaining true black pixels.
    """
    if coverage is None:
        if not has_warp(params):
            return None
        coverage = np.ones(shape, np.uint8)
    transformed, _ = apply_transform(coverage, {**params, "method": "nearest"}, [])
    return transformed


def combine_coverage(
    current: np.ndarray | None, incoming: np.ndarray | None, shape: tuple[int, int],
    params: dict[str, Any], mask: np.ndarray | None,
) -> np.ndarray | None:
    """
    Propagate occupied pixels through nested blends without modifying shared input footprints.
    """
    mode = params["mode"]
    if mode == "mask" or mode != "replace" and params["opacity"] == 0:
        return current
    if current is None and (mode != "opacity" or incoming is None):
        return None
    current = resize_coverage(current, shape)
    incoming = resize_coverage(incoming, shape)
    strength = combine_strength(shape, params, mask, incoming if mode == "replace" else None)
    base_present = current > 0 if current is not None else True
    input_present = incoming > 0 if incoming is not None else True
    if mode in ("opacity", "replace"):
        occupied = (base_present & (strength < 1)) | (input_present & (strength > 0))
    else:
        occupied = base_present | (input_present & (strength > 0))
    if np.all(occupied):
        return None
    return np.broadcast_to(occupied, shape).astype(np.uint8)


def replace_labels(
    current: list[DisplayLabel], incoming: list[DisplayLabel], shape: tuple[int, int],
    params: dict[str, Any], mask: np.ndarray | None, region: np.ndarray | None,
) -> list[DisplayLabel]:
    """
    Remove overwritten base anchors and admit incoming labels only in the copied region.
    """
    if not current and not incoming:
        return []
    strength = combine_strength(shape, params, mask, region)

    def at(label: DisplayLabel) -> float:
        """
        Sample the normalized anchor using the same coverage as its branch pixels.
        """
        if not isinstance(strength, np.ndarray):
            return float(strength)
        height, width = shape
        x = min(width - 1, max(0, round(label.x * (width - 1))))
        y = min(height - 1, max(0, round(label.y * (height - 1))))
        return float(strength[y, x])

    labels = [label for label in current if at(label) < 1]
    labels.extend(label for label in incoming if at(label) > 0)
    return labels
