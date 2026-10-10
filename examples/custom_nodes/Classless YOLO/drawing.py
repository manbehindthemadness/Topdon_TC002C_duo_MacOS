"""
Draw classless detections while preserving the original display geometry.
"""

import cv2
import numpy as np

from topdon_duo.custom_nodes.colors import color_bgr
from topdon_duo.custom_nodes.labels import defer_labels
from topdon_duo.desktop_app.overlays import draw_contrasting_overlay

from .detector import Detection
from .labels import confidence_mask


def draw_detections(
    image: np.ndarray, detections: list[Detection], thickness: int, show_scores: bool,
    border_color: str = "#00ff00", fill_color: str = "#00ff00", fill_opacity: float = 0.0,
) -> np.ndarray:
    """
    Map normalized model boxes back to the entire source view and annotate a copy.
    """
    output = np.clip(image, 0, 255).astype(np.uint8)
    height, width = output.shape[:2]
    border = np.zeros((height, width), np.uint8)
    text = np.zeros_like(border)
    labels: list[tuple[str, tuple[int, int]]] = []
    fill = np.zeros_like(border)
    for y1, x1, y2, x2, score in detections:
        left, top = min(width - 1, round(x1 * width)), min(height - 1, round(y1 * height))
        right, bottom = min(width - 1, round(x2 * width)), min(height - 1, round(y2 * height))
        cv2.rectangle(border, (left, top), (right, bottom), 255, thickness)
        if fill_opacity > 0:
            cv2.rectangle(fill, (left, top), (right, bottom), 255, cv2.FILLED)
        if show_scores:
            labels.append((f"{score:.2f}", (left, top)))
    if labels and not defer_labels(labels, (height, width)):
        text = confidence_mask((height, width), labels)
    if fill_opacity > 0:
        color = 255 - output if fill_color == "dynamic" else np.array(color_bgr(fill_color))
        alpha = (fill.astype(np.float32) / 255.0 * fill_opacity)[..., None]
        output = np.rint(output * (1.0 - alpha) + color * alpha).astype(np.uint8)
    if border_color == "dynamic":
        return draw_contrasting_overlay(output, border, text)
    color = np.array(color_bgr(border_color))
    alpha = (border.astype(np.float32) / 255.0)[..., None]
    output = np.rint(output * (1.0 - alpha) + color * alpha).astype(np.uint8)
    if labels:
        output = draw_contrasting_overlay(output, np.zeros_like(border), text)
    return output
