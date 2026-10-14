"""
Legacy bitmap graph toolbar geometry and drawing, shared by controller hit testing.
"""

import cv2
import numpy as np

from .graph_settings import GRAPH_INTERVAL

Rectangle = tuple[int, int, int, int]


def graph_log_button_rect(width: int) -> Rectangle:
    """
    Locate the CSV logging button within the legacy graph toolbar.
    """
    button_width = min(150, max(90, width // 3))
    rectangle = max(0, width - button_width - 6), 4, width - 6, 28
    return rectangle


def graph_interval_rect(width: int) -> Rectangle:
    """
    Locate the update interval editor beside the CSV logging control.
    """
    left = graph_log_button_rect(width)[0]
    rectangle = max(0, left - 88), 4, max(0, left - 8), 28
    return rectangle


def graph_config_rect(width: int) -> Rectangle:
    """
    Locate the graph configuration button beside the interval editor.
    """
    left = graph_interval_rect(width)[0]
    rectangle = max(0, left - 152), 4, max(0, left - 84), 28
    return rectangle


def graph_reset_rect(width: int) -> Rectangle:
    """
    Locate the history reset button beside graph configuration.
    """
    left = graph_config_rect(width)[0]
    rectangle = max(0, left - 68), 4, max(0, left - 4), 28
    return rectangle


def draw_graph_logging_control(
    canvas: np.ndarray,
    logging: bool = False,
    pending: bool = False,
    interval: float = GRAPH_INTERVAL,
    edit_text: str | None = None,
) -> None:
    """
    Draw live legacy toolbar controls into the caller-owned graph bitmap.
    """
    x0, y0, x1, y1 = graph_log_button_rect(canvas.shape[1])
    label = "Cancel logging" if pending else "Stop logging" if logging else "Log to CSV"
    color = (65, 65, 150) if logging else (47, 51, 61)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_width = cv2.getTextSize(label, font, 0.4, 1)[0][0]
    scale = 0.4 * min(1, max(1, x1 - x0 - 10) / text_width)
    cv2.putText(canvas, label, (x0 + 5, y0 + 16), font, scale, (235, 238, 245), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = graph_config_rect(canvas.shape[1])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (47, 51, 61), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
    cv2.putText(canvas, "Config", (x0 + 5, y0 + 16), font, 0.4, (235, 238, 245), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = graph_reset_rect(canvas.shape[1])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (47, 51, 61), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
    cv2.putText(
        canvas,
        "Reset",
        (x0 + 5, y0 + 16),
        font,
        0.4,
        (125, 130, 140) if logging or pending else (235, 238, 245),
        1,
        cv2.LINE_AA,
    )
    x0, y0, x1, y1 = graph_interval_rect(canvas.shape[1])
    border = (90, 190, 240) if edit_text is not None else (105, 112, 128)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (35, 38, 46), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 1)
    value = f"{interval:g}" if edit_text is None else edit_text + "|"
    cv2.putText(
        canvas,
        value,
        (x0 + 5, y0 + 16),
        font,
        0.4,
        (125, 130, 140) if logging or pending else (235, 238, 245),
        1,
        cv2.LINE_AA,
    )
    if x0 >= 85:
        cv2.putText(
            canvas, "Update (s)", (x0 - 80, y0 + 16), font, 0.4, (185, 190, 205), 1, cv2.LINE_AA
        )
