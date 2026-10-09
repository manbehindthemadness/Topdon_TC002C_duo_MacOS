"""
Display-only markers, labels, and calibration overlays.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..render import (
    RenderedThermalFrame,
)
from .layout import image_position_at
from .spots import MousePicker, SampleSpots, spot_hit


def draw_contrasting_overlay(
    image: np.ndarray, mask: np.ndarray, text_mask: np.ndarray
) -> np.ndarray:
    """
    Draw inverted crosshairs and stable white labels with a black outline.
    """
    result = image.copy()
    x, y, width, height = cv2.boundingRect(cv2.max(mask, text_mask))
    if width == 0 or height == 0:
        return result
    # Limit compositing to the overlay bounds, including the two-pixel text outline.
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(image.shape[1], x + width + 2), min(image.shape[0], y + height + 2)
    region = image[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    text_mask = text_mask[y0:y1, x0:x1]
    inverted = 255 - region
    # Inversion alone disappears on middle gray. A black/white halo also
    # separates the strokes from busy thermal detail without hiding a whole box.
    luminance = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    halo_color = np.where(luminance >= 128, 0, 255).astype(np.uint8)
    halo_mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    halo_alpha = (halo_mask.astype(np.float32) / 255.0)[..., None]
    outlined = region * (1.0 - halo_alpha) + halo_color[..., None] * halo_alpha
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    composited = outlined * (1.0 - alpha) + inverted * alpha
    # Keep each digit a consistent color even when it straddles hot/cold detail.
    text_outline = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    outline_alpha = (text_outline.astype(np.float32) / 255.0)[..., None]
    text_alpha = (text_mask.astype(np.float32) / 255.0)[..., None]
    composited *= 1.0 - outline_alpha
    result[y0:y1, x0:x1] = np.rint(composited * (1.0 - text_alpha) + 255.0 * text_alpha).astype(
        np.uint8
    )
    return result


def place_temperature_label(
    text: str,
    anchor: tuple[int, int],
    image_shape: tuple[int, ...],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """
    Try nearby positions first, reserving the text outline and a small gap.
    """
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    padding = 3
    box_width, box_height = width + padding * 2, height + baseline + padding * 2
    image_height, image_width = image_shape[:2]
    ax, ay = anchor
    best = None
    seen = set()
    row_step = box_height + 4
    for ring in range(max(image_height, image_width) // row_step + 1):
        above = ay - 8 - height - padding - ring * row_step
        below = ay + 8 + ring * row_step
        right = ax + 8 + ring * (box_width + 4)
        left = ax - 8 - box_width - ring * (box_width + 4)
        for x, y in (
            (right, above),
            (right, below),
            (left, above),
            (left, below),
            (ax - box_width // 2, above),
            (ax - box_width // 2, below),
            (right, ay - box_height // 2),
            (left, ay - box_height // 2),
        ):
            x = max(0, min(x, image_width - box_width))
            y = max(0, min(y, image_height - box_height))
            if (x, y) in seen:
                continue
            seen.add((x, y))
            rect = (x, y, x + box_width, y + box_height)
            overlap = sum(
                max(0, min(rect[2], other[2]) - max(x, other[0]))
                * max(0, min(rect[3], other[3]) - max(y, other[1]))
                for other in occupied
            )
            origin = (x + padding, y + height + padding)
            if overlap == 0:
                return origin, rect
            distance = (x + box_width / 2 - ax) ** 2 + (y + box_height / 2 - ay) ** 2
            score = (overlap, distance)
            if best is None or score < best[0]:
                best = (score, origin, rect)
    # When the image is completely crowded, choose the least obstructed position.
    assert best is not None
    return best[1], best[2]


def spot_label_layout(
    rendered: RenderedThermalFrame, spots: SampleSpots, scale: int, temperature_unit: str
) -> tuple[
    list[tuple[str, tuple[int, int], tuple[int, int, int, int]]], list[tuple[int, int, int, int]]
]:
    """
    Spot label layout.
    """
    active = spots.active
    anchors = [(x * scale + scale // 2, y * scale + scale // 2) for _number, (x, y) in active]
    occupied = [(x - 6, y - 6, x + 7, y + 7) for x, y in anchors]
    labels = []
    for (number, (sensor_x, sensor_y)), anchor in zip(active, anchors, strict=True):
        temperature = float(rendered.temperatures_celsius[sensor_y, sensor_x])
        if temperature_unit == "F":
            temperature = temperature * 9.0 / 5.0 + 32.0
        reading = f"{temperature:.2f}" if np.isfinite(temperature) else "--"
        text = f"{number}: {reading} {temperature_unit}"
        origin, rect = place_temperature_label(text, anchor, rendered.image.shape, occupied)
        occupied.append(rect)
        labels.append((text, origin, rect))
    return labels, occupied


def draw_label_leader(
    mask: np.ndarray, anchor: tuple[int, int], rect: tuple[int, int, int, int]
) -> None:
    """
    Connect every temperature label to its sampled pixel.
    """
    endpoint = (
        min(max(anchor[0], rect[0]), rect[2] - 1),
        min(max(anchor[1], rect[1]), rect[3] - 1),
    )
    cv2.line(mask, anchor, endpoint, 255, 1, cv2.LINE_AA)


def draw_sample_spots(
    image: np.ndarray,
    rendered: RenderedThermalFrame,
    spots: SampleSpots,
    scale: int,
    temperature_unit: str = "C",
) -> np.ndarray:
    """
    Draw sample spots.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    text_mask = np.zeros_like(mask)
    labels, _occupied = spot_label_layout(rendered, spots, scale, temperature_unit)
    active = spots.active
    for (_number, (sensor_x, sensor_y)), (text, origin, rect) in zip(active, labels, strict=True):
        anchor = (sensor_x * scale + scale // 2, sensor_y * scale + scale // 2)
        draw_label_leader(mask, anchor, rect)
        cv2.putText(
            text_mask,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            255,
            1,
            cv2.LINE_AA,
        )
    # Union the markers so overlapping spots are inverted only once.
    for _number, (sensor_x, sensor_y) in active:
        cv2.drawMarker(
            mask,
            (sensor_x * scale + scale // 2, sensor_y * scale + scale // 2),
            255,
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_8,
        )
    return draw_contrasting_overlay(image, mask, text_mask)


def draw_distance_selection(image: Any, calibration: Any, scale: Any) -> Any:
    """
    Draw distance selection.
    """
    if not calibration.corners:
        return image
    mask = np.zeros(image.shape[:2], np.uint8)
    text_mask = np.zeros_like(mask)
    points = np.rint(np.asarray(calibration.corners) * scale).astype(np.int32)
    if len(points) > 1:
        cv2.polylines(mask, [points], len(points) == 4, 255, 1, cv2.LINE_AA)
    for index, (x, y) in enumerate(points):
        cv2.drawMarker(mask, (int(x), int(y)), 255, cv2.MARKER_CROSS, 11, 1)
        cv2.putText(
            text_mask,
            str(index + 1),
            (int(x) + 6, max(12, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            255,
            1,
            cv2.LINE_AA,
        )
    return draw_contrasting_overlay(image, mask, text_mask)


def draw_reflector_target(image: Any, calibration: Any, scale: Any, unit: Any) -> Any:
    """
    Draw reflector target.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    text_mask = np.zeros_like(mask)
    center = (
        (image.shape[1] // scale // 2) * scale + scale // 2,
        (image.shape[0] // scale // 2) * scale + scale // 2,
    )
    radius = calibration.RADIUS * scale
    cv2.circle(mask, center, radius, 255, 1, cv2.LINE_AA)
    label = "Reflector"
    if calibration.measured_celsius is not None:
        value = calibration.measured_celsius
        if unit == "F":
            value = value * 1.8 + 32
        label += f" {float(value):.1f} {unit}"
    cv2.putText(
        text_mask,
        label,
        (max(2, center[0] - 55), max(15, center[1] - radius - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        255,
        1,
        cv2.LINE_AA,
    )
    return draw_contrasting_overlay(image, mask, text_mask)


def draw_emissivity_point(image: Any, calibration: Any, scale: Any, unit: Any) -> Any:
    """
    Draw emissivity point.
    """
    if calibration.point is None:
        return image
    x, y = (coordinate * scale + scale // 2 for coordinate in calibration.point)
    mask = np.zeros(image.shape[:2], np.uint8)
    text_mask = np.zeros_like(mask)
    cv2.drawMarker(mask, (x, y), 255, cv2.MARKER_CROSS, 11, 1)
    temperature = calibration.measured_celsius
    label = "Reference"
    if temperature is not None:
        display = temperature * 1.8 + 32 if unit == "F" else temperature
        label = f"Ref {float(display):.2f} {unit}"
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = max(2, min(x + 10, image.shape[1] - width - 2))
    label_y = y - 10 if y > height + 12 else y + height + 12
    label_y = min(label_y, image.shape[0] - baseline - 2)
    cv2.putText(
        text_mask, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA
    )
    return draw_contrasting_overlay(image, mask, text_mask)


def draw_picker(
    rendered: RenderedThermalFrame,
    picker: MousePicker,
    scale: int,
    viewport_size: tuple[int, int] | None = None,
    toolbar_height: int = 0,
    temperature_unit: str = "C",
    pointer_over_image: bool | None = None,
    spots: SampleSpots | None = None,
    dragging_spot: bool = False,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """
    Draw picker.
    """
    image = rendered.image.copy()
    if pointer_over_image is False or picker.x is None or picker.y is None:
        return image, None
    position = image_position_at(picker.x, picker.y, image.shape, viewport_size, toolbar_height)
    if position is None:
        return image, None
    image_x, image_y = position
    sensor_x = min(max(image_x // scale, 0), rendered.temperatures_celsius.shape[1] - 1)
    sensor_y = min(max(image_y // scale, 0), rendered.temperatures_celsius.shape[0] - 1)
    temperature = float(rendered.temperatures_celsius[sensor_y, sensor_x])
    if temperature_unit == "F":
        temperature = temperature * 9.0 / 5.0 + 32.0

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    text_mask = np.zeros_like(mask)
    cv2.drawMarker(
        mask,
        (image_x, image_y),
        255,
        markerType=cv2.MARKER_CROSS,
        markerSize=9,
        thickness=1,
        line_type=cv2.LINE_8,
    )
    if dragging_spot or (
        spots is not None
        and spot_hit(position, spots, image.shape, scale, viewport_size) is not None
    ):
        return draw_contrasting_overlay(image, mask, text_mask), (sensor_x, sensor_y)
    reading = f"{temperature:.2f}" if np.isfinite(temperature) else "--"
    text = f"({sensor_x}, {sensor_y}) {reading} {temperature_unit}"
    _, occupied = spot_label_layout(rendered, spots or SampleSpots(), scale, temperature_unit)
    occupied.append((image_x - 6, image_y - 6, image_x + 7, image_y + 7))
    origin, rect = place_temperature_label(text, (image_x, image_y), image.shape, occupied)
    draw_label_leader(mask, (image_x, image_y), rect)
    cv2.putText(
        text_mask,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        255,
        1,
        cv2.LINE_AA,
    )
    return draw_contrasting_overlay(image, mask, text_mask), (sensor_x, sensor_y)


def draw_control_instructions(image: np.ndarray) -> np.ndarray:
    """
    Draw a translucent keyboard/mouse help panel over the image.
    """
    result = image.copy()
    panel_width = min(390, result.shape[1] - 20)
    panel_height = min(349, result.shape[0] - 20)
    x0, y0 = 10, result.shape[0] - panel_height - 10
    x1, y1 = x0 + panel_width, y0 + panel_height

    overlay = result.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (8, 10, 16), -1)
    cv2.addWeighted(overlay, 0.82, result, 0.18, 0, result)
    cv2.rectangle(result, (x0, y0), (x1, y1), (120, 130, 150), 1)

    lines = (
        ("Controls", (255, 255, 255)),
        ("Mouse move   Inspect pixel temperature", (210, 215, 225)),
        ("P / Add spots  Toggle spot placement", (210, 215, 225)),
        ("Camera Hardware ambient and image controls", (210, 215, 225)),
        ("S            Save image data", (210, 215, 225)),
        ("C            Open Capture controls", (210, 215, 225)),
        ("O            Rotate 90 degrees clockwise", (210, 215, 225)),
        ("F            Toggle metric / imperial", (210, 215, 225)),
        ("V            Open Camera controls", (210, 215, 225)),
        ("G            Show / hide graph area", (210, 215, 225)),
        ("L            Start / stop CSV logging", (210, 215, 225)),
        ("Space        Hide controls", (210, 215, 225)),
        ("Q / Esc      Quit", (210, 215, 225)),
    )
    for index, (line, color) in enumerate(lines):
        cv2.putText(
            result,
            line,
            (x0 + 14, y0 + 25 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48 if index else 0.58,
            color,
            1,
            cv2.LINE_AA,
        )
    return result
