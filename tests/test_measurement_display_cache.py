"""Local raster references, cache invalidation and frame-scoped sampling regressions."""

from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from based_thermals.desktop_app.overlays import draw_contrasting_overlay, place_temperature_label
from based_thermals.desktop_app.spots import SampleSpots
from based_thermals.measurements import display_cache, overlays
from based_thermals.measurements.appearance import styled_shapes
from based_thermals.measurements.display_masks import shape_masks
from based_thermals.measurements.overlays import display_boundary, draw_boundary, label_layout
from based_thermals.measurements.paths import initial_region
from based_thermals.render import RenderedThermalFrame, TemperatureStats


def full_canvas_reference(image: np.ndarray, region: dict, center: tuple[int, int]) -> np.ndarray:
    """
    Reproduce the original full-canvas alpha/contrast formula independently of local rasters.
    """
    lines = np.zeros(image.shape[:2], np.uint8)
    draw_boundary(lines, region, center, 3)
    filled = np.zeros_like(lines)
    points = display_boundary(region, center, 3)
    # The independent reference intentionally retains the legacy full-canvas formula.
    # noinspection DuplicatedCode
    cv2.fillPoly(filled, [points], 255, lineType=cv2.LINE_AA)
    if region["exclude"]:
        perimeter = np.zeros_like(lines)
        cv2.polylines(perimeter, [points], True, 255, 1, cv2.LINE_AA)
        filled[perimeter > 0] = 0
    opacity, color = region["opacity"], region["line_color"]
    result = image.copy()
    if color == "dynamic":
        inverted = 255 - image
        alpha = (filled.astype(np.float32) / 255)[..., None]
        target = np.rint(image * (1 - alpha) + inverted * alpha).astype(np.uint8)
        outlined = draw_contrasting_overlay(image, lines, np.zeros_like(lines))
        halo = cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0
        target[halo] = outlined[halo]
        coverage = cv2.dilate(cv2.max(filled, lines), np.ones((3, 3), np.uint8))
        x, y, width, height = cv2.boundingRect(coverage)
        view = image[y : y + height, x : x + width]
        result[y : y + height, x : x + width] = np.rint(
            view * (1 - opacity) + target[y : y + height, x : x + width] * opacity,
        ).astype(np.uint8)
    else:
        target = np.array([int(color[i : i + 2], 16) for i in (5, 3, 1)], dtype=np.uint8)
        alpha = (cv2.max(lines, filled).astype(np.float32) * (opacity / 255))[..., None]
        result = np.rint(image * (1 - alpha) + target * alpha).astype(np.uint8)
    return result


@pytest.mark.parametrize("kind", ["spot", "rectangle", "oval", "polygon", "bezier"])
@pytest.mark.parametrize("color", ["dynamic", "#ab1234"])
@pytest.mark.parametrize("exclude", [False, True])
def test_local_compositing_matches_full_canvas_reference(
    kind: str, color: str, exclude: bool
) -> None:
    """
    Preserve original AA borders, dashed gaps, fill and opacity on clipped sensor edges.
    """
    center, region = initial_region(kind, (32, 24))
    region.update(fill=True, opacity=0.37, line_color=color, exclude=exclude, labels=[])
    if kind == "spot":
        center, region["radius"] = (0, 0), 3
    else:
        # Include a canvas edge to exercise clipping without moving native coordinates.
        region["points"][0] = [0, 0]
    image = np.random.default_rng(23).integers(0, 256, (72, 96, 3), dtype=np.uint8)
    spots = SampleSpots(native_size=(32, 24))
    number = spots.create_region(center, region, (32, 24))
    expected = full_canvas_reference(image, region, center)
    result, mask = styled_shapes(image, spots, 3, draw_contrasting_overlay)
    np.testing.assert_array_equal(result, expected)
    assert not np.any(mask)
    assert (
        shape_masks(spots, number, center, 3, (image.shape[0], image.shape[1])).lines.nbytes
        < image.size
    )


def test_masks_reuse_invalidate_prune_and_respect_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Bound retained raster memory and invalidate geometry, scale, fill and exclusion changes.
    """
    spots = SampleSpots(native_size=(32, 24))
    center, region = initial_region("rectangle", (32, 24))
    number = spots.create_region(center, region, (32, 24))
    first = shape_masks(spots, number, center, 3, (72, 96))
    assert shape_masks(spots, number, center, 3, (72, 96)) is first
    spots.configure_region(number, {**region, "line_color": "#ff0000", "z_index": 4}, (32, 24))
    assert shape_masks(spots, number, center, 3, (72, 96)) is first
    for changes in ({"fill": True}, {"exclude": True}, {"points": [[1, 1], [10, 10]]}):
        previous = shape_masks(spots, number, center, 3, (72, 96))
        spots.configure_region(number, {**spots.region(number), **changes}, (32, 24))
        assert shape_masks(spots, number, center, 3, (72, 96)) is not previous
    old = shape_masks(spots, number, center, 3, (72, 96))
    assert shape_masks(spots, number, center, 2, (48, 64)) is not old
    monkeypatch.setattr(display_cache, "MASK_CACHE_BYTES", first.byte_size)
    for _index in range(4):
        added = spots.create_region(center, region, (32, 24))
        shape_masks(spots, added, center, 3, (72, 96))
    assert spots.display_cache.mask_bytes <= first.byte_size
    spots.clear()
    spots.display_cache.prune(spots.numbers)
    assert spots.display_cache.mask_bytes == 0
    assert not spots.display_cache.masks and not spots.display_cache.boundaries


def test_readings_are_shared_only_inside_one_explicit_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Share cursor/shape/graph sampling but refresh reused temperature buffers each frame.
    """
    spots = SampleSpots(native_size=(32, 24))
    center, region = initial_region("rectangle", (32, 24))
    number = spots.create_region(center, region, (32, 24))
    temperatures = np.full((24, 32), 20.0)
    rendered = RenderedThermalFrame(
        np.zeros((72, 96, 3), np.uint8), TemperatureStats(20, 20, 20, 20), temperatures, None
    )
    readings = Mock(wraps=spots.readings)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(spots, "readings", readings)
    spots.display_cache.begin_frame()
    label_layout(rendered, spots, 3, "C", place_temperature_label)
    spots.graph_readings(temperatures)
    label_layout(rendered, spots, 3, "C", place_temperature_label)
    assert readings.call_count == 1
    spots.configure_region(number, {**region, "exclude": True}, (32, 24))
    assert spots.graph_readings(temperatures) == ()
    assert readings.call_count == 2
    spots.configure_region(number, region, (32, 24))
    spots.display_cache.end_frame()
    temperatures[:] = 30
    assert spots.graph_readings(temperatures)[0][1][1] == 30
    spots.display_cache.begin_frame()
    temperatures[:] = 40
    assert spots.graph_readings(temperatures)[0][1][1] == 40
    spots.display_cache.end_frame()


def test_label_layout_reuses_bounds_while_text_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Reuse stable label positions but invalidate geometry, units, dimensions and text bounds.
    """
    spots = SampleSpots(native_size=(32, 24))
    center, region = initial_region("rectangle", (32, 24))
    spots.create_region(center, region, (32, 24))
    temperatures = np.full((24, 32), 20.0)
    rendered = RenderedThermalFrame(
        np.zeros((72, 96, 3), np.uint8), TemperatureStats(20, 20, 20, 20), temperatures, None
    )
    placer = Mock(wraps=place_temperature_label)
    first, occupied = label_layout(rendered, spots, 3, "C", placer)
    temperatures[:] = 21
    second, again = label_layout(rendered, spots, 3, "C", placer)
    assert placer.call_count == 1 and first[0].rectangle == second[0].rectangle
    assert first[0].text != second[0].text and occupied == again
    temperatures[:] = 100
    label_layout(rendered, spots, 3, "C", placer)
    assert placer.call_count == 2
    label_layout(rendered, spots, 3, "F", placer)
    assert placer.call_count == 3
    spots.move(1, (center[0] + 1, center[1]))
    label_layout(rendered, spots, 3, "F", placer)
    assert placer.call_count == 4
    resized = replace(rendered, image=np.zeros((96, 128, 3), np.uint8))
    label_layout(resized, spots, 4, "F", placer)
    assert placer.call_count == 5
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(overlays, "LABEL_FONT_SCALE", 0.6)
    label_layout(resized, spots, 4, "F", placer)
    assert placer.call_count == 6


def test_colored_z_layers_do_not_flush_empty_dynamic_masks() -> None:
    """
    Distinct colored layers do no full-frame contrast work when masks are empty.
    """
    spots = SampleSpots(native_size=(32, 24))
    center, region = initial_region("rectangle", (32, 24))
    for index in range(4):
        spots.create_region(center, {**region, "line_color": "#ff0000", "z_index": index}, (32, 24))
    contrast = Mock(wraps=draw_contrasting_overlay)
    styled_shapes(np.zeros((72, 96, 3), np.uint8), spots, 3, contrast)
    contrast.assert_not_called()
