"""
Verify nesting, fixed/dynamic box styling and optional labels without weights.
"""

from typing import Any

import numpy as np
import pytest
from support.classless import helper
from support.classless import package as package  # noqa: PLC0414 - pytest fixture re-export

from topdon_duo.custom_nodes.runtime import CustomPackage
from topdon_duo.desktop_app.overlays import draw_contrasting_overlay


def test_inside_out_keeps_leaves_before_confidence_limit(package: CustomPackage) -> None:
    """
    Remove complete parents while preserving partial overlaps and independent boxes.
    """
    parent = (0.1, 0.1, 0.9, 0.9, 0.99)
    child = (0.2, 0.2, 0.8, 0.8, 0.8)
    leaf = (0.3, 0.3, 0.4, 0.4, 0.4)
    peer = (0.0, 0.0, 0.15, 0.15, 0.9)
    isolated = (0.91, 0.91, 0.99, 0.99, 0.5)
    boxes = [parent, child, leaf, peer, isolated]
    select = helper(package, "nesting").select_detections
    assert set(select(boxes, "Inside-out", 10)) == {leaf, peer, isolated}
    assert select(boxes, "Inside-out", 1) == [peer]
    assert boxes == [parent, child, leaf, peer, isolated]
    same_bounds = (*leaf[:4], 0.6)
    assert set(select([leaf, same_bounds], "Inside-out", 10, duplicate_iou=None)) == {leaf, same_bounds}


def test_outside_in_retains_children_and_paints_parents_first(package: CustomPackage) -> None:
    """
    Nesting depth, rather than confidence order, determines painting order.
    """
    parent = (0.1, 0.1, 0.9, 0.9, 0.4)
    child = (0.2, 0.2, 0.8, 0.8, 0.8)
    leaf = (0.3, 0.3, 0.4, 0.4, 0.9)
    select = helper(package, "nesting").select_detections
    assert select([leaf, parent, child], "Outside-in", 10) == [parent, child, leaf]
    assert select([leaf, parent, child], "Outside-in", 1) == [leaf]


def test_fixed_colors_transparent_fill_and_input_ownership(package: CustomPackage) -> None:
    """
    Apply RGB controls in BGR order and blend overlapping fills only once.
    """
    image = np.full((40, 40, 3), [20, 40, 60], np.float32)
    box = (0.25, 0.25, 0.75, 0.75, 0.9)
    draw = helper(package, "drawing").draw_detections
    result = draw(image, [box, box], 1, False, "#0000ff", "#ff0000", 0.5)
    assert np.array_equal(result[10, 10], [255, 0, 0])
    assert np.array_equal(result[20, 20], [10, 20, 158])
    assert np.array_equal(result[0, 0], [20, 40, 60])
    assert np.array_equal(image[10, 10], [20, 40, 60])
    unfilled = draw(image, [box], 1, False, "#00ff00", "dynamic", 0)
    assert np.array_equal(unfilled[20, 20], image[20, 20])


@pytest.mark.parametrize("intensity", [0, 127, 255])
def test_dynamic_border_matches_measurement_overlay_and_dynamic_fill(
    package: CustomPackage, intensity: int
) -> None:
    """
    Match measurement inversion/halos exactly, including visibility on middle gray.
    """
    image = np.full((40, 40, 3), intensity, np.uint8)
    box = (0.25, 0.25, 0.75, 0.75, 0.9)
    draw = helper(package, "drawing").draw_detections
    mask = np.zeros((40, 40), np.uint8)
    # Independent rectangle reference: coordinates 10 through 30, inclusive.
    mask[10, 10:31] = mask[30, 10:31] = 255
    mask[10:31, 10] = mask[10:31, 30] = 255
    reference = draw_contrasting_overlay(image, mask, np.zeros_like(mask))
    assert np.array_equal(draw(image, [box], 1, False, "dynamic"), reference)
    filled = draw(image, [box], 1, False, "#ff0000", "dynamic", 1)
    assert np.all(filled[20, 20] == 255 - intensity)


def test_labels_optional_and_off_by_default(package: CustomPackage) -> None:
    """
    Enabling confidence labels changes only the overlay, not detection settings.
    """
    options = helper(package, "settings").Settings.from_config({})
    assert not options.show_scores
    assert options.fill_opacity == 0
    image = np.zeros((80, 80, 3), np.uint8)
    boxes = [(0.5, 0.25, 0.9, 0.75, 0.87)]
    draw = helper(package, "drawing").draw_detections
    without = draw(image, boxes, 1, False, "dynamic")
    with_labels = draw(image, boxes, 1, True, "dynamic")
    assert not without[:35].any()
    assert with_labels[:35].any()
    assert np.array_equal(without[45:60], with_labels[45:60])


@pytest.mark.parametrize("config", [
    {"nesting": "unknown"}, {"fill_opacity": float("nan")}, {"fill_opacity": True},
    {"fill_opacity": 2}, {"border_color": "red"}, {"fill_color": "#gg0000"},
])
def test_invalid_overlay_settings_rejected(package: CustomPackage, config: dict[str, Any]) -> None:
    """
    Refuse invalid styling before model loading or drawing.
    """
    with pytest.raises(ValueError):
        helper(package, "settings").Settings.from_config(config)
