"""
Regression coverage for confidence ranges, approximate nesting and visible labels.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.classless import helper
from support.classless import package as package  # noqa: PLC0414 - pytest fixture re-export

from topdon_duo.custom_nodes.runtime import CustomPackage


def test_inside_out_accepts_edge_crossing_children_with_adjustable_coverage(
    package: CustomPackage,
) -> None:
    """
    Remove near-containing parents missed by the previous exact-coordinate test.
    """
    parent = (0.1, 0.1, 0.9, 0.9, 0.95)
    child = (0.5, 0.5, 0.92, 0.92, 0.7)
    select = helper(package, "nesting").select_detections
    assert select([parent, child], "Inside-out", 10) == [child]
    assert select([parent, child], "Inside-out", 10, coverage=1) == [parent, child]
    assert select([parent, child], "Outside-in", 10) == [parent, child]


def test_peer_suppression_keeps_best_duplicate_and_preserves_children(package: CustomPackage) -> None:
    """
    Collapse high-overlap peers by confidence while leaving real hierarchies intact.
    """
    first = (0.2, 0.2, 0.6, 0.6, 0.4)
    second = (0.21, 0.21, 0.61, 0.61, 0.8)
    child = (0.3, 0.3, 0.4, 0.4, 0.7)
    select = helper(package, "nesting").select_detections
    assert select([first, second], "Inside-out", 10) == [second]
    assert len(select([first, second], "Inside-out", 10, duplicate_iou=None)) == 2
    assert select([second, child], "Outside-in", 10) == [second, child]
    assert select([second, child], "Inside-out", 1) == [child]


def test_confidence_range_toggle_and_filter_order(
    package: CustomPackage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Apply both inclusive confidence bounds before nesting and count limits.
    """
    model = tmp_path / "model.onnx"
    model.touch()
    parent = (0.1, 0.1, 0.9, 0.9, 0.7)
    child = (0.3, 0.3, 0.4, 0.4, 0.9)
    low = (0.91, 0.91, 0.99, 0.99, 0.1)
    engine = Mock(key=(str(model), ("CPUExecutionProvider",)))

    def detect(_image: np.ndarray, threshold: float) -> list[tuple[float, float, float, float, float]]:
        """
        Mimic the detector's documented minimum-confidence boundary.
        """
        return [box for box in (parent, child, low) if box[4] >= threshold]

    engine.detect.side_effect = detect
    module = sys.modules[package.name]
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(module, "Detector", Mock(return_value=engine))
    drawing = Mock(side_effect=lambda pixels, *args: pixels)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(module, "draw_detections", drawing)
    image = np.zeros((40, 40, 3), np.float32)
    config: dict[str, Any] = {"model_path": str(model), "score_threshold": 0.2,
                              "score_maximum": 0.7, "nesting": "Inside-out", "max_detections": 1}
    package.process(image, config)
    assert drawing.call_args.args[1] == [parent]  # Excluded child must not remove its parent.
    config["score_maximum"] = 0.9
    package.process(image, config)
    assert drawing.call_args.args[1] == [child]
    config.update(filter_scores=False, nesting="Outside-in", max_detections=10)
    package.process(image, config)
    assert set(drawing.call_args.args[1]) == {parent, child, low}
    assert engine.detect.call_args.args[1] == 0


@pytest.mark.parametrize("border", ["#00ff00", "dynamic"])
def test_labels_stay_visible_at_top_right_edge_and_on_same_color_image(
    package: CustomPackage, border: str
) -> None:
    """
    Avoid right-edge clipping and green-on-green text with stable outlined labels.
    """
    image = np.full((80, 80, 3), [0, 255, 0], np.uint8)
    boxes = [(0, 0.95, 0.3, 1, 0.87)]
    draw = helper(package, "drawing").draw_detections
    without = draw(image, boxes, 1, False, border)
    with_labels = draw(image, boxes, 1, True, border)
    assert np.any(with_labels != without)
    assert np.any(np.all(with_labels == 255, axis=2))
    assert np.any(np.all(with_labels == 0, axis=2))
    mask = helper(package, "labels").confidence_mask((80, 80), [("0.87", (79, 0))])
    rows, columns = np.nonzero(mask)
    assert columns.max() < 78 and columns.min() > 1
    assert rows.min() > 1 and rows.max() < 78


def test_labels_reserve_nonoverlapping_bounds_and_support_small_images(package: CustomPackage) -> None:
    """
    Crowded anchors retain separate labels and tiny inputs do not crash layout.
    """
    mask = helper(package, "labels").confidence_mask
    single = mask((100, 100), [("0.87", (50, 50))])
    multiple = mask((100, 100), [("0.87", (50, 50)), ("0.87", (50, 50))])
    assert np.count_nonzero(multiple) >= 2 * np.count_nonzero(single)
    assert mask((16, 16), [("0.87", (15, 0))]).any()
    assert mask((1, 1), [("0.87", (0, 0))]).shape == (1, 1)


@pytest.mark.parametrize("config", [
    {"score_threshold": 0.8, "score_maximum": 0.2}, {"score_maximum": float("nan")},
    {"filter_scores": 1}, {"nested_coverage": 0.4}, {"duplicate_iou": 0},
    {"suppress_duplicates": "true"},
])
def test_invalid_filter_settings_are_reported(package: CustomPackage, config: dict[str, Any]) -> None:
    """
    Refuse invalid confidence ranges, nesting coverage and suppression settings.
    """
    with pytest.raises(ValueError):
        helper(package, "settings").Settings.from_config(config)
