"""
Keep custom confidence glyphs in display space through pipeline geometry changes.
"""

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from support.classless import helper
from support.classless import package as package  # noqa: PLC0414 - pytest fixture re-export
from support.pipeline import raw_pipeline
from test_render import frame_with_preview

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.labels import DisplayLabel, collect_labels, defer_labels
from topdon_duo.custom_nodes.runtime import CustomPackage
from topdon_duo.pipeline import node
from topdon_duo.pipeline_processing import PipelineProcessor
from topdon_duo.processing.images import encode_thumbnail


def label_node(tmp_path: Path) -> dict[str, Any]:
    """
    Build a portable annotation package with a deliberately asymmetric confidence glyph.
    """
    (tmp_path / "__init__.py").write_text(
        'import numpy as np\n'
        'from topdon_duo.custom_nodes.labels import defer_labels\n'
        'def process(image, config):\n'
        '    height, width = image.shape[:2]\n'
        '    defer_labels([("0.87", (width // 4, height // 3))], (height, width))\n'
        '    return np.zeros_like(image)\n', encoding="utf-8",
    )
    item = node("software", "custom")
    item["params"].update(load_folder(tmp_path))
    return item


def glyph(image: np.ndarray) -> np.ndarray:
    """
    Crop white text tightly, excluding its black outline and surrounding background.
    """
    mask = image[..., 0]
    ys, xs = np.nonzero(mask)
    assert len(xs)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def measurement_glyph() -> np.ndarray:
    """
    Draw an independent reference using the measurement font's established settings.
    """
    expected = np.zeros((40, 100, 3), np.uint8)
    cv2.putText(expected, "0.87", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return glyph(expected)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("scale", [1, 3])
@pytest.mark.parametrize("horizontal,vertical", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_viewer_labels_match_measurement_font_after_rotation_resize_and_mirror(
    tmp_path: Path, rotation: int, scale: int, horizontal: bool, vertical: bool,
) -> None:
    """
    Compare the actual scheduler output to upright OpenCV measurement-size text.
    """
    document = raw_pipeline()
    custom = label_node(tmp_path)
    document["software"][-1:-1] = [
        custom, node("software", "mirror", horizontal=horizontal, vertical=vertical),
        node("software", "interpolation", scale=2),
    ]
    frame, _ = frame_with_preview()
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        image, _ = processor.process(frame, None, document, scale=scale, rotation=rotation)
        np.testing.assert_array_equal(glyph(image), measurement_glyph())
        anchor_x, anchor_y = 64 / 255, 64 / 191
        if horizontal:
            anchor_x = 1 - anchor_x
        if vertical:
            anchor_y = 1 - anchor_y
        for _ in range(rotation // 90):
            anchor_x, anchor_y = 1 - anchor_y, anchor_x
        ys, xs = np.nonzero(image[..., 0])
        assert abs(xs.mean() - anchor_x * (image.shape[1] - 1)) < 55
        assert abs(ys.mean() - anchor_y * (image.shape[0] - 1)) < 40
        # Bypassing the node on the next frame must not retain stale annotations.
        custom["bypass"] = True
        image, _ = processor.process(frame, None, document, scale=scale, rotation=rotation)
        assert processor.branches["A"].custom.labels == []
    finally:
        processor.close()


def test_classless_labels_are_deferred_without_changing_detection_boxes(package: CustomPackage) -> None:
    """
    The actual example emits text annotations separately while retaining box pixels.
    """
    image = np.zeros((80, 100, 3), np.uint8)
    boxes = [(0.3, 0.2, 0.6, 0.7, 0.87)]
    draw = helper(package, "drawing").draw_detections
    without = draw(image, boxes, 1, False)
    labels: list[DisplayLabel] = []
    with collect_labels(labels):
        output = draw(image, boxes, 1, True)
    np.testing.assert_array_equal(output, without)
    assert labels == [DisplayLabel("0.87", 20 / 99, 24 / 79)]
    assert not defer_labels([], (80, 100))


def test_branch_labels_follow_combined_input_and_reset_each_frame(tmp_path: Path) -> None:
    """
    Carry annotations through a connected branch without duplicating them across frames.
    """
    document = raw_pipeline()
    document["branches"]["B"] = [node("software", "source", source="raw"), label_node(tmp_path)]
    document["software"].insert(-1, node("software", "combine", tab="B", mode="opacity"))
    frame, _ = frame_with_preview()
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        first, _ = processor.process(frame, None, document, scale=2, rotation=90)
        second, _ = processor.process(frame, None, document, scale=2, rotation=90)
        assert len(processor.branches["A"].custom.labels) == 1
        np.testing.assert_array_equal(first, second)
        assert np.any(np.all(first == 255, axis=2))
    finally:
        processor.close()


def test_label_collector_restores_context_after_callback_failure() -> None:
    """
    Failed custom code cannot leak its collector into other node executions.
    """
    with pytest.raises(ValueError, match="anchor"), collect_labels([]):
        defer_labels([("0.87", (100, 0))], (80, 100))
    assert not defer_labels([], (80, 100))


def test_thumbnail_keeps_labels_at_measurement_font_size_after_downsampling() -> None:
    """
    Preview thumbnails retain readable labels even after large pipeline upscaling.
    """
    payload = encode_thumbnail(np.zeros((768, 1024, 3), np.uint8),
                               [DisplayLabel("0.87", 0.3, 0.4)])
    thumbnail = cv2.imdecode(np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_COLOR)
    assert isinstance(thumbnail, np.ndarray)
    assert thumbnail.shape == (240, 320, 3)
    np.testing.assert_array_equal(glyph(thumbnail), measurement_glyph())
