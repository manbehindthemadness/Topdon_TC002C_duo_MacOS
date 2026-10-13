"""
Protect native image detail when a small transformed branch reaches a larger canvas.
"""

import cv2
import numpy as np
import pytest
from support.frames import make_frame
from support.pipeline import raw_pipeline

from based_thermals.camera import HEADER_U16, SENSOR_PIXELS
from based_thermals.pipeline import node
from based_thermals.pipeline_processing import PipelineProcessor
from based_thermals.processing.branch import BranchProcessor


def striped_frame() -> bytes:
    """
    Alternate cold and hot native sensor columns, independently of display operations.
    """
    words = np.frombuffer(make_frame(), dtype="<u2").copy()
    counts = np.tile(np.where(np.arange(256) % 2, 110 * 64, 60 * 64), (192, 1))
    words[HEADER_U16:HEADER_U16 + SENSOR_PIXELS] = counts.ravel()
    return words.tobytes()


@pytest.mark.parametrize("main_scale", [1, 2])
@pytest.mark.parametrize("rotation", [0, 90])
def test_raw_inset_keeps_native_stripes_at_display_resolution(main_scale: int, rotation: int) -> None:
    """
    A half-sized inset on a 2x output still has room for every native sensor column.
    """
    document = raw_pipeline(node("software", "brightness", amount=100))
    if main_scale == 2:
        document["software"].insert(-1, node("software", "interpolation", scale=2))
    document["branches"]["B"] = raw_pipeline(
        node("software", "transform", width=50, height=50, method="nearest"),
    )["software"][:-1]
    document["software"].insert(-1, node("software", "combine", tab="B", mode="replace"))
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, _ = processor.process(striped_frame(), None, document, scale=2, rotation=rotation)
        upright = np.rot90(output, rotation // 90)
        # Each alternating column survives. Premature 50% sampling would erase all hot columns.
        np.testing.assert_array_equal(upright[20, :250, 0], np.arange(250) % 2 * 255)
        np.testing.assert_array_equal(upright[250:, 300:], 255)
    finally:
        processor.close()


def test_terminal_transform_uses_selected_method_at_final_viewer_size() -> None:
    """
    Final viewer enlargement must not replace the selected interpolation with nearest pixels.
    """
    transform = node("software", "transform", width=35, height=35, method="nearest")
    document = raw_pipeline(transform)
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        nearest, _ = processor.process(striped_frame(), None, document, scale=3)
        transform["params"]["method"] = "linear"
        linear, _ = processor.process(striped_frame(), None, document, scale=3)
        assert np.unique(nearest[20, :250, 0]).tolist() == [0, 255]
        assert np.unique(linear[20, :250, 0]).size > 10
        assert not np.array_equal(nearest, linear)
    finally:
        processor.close()


def test_separate_transformed_mask_samples_at_same_receiving_resolution() -> None:
    """
    A striped mask branch retains native detail when restricting a solid black inset.
    """
    document = raw_pipeline(node("software", "brightness", amount=100),
                            node("software", "interpolation", scale=2))
    document["branches"]["B"] = raw_pipeline(
        node("software", "transform", width=50, height=50, method="nearest"),
    )["software"][:-1]
    document["branches"]["B"][1]["params"].update(low=60, high=100)
    document["branches"]["C"] = raw_pipeline(
        node("software", "transform", width=50, height=50, method="nearest"),
    )["software"][:-1]
    document["software"].insert(-1, node("software", "combine", tab="B", mode="replace",
                                        mask_source="C", mask_kind="threshold"))
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, _ = processor.process(striped_frame(), None, document, scale=2)
        np.testing.assert_array_equal(output[20, :250, 0], (1 - np.arange(250) % 2) * 255)
    finally:
        processor.close()


def test_later_pixel_operations_materialize_detail_and_bypass_resets_it() -> None:
    """
    Following filters receive preserved detail and stale geometry cannot leak frames.
    """
    transform = node("software", "transform", width=50, height=50)
    document = raw_pipeline(transform)
    processor = BranchProcessor(apple_available=False, nvidia_available=False)
    try:
        image, _ = processor.process(striped_frame(), None, document)
        assert image.shape == (192, 256, 3)
        assert processor.output_transform is not None
        document["software"].insert(-1, node("software", "brightness", amount=10))
        image, _ = processor.process(striped_frame(), None, document)
        assert processor.output_transform is None
        assert image.shape == (384, 512, 3)
        assert processor.output_logical_size == (256, 192)
        document["software"].pop(-2)
        transform["bypass"] = True
        processor.process(striped_frame(), None, document)
        assert processor.output_transform is None
    finally:
        processor.close()


@pytest.mark.parametrize("kind,values,inverted", [
    ("brightness", {"amount": 0}, False),
    ("gamma", {"amount": 2}, False),
    ("invert", {}, True),
    ("colors", {"palette": "black_hot"}, True),
    ("filter", {"filter": "threshold", "threshold": 127}, False),
    ("interpolation", {"scale": 2, "method": "nearest"}, False),
])
@pytest.mark.parametrize("after_combine", [False, True])
def test_downstream_processing_keeps_native_inset_detail(
    kind: str, values: dict[str, object], inverted: bool, after_combine: bool,
) -> None:
    """
    Pixel operations after either Transform or the receiving Combine cannot erase native stripes.
    """
    operation = node("software", kind, **values)
    transform = node("software", "transform", width=50, height=50, method="nearest")
    document = raw_pipeline(node("software", "brightness", amount=100))
    document["branches"]["B"] = raw_pipeline(transform,
        *([] if after_combine else [operation]))["software"][:-1]
    document["software"].insert(-1, node("software", "combine", tab="B", mode="replace"))
    if after_combine:
        document["software"].insert(-1, operation)
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, _ = processor.process(striped_frame(), None, document, scale=2)
        expected = np.arange(250) % 2 * 255
        if inverted:
            expected = 255 - expected
        np.testing.assert_array_equal(output[20, :250, 0], expected)
    finally:
        processor.close()


def test_pixel_positions_and_mirrors_scale_with_original_canvas_units() -> None:
    """
    Direct sampling keeps native pixel offsets and upper-left positioning after enlargement.
    """
    document = raw_pipeline(node("software", "transform", width=50, height=50,
                                position_x=10, position_y=20, horizontal=True, method="nearest"))
    processor = BranchProcessor(apple_available=False, nvidia_available=False)
    try:
        processor.process(striped_frame(), None, document)
        retained = processor.output_transform
        assert retained is not None
        image, coverage = retained.render((512, 384))
        np.testing.assert_array_equal(image[50, 21:250, 0], np.arange(229) % 2 * 255)
        assert coverage is not None
        assert coverage[40, 20] == 1
        assert coverage[39, 20] == 0
        assert coverage[40, 19] == 0
        assert image.dtype == np.float32
        # An independent constant mask verifies occupied black data survives the resampling.
        assert cv2.countNonZero(coverage) > 48_000
    finally:
        processor.close()
