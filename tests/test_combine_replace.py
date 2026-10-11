"""
Regional replacement preserves occupied black pixels, padding and branch geometry.
"""

from typing import Any

import numpy as np
import pytest
from support.frames import make_frame
from support.pipeline import raw_pipeline

from topdon_duo.custom_nodes.labels import DisplayLabel
from topdon_duo.pipeline import node
from topdon_duo.pipeline_processing import PipelineProcessor, combine_images
from topdon_duo.processing.coverage import combine_coverage, replace_labels, transform_coverage
from topdon_duo.render import ThermalRenderer


def combine_params(**values: Any) -> dict[str, Any]:
    """
    Use the real Combine defaults, including its previously saved opacity value.
    """
    return node("software", "combine", mode="replace", **values)["params"]


def test_replace_copies_true_black_and_preserves_empty_padding() -> None:
    """
    A black corner replaces white base pixels independently of node opacity.
    """
    base = np.full((6, 8, 3), 200, np.float32)
    incoming = np.zeros_like(base)
    footprint = np.zeros(base.shape[:2], np.uint8)
    footprint[1:4, 2:6] = 1
    before = footprint.copy()
    for opacity in (0.0, 0.5, 1.0):
        output = combine_images(base, incoming, combine_params(opacity=opacity),
                                incoming_coverage=footprint)
        expected = base.copy()
        expected[1:4, 2:6] = 0
        np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(footprint, before)
    np.testing.assert_array_equal(base, 200)
    np.testing.assert_array_equal(incoming, 0)
    np.testing.assert_array_equal(combine_images(base, incoming, combine_params()), incoming)


def test_replace_resizes_footprints_and_honors_soft_or_inverted_binary_masks() -> None:
    """
    Masks restrict replacement without opening the empty part of a transformed branch.
    """
    base = np.full((4, 6, 3), 200, np.float32)
    incoming = np.zeros((2, 3, 3), np.float32)
    region = np.array(((1, 0, 0), (1, 0, 0)), np.uint8)
    soft_mask = np.full((4, 6), 127.5, np.float32)
    output = combine_images(base, incoming, combine_params(), soft_mask, region)
    expected = base.copy()
    expected[:, :2] = 100
    np.testing.assert_array_equal(output, expected)
    binary_mask = np.zeros((4, 6), np.float32)
    binary_mask[:2] = 255
    output = combine_images(base, incoming,
        combine_params(mask_kind="threshold", mask_invert=True), binary_mask, region)
    expected = base.copy()
    expected[2:, :2] = 0
    np.testing.assert_array_equal(output, expected)


def test_existing_opacity_mode_keeps_its_full_canvas_behavior() -> None:
    """
    Introduced footprint metadata must not change existing blend results.
    """
    base = np.full((4, 6, 3), 200, np.float32)
    incoming = np.zeros_like(base)
    region = np.zeros(base.shape[:2], np.uint8)
    region[:2, :3] = 1
    values = node("software", "combine", mode="opacity", opacity=1.0)["params"]
    np.testing.assert_array_equal(combine_images(base, incoming, values,
                                               incoming_coverage=region), incoming)


def test_skewed_inset_replaces_its_occupied_polygon_instead_of_bounding_box() -> None:
    """
    Copy real black pixels inside a sheared footprint and leave its uncovered corners visible.
    """
    shape = (9, 9)
    values = node("software", "transform", width=50, height=50, skew_x=50)["params"]
    region = transform_coverage(None, shape, values)
    expected_region = np.zeros(shape, np.uint8)
    expected_region[:2, :3] = 1
    expected_region[2:4, :4] = 1
    expected_region[4, :5] = 1
    np.testing.assert_array_equal(region, expected_region)
    base = np.full((*shape, 3), 200, np.float32)
    output = combine_images(base, np.zeros_like(base), combine_params(), incoming_coverage=region)
    expected = np.where(expected_region[..., None], 0, base)
    np.testing.assert_array_equal(output, expected)


def test_replace_label_anchors_follow_visible_copied_region() -> None:
    """
    Retain base text outside the inset and reject incoming text blocked by its mask.
    """
    region = np.zeros((9, 9), np.uint8)
    region[:5, :5] = 1
    mask = np.full((9, 9), 255, np.float32)
    mask[1, 1] = 0
    current = [DisplayLabel("Overwritten", 0.25, 0.25), DisplayLabel("Outside", 1, 1)]
    incoming = [DisplayLabel("Inset", 0.25, 0.25), DisplayLabel("Masked", 0.125, 0.125),
                DisplayLabel("Padding", 0.75, 0.75)]
    labels = replace_labels(current, incoming, region.shape, combine_params(), mask, region)
    assert labels == [current[1], incoming[0]]


@pytest.mark.parametrize("mode", ["opacity", "replace"])
def test_nested_footprints_preserve_empty_regions(mode: str) -> None:
    """
    Pass a sparse inset through another branch, including full-strength Opacity.
    """
    current = np.zeros((9, 9), np.uint8)
    incoming = current.copy()
    incoming[:5, :5] = 1
    values = node("software", "combine", mode=mode, opacity=1.0)["params"]
    occupied = combine_coverage(current, incoming, current.shape, values, None)
    np.testing.assert_array_equal(occupied, incoming)
    assert combine_coverage(None, incoming, current.shape, combine_params(), None) is None
    mirrored = transform_coverage(occupied, current.shape,
                                  node("software", "transform", horizontal=True)["params"])
    np.testing.assert_array_equal(mirrored, incoming[:, ::-1])


@pytest.mark.parametrize("scale", [1, 2])
def test_corner_raw_inset_replaces_processed_main_image_and_resets_each_frame(scale: int) -> None:
    """
    Exercise real worker dependency transfer, different resolutions and padding with black data.
    """
    frame = make_frame((10 + 50) * 64)
    document = raw_pipeline(node("software", "brightness", amount=100))
    if scale == 2:
        document["software"].insert(-1, node("software", "interpolation", scale=2, method="nearest"))
    inset = node("software", "transform", width=50, height=50, position_x=10, position_y=20,
                 method="nearest")
    document["branches"]["B"] = raw_pipeline(inset)["software"][:-1]
    combine = node("software", "combine", mode="replace", tab="B")
    document["software"].insert(-1, combine)
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, _ = processor.process(frame, None, document, scale=scale)
        expected = np.full((192 * scale, 256 * scale, 3), 255, np.uint8)
        expected[20 * scale:116 * scale, 10 * scale:138 * scale] = 0
        np.testing.assert_array_equal(output, expected)
        assert processor.branches["B"].output_coverage is not None
        combine["params"]["mode"] = "opacity"
        combine["params"]["opacity"] = 1.0
        output, _ = processor.process(frame, None, document, scale=scale)
        np.testing.assert_array_equal(output, 0)
        assert not processor.branches["B"].track_coverage
        assert processor.branches["B"].output_coverage is None
        combine["params"]["mode"] = "replace"
        output, _ = processor.process(frame, None, document, scale=scale)
        np.testing.assert_array_equal(output, expected)
        inset["bypass"] = True
        output, _ = processor.process(frame, None, document, scale=scale)
        np.testing.assert_array_equal(output, 0)
        assert processor.branches["B"].output_coverage is None
        combine["bypass"] = True
        output, _ = processor.process(frame, None, document, scale=scale)
        np.testing.assert_array_equal(output, 255)
        assert set(processor.branches) == {"A"}
    finally:
        processor.close()
    renderer = ThermalRenderer(scale=1)
    renderer.set_pipeline(raw_pipeline())
    baseline = renderer.render_detailed(frame, image_processing=False)
    combine["bypass"] = False
    inset["bypass"] = False
    renderer.set_pipeline(document)
    detailed = renderer.render_detailed(frame, image_processing=False)
    assert detailed.stats == baseline.stats
    np.testing.assert_array_equal(detailed.raw_counts, baseline.raw_counts)
    np.testing.assert_array_equal(detailed.temperatures_celsius, baseline.temperatures_celsius)


def test_footprint_survives_filters_and_nested_replace_then_transform() -> None:
    """
    Ignore colored padding and carry a replaced inset through a second branch's mirror.
    """
    frame = make_frame((10 + 50) * 64)
    document = raw_pipeline(node("software", "brightness", amount=100))
    document["branches"]["C"] = raw_pipeline(
        node("software", "transform", width=50, height=50, method="nearest"),
        node("software", "brightness", amount=100),
    )["software"][:-1]
    document["branches"]["B"] = raw_pipeline(
        node("software", "transform", position_x=8192),
        node("software", "combine", mode="replace", tab="C"),
        node("software", "transform", horizontal=True),
        node("software", "invert"),
    )["software"][:-1]
    document["software"].insert(-1, node("software", "combine", mode="replace", tab="B"))
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, _ = processor.process(frame, None, document, scale=1)
        expected = np.full_like(output, 255)
        expected[:96, 128:] = 0
        np.testing.assert_array_equal(output, expected)
    finally:
        processor.close()
