"""
Transform geometry, legacy migration and isolation of native thermal measurements.
"""

from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import pytest
from support.frames import frame_with_preview
from support.pipeline import raw_pipeline

from topdon_duo.custom_nodes.labels import DisplayLabel
from topdon_duo.image_transform import apply_transform
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_model.sources import spatial_mapping_supported
from topdon_duo.processing.branch import BranchProcessor
from topdon_duo.render import ThermalRenderer


def params(**values: Any) -> dict[str, Any]:
    """
    Supply complete validated Transform defaults for an isolated pixel operation.
    """
    return node("software", "transform", **values)["params"]


def test_legacy_mirror_import_preserves_identity_state_order_and_inputs() -> None:
    """
    Migrate main and auxiliary nodes without mutating the saved document.
    """
    legacy = {"id": "old-mirror", "type": "mirror", "params": {
        "horizontal": True, "vertical": False,
    }, "bypass": True, "expanded": True}
    document = raw_pipeline()
    document["software"].insert(2, legacy)
    document["branches"]["B"] = [node("software", "source"),
                                   {**deepcopy(legacy), "id": "branch-mirror"}]
    before = deepcopy(document)
    migrated = validate_pipeline(document)
    assert document == before
    for item in (migrated["software"][2], migrated["branches"]["B"][1]):
        assert item["type"] == "transform"
        assert item["bypass"] and item["expanded"]
        assert item["params"] == params(horizontal=True)
    assert [item["id"] for item in migrated["software"]] == [
        item["id"] for item in document["software"]
    ]
    assert validate_pipeline(migrated) == migrated


@pytest.mark.parametrize("legacy_params", [
    {"horizontal": True}, {"horizontal": True, "vertical": False, "extra": 0},
    {"horizontal": 1, "vertical": False}, None,
])
def test_malformed_legacy_mirror_is_rejected(legacy_params: Any) -> None:
    """
    Keep strict saved-data validation through the migration path.
    """
    document = raw_pipeline()
    document["software"].insert(2, {"id": "legacy", "type": "mirror",
        "params": legacy_params, "bypass": False, "expanded": False})
    with pytest.raises(ValueError):
        validate_pipeline(document)


def test_transform_without_position_controls_migrates_without_changing_other_settings() -> None:
    """
    Preserve earlier Transform documents and add a neutral pixel position.
    """
    document = raw_pipeline(node("software", "transform", width=75, horizontal=True, skew_x=10))
    old = document["software"][2]
    for key in ("position_unit", "position_x", "position_y"):
        del old["params"][key]
    before = deepcopy(document)
    migrated = validate_pipeline(document)
    assert document == before
    item = migrated["software"][2]
    assert {key: value for key, value in item.items() if key != "params"} == {
        key: value for key, value in old.items() if key != "params"
    }
    assert item["params"] == {**old["params"], "position_unit": "pixels",
                              "position_x": 0, "position_y": 0}


@pytest.mark.parametrize("values", [
    {"position_unit": "unknown"}, {"position_x": float("nan")},
    {"position_y": float("inf")}, {"position_x": True}, {"position_y": 8193},
])
def test_position_parameters_are_validated(values: dict[str, Any]) -> None:
    """
    Reject invalid units, non-finite numbers, booleans and out-of-range offsets.
    """
    with pytest.raises(ValueError):
        validate_pipeline(raw_pipeline(node("software", "transform", **values)))


@pytest.mark.parametrize("size", [(20, 10), (40, 20)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_pixel_and_percent_positions_match_pixels_and_label_anchors(
    size: tuple[int, int], sign: int,
) -> None:
    """
    Percent uses full canvas dimensions and signed offsets match exact pixel shifts.
    """
    width, height = size
    image = np.arange(width * height * 3, dtype=np.float32).reshape(height, width, 3)
    dx, dy = sign * width // 5, sign * height // 5
    labels = [DisplayLabel("Center", 0.5, 0.5)]
    pixels, pixel_labels = apply_transform(image, params(position_x=dx, position_y=dy,
                                                       method="nearest"), labels)
    percent, percent_labels = apply_transform(image, params(position_unit="percent",
        position_x=sign * 20, position_y=sign * 20, method="nearest"), labels)
    expected = np.zeros_like(image)
    if sign > 0:
        expected[dy:, dx:] = image[:-dy, :-dx]
    else:
        expected[:dy, :dx] = image[-dy:, -dx:]
    np.testing.assert_array_equal(pixels, expected)
    np.testing.assert_array_equal(percent, expected)
    assert percent_labels == pixel_labels
    assert (pixel_labels[0].x, pixel_labels[0].y) == pytest.approx(
        (0.5 + dx / (width - 1), 0.5 + dy / (height - 1))
    )


def test_position_uses_resized_canvas_and_runs_after_perspective() -> None:
    """
    Translate the completed distortion in the output canvas's pixel coordinates.
    """
    image = np.zeros((9, 9, 3), np.float32)
    image[0, 0] = 255
    output, labels = apply_transform(image, params(top_left_x=25, position_x=1, position_y=2,
                                                  method="nearest"), [DisplayLabel("T", 0, 0)])
    np.testing.assert_array_equal(output[2, 3], (255, 255, 255))
    assert (labels[0].x, labels[0].y) == pytest.approx((3 / 8, 2 / 8))
    resized, resized_labels = apply_transform(image, params(resize_mode="resolution", width=200,
        height=200, position_unit="percent", position_x=50, method="nearest"),
        [DisplayLabel("T", 0, 0)])
    assert resized.shape == (18, 18, 3)
    np.testing.assert_array_equal(resized[0, 9], (255, 255, 255))
    assert resized_labels[0].x == pytest.approx(9 / 17)


def test_fractional_position_and_clipped_anchors() -> None:
    """
    Interpolate subpixel offsets and discard labels translated beyond the canvas.
    """
    image = np.zeros((9, 9, 3), np.float32)
    image[4, 4] = 255
    output, labels = apply_transform(image, params(position_x=0.5, position_y=-0.5),
                                    [DisplayLabel("T", 0.5, 0.5)])
    np.testing.assert_array_equal(output[3:5, 4:6], np.full((2, 2, 3), 63.75))
    assert (labels[0].x, labels[0].y) == pytest.approx((0.5625, 0.4375))
    _, labels = apply_transform(image, params(position_x=10), [DisplayLabel("T", 0.5, 0.5)])
    assert labels == []


@pytest.mark.parametrize("unit", ["pixels", "percent"])
@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, True)])
def test_smaller_canvas_image_at_zero_position_is_anchored_upper_left(
    unit: str, horizontal: bool, vertical: bool,
) -> None:
    """
    Zero position aligns the shrunk image with the canvas origin in either unit mode.
    """
    image = np.full((9, 9, 3), 255, np.float32)
    anchor = DisplayLabel("Corner", 1 if horizontal else 0, 1 if vertical else 0)
    output, labels = apply_transform(image, params(width=50, height=25, position_unit=unit,
        horizontal=horizontal, vertical=vertical, method="nearest"), [anchor])
    expected = np.zeros_like(image)
    expected[:3, :5] = 255
    np.testing.assert_array_equal(output, expected)
    assert labels == [DisplayLabel("Corner", 0, 0)]


@pytest.mark.parametrize("unit,x,y", [("pixels", 2, 1), ("percent", 10, 10)])
def test_smaller_canvas_image_position_is_measured_from_upper_left(
    unit: str, x: int, y: int,
) -> None:
    """
    Offset a smaller image from the canvas origin, moving its pixel and label together.
    """
    image = np.zeros((10, 20, 3), np.float32)
    image[0, 0] = 255
    output, labels = apply_transform(image, params(width=50, height=50, position_unit=unit,
        position_x=x, position_y=y, method="nearest"), [DisplayLabel("Corner", 0, 0)])
    expected = np.zeros_like(image)
    expected[1, 2] = 255
    np.testing.assert_array_equal(output, expected)
    assert len(labels) == 1
    assert (labels[0].x, labels[0].y) == pytest.approx((2 / 19, 1 / 9))


@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, False),
                                                 (False, True), (True, True)])
def test_identity_and_mirrors_preserve_exact_pixels(horizontal: bool, vertical: bool) -> None:
    """
    Avoid resampling for the old Mirror operation, including single-pixel inputs.
    """
    for shape in ((5, 7, 3), (1, 1, 3)):
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        before = image.copy()
        output, labels = apply_transform(image, params(horizontal=horizontal, vertical=vertical),
                                         [DisplayLabel("T", 0.25, 0.75)])
        expected = image[::-1 if vertical else 1, ::-1 if horizontal else 1]
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(image, before)
        assert labels == [DisplayLabel("T", 0.75 if horizontal else 0.25,
                                      0.25 if vertical else 0.75)]


def test_resolution_resize_and_branch_execution() -> None:
    """
    Change the working dimensions before later nodes without enlarging native arrays.
    """
    image = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    output, labels = apply_transform(image, params(resize_mode="resolution", width=200,
                                                 height=50, method="nearest"), [])
    np.testing.assert_array_equal(output, image[::2].repeat(2, axis=1))
    assert labels == []
    processor = BranchProcessor(apple_available=False, nvidia_available=False)
    frame, _ = frame_with_preview()
    try:
        output, _ = processor.process(frame, None, raw_pipeline(node("software", "transform",
            resize_mode="resolution", width=200, height=50)))
        assert output.shape == (96, 512, 3)
    finally:
        processor.close()


@pytest.mark.parametrize("settings,source,target", [
    ({"width": 50, "height": 50}, (0, 4), (0, 2)),
    ({"skew_x": 50}, (4, 0), (2, 0)),
    ({"top_left_x": 25}, (0, 0), (2, 0)),
])
def test_canvas_scale_skew_and_perspective_move_pixels_and_anchors(
    settings: dict[str, int], source: tuple[int, int], target: tuple[int, int],
) -> None:
    """
    Check independently computed landmark positions and upright deferred labels.
    """
    image = np.zeros((9, 9, 3), np.float32)
    image[source[1], source[0]] = 255
    output, labels = apply_transform(image, params(method="nearest", **settings),
        [DisplayLabel("Landmark", source[0] / 8, source[1] / 8)])
    assert output.shape == image.shape
    np.testing.assert_array_equal(output[target[1], target[0]], (255, 255, 255))
    assert len(labels) == 1
    assert (labels[0].x, labels[0].y) == pytest.approx((target[0] / 8, target[1] / 8))
    assert labels[0].text == "Landmark"


def test_clipped_anchors_are_removed_and_empty_canvas_is_black() -> None:
    """
    Prevent invisible anchors from placing text over unrelated display pixels.
    """
    output, labels = apply_transform(np.full((9, 9, 3), 255, np.float32),
        params(width=200, skew_x=50, method="nearest"), [DisplayLabel("Outside", 0, 0)])
    assert labels == []
    assert np.any(output)
    output, _ = apply_transform(np.full((9, 9, 3), 255, np.float32), params(width=50), [])
    np.testing.assert_array_equal(output[:, -1], 0)


def test_canvas_zoom_limits_actual_output_allocation() -> None:
    """
    Keep a large zoom in the existing canvas rather than allocating an enlarged image.
    """
    image = np.full((600, 600, 3), 255, np.uint8)
    output, _ = apply_transform(image, params(width=400, height=400), [])
    assert output.shape == image.shape
    np.testing.assert_array_equal(output, image)


def test_invalid_geometry_and_size_limit_precede_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Reject concave corners and oversized resolutions before invoking OpenCV allocation.
    """
    document = raw_pipeline(node("software", "transform", top_left_x=45, top_left_y=45,
        top_right_x=45, top_right_y=45, bottom_right_x=-45, bottom_right_y=-45,
        bottom_left_x=45, bottom_left_y=45))
    with pytest.raises(ValueError, match="quadrilateral"):
        validate_pipeline(document)

    def unexpected_resize(*_args: Any, **_kwargs: Any) -> np.ndarray:
        """
        Fail if the size guard allows an oversized allocation.
        """
        raise AssertionError("Unexpected allocation")

    monkeypatch.setattr(cv2, "resize", unexpected_resize)
    with pytest.raises(ValueError, match="4 megapixel"):
        apply_transform(np.zeros((600, 600, 3), np.uint8),
                        params(resize_mode="resolution", width=400, height=400), [])


@pytest.mark.parametrize("settings", [{"skew_x": 30}, {"position_x": 2},
                                      {"position_unit": "percent", "position_y": -10}])
def test_warps_preserve_radiometry_and_guard_spatial_picking(settings: dict[str, Any]) -> None:
    """
    Keep sensor statistics unchanged and restore picking when a warp is bypassed.
    """
    frame, _ = frame_with_preview()
    renderer = ThermalRenderer(scale=1)
    renderer.set_pipeline(raw_pipeline())
    baseline = renderer.render_detailed(frame, image_processing=False)
    document = raw_pipeline(node("software", "transform", **settings))
    renderer.set_pipeline(document)
    warped = renderer.render_detailed(frame, image_processing=False)
    assert not warped.measurement_mapping_valid
    assert warped.stats == baseline.stats
    np.testing.assert_array_equal(warped.raw_counts, baseline.raw_counts)
    np.testing.assert_array_equal(warped.temperatures_celsius, baseline.temperatures_celsius)
    document["software"][2]["bypass"] = True
    renderer.set_pipeline(document)
    assert renderer.render_detailed(frame, image_processing=False).measurement_mapping_valid
    document["software"][2] = node("software", "transform", resize_mode="resolution", width=50)
    assert spatial_mapping_supported(document)
    document["branches"]["B"] = [node("software", "source"), node("software", "transform", width=50)]
    assert spatial_mapping_supported(document)
    document["software"].insert(-1, node("software", "combine", tab="B"))
    assert not spatial_mapping_supported(document)
