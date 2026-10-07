"""Feature selection should pick actual shapes and produce reusable display masks."""

import json
import subprocess
import sys
from copy import deepcopy

import cv2
import numpy as np
import pytest

from topdon_duo.feature_processing import apply_features
from topdon_duo.pipeline import default_pipeline, node, validate_pipeline


def scene():
    image = np.zeros((192, 256, 3), np.float32)
    cv2.rectangle(image, (20, 20), (65, 60), (160, 160, 160), -1)
    cv2.circle(image, (130, 50), 20, (230, 230, 230), -1)
    cv2.rectangle(image, (20, 100), (110, 120), (190, 190, 190), -1)
    polygon = np.array(
        [[160, 95], [215, 95], [215, 115], [180, 115], [180, 155], [160, 155]], np.int32
    )
    cv2.fillPoly(image, [polygon], (180, 180, 180))
    return image


def params(kind, **values):
    return node("software", kind, **values)["params"]


def contour_mask(image=None, **values):
    settings = {"blur": 0, "close_kernel": 0, "segmentation": "threshold", "output": "mask"}
    settings.update(values)
    return apply_features(
        scene() if image is None else image, "contours", params("contours", **settings)
    )[..., 0]


@pytest.mark.parametrize(
    "shape,inside,outside",
    [
        ("rectangle", [(40, 40), (50, 110)], [(130, 50), (170, 130)]),
        ("circle", [(130, 50)], [(40, 40), (50, 110), (170, 130)]),
        ("convex", [(40, 40), (50, 110), (130, 50)], [(170, 130)]),
    ],
)
def test_shape_selection_distinguishes_round_rectangular_and_concave_regions(
    shape, inside, outside
):
    mask = contour_mask(shape=shape)
    for x, y in inside:
        assert mask[y, x] == 255
    for x, y in outside:
        assert mask[y, x] == 0


@pytest.mark.parametrize(
    "rank,point",
    [
        ("largest", (170, 130)),
        ("smallest", (130, 50)),
        ("brightest", (130, 50)),
        ("darkest", (40, 40)),
        ("center", (130, 50)),
    ],
)
def test_ranking_selects_the_intended_single_region(rank, point):
    mask = contour_mask(rank=rank, max_count=1)
    x, y = point
    assert mask[y, x] == 255
    count, _ = cv2.connectedComponents(mask.astype(np.uint8))
    assert count == 2


def test_area_aspect_solidity_and_circularity_filters():
    assert contour_mask(min_area=3.7, max_area=4)[130, 170] == 255
    assert contour_mask(min_area=3.7, max_area=4)[50, 130] == 0
    narrow = contour_mask(min_aspect=3, max_aspect=5)
    assert narrow[110, 50] == 255 and narrow[40, 40] == 0
    solid = contour_mask(solidity=0.95)
    assert solid[40, 40] == 255 and solid[130, 170] == 0
    round_regions = contour_mask(circularity=0.8)
    assert round_regions[50, 130] == 255 and round_regions[110, 50] == 0


def test_border_exclusion_inversion_and_center_region():
    image = scene()
    cv2.rectangle(image, (0, 160), (40, 191), (240, 240, 240), -1)
    assert contour_mask(image)[170, 20] == 0
    assert contour_mask(image, exclude_border=False)[170, 20] == 255
    center = contour_mask(region="center", region_size=50)
    assert center[50, 130] == 255 and center[40, 40] == 0
    dark_scene = 255 - scene()
    dark = contour_mask(dark_scene, invert=True)
    np.testing.assert_array_equal(dark, contour_mask())


@pytest.mark.parametrize("segmentation", ["otsu", "threshold", "adaptive", "canny"])
@pytest.mark.parametrize("output", ["overlay", "fill", "mask", "cutout", "mean"])
def test_all_region_detection_and_output_modes(segmentation, output):
    original = scene()
    before = original.copy()
    result = apply_features(
        original,
        "contours",
        params("contours", segmentation=segmentation, output=output, max_count=10),
    )
    assert result.dtype == np.float32 and result.shape == original.shape
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255
    np.testing.assert_array_equal(original, before)
    if output == "mask":
        assert set(np.unique(result)) <= {0, 255}
        assert result.any()
    assert apply_features(original, "contours", params("contours", mix=0)) is original


def test_cutout_outline_fill_and_region_mean_are_distinct():
    original = scene()
    selected = {
        "blur": 0,
        "close_kernel": 0,
        "segmentation": "threshold",
        "shape": "rectangle",
        "max_count": 1,
        "rank": "smallest",
        "max_aspect": 2,
    }
    cutout = apply_features(original, "contours", params("contours", output="cutout", **selected))
    assert cutout[40, 40, 0] == 160 and not cutout[50, 130].any()
    outline = apply_features(
        original, "contours", params("contours", color="red", output="overlay", **selected)
    )
    np.testing.assert_array_equal(outline[40, 40], original[40, 40])
    np.testing.assert_array_equal(outline[20, 40], [0, 0, 255])
    fill = apply_features(
        original, "contours", params("contours", color="red", output="fill", **selected)
    )
    np.testing.assert_array_equal(fill[40, 40], [0, 0, 255])
    gradient = original.copy()
    gradient[20:61, 20:66] += np.arange(46, dtype=np.float32)[None, :, None]
    averaged = apply_features(gradient, "contours", params("contours", output="mean", **selected))
    np.testing.assert_array_equal(averaged[30, 30], averaged[40, 50])
    assert averaged[40, 40, 0] == pytest.approx(182.5)
    mixed = apply_features(
        original, "contours", params("contours", color="red", output="fill", mix=0.25, **selected)
    )
    np.testing.assert_allclose(mixed, original * 0.75 + fill * 0.25)


@pytest.mark.parametrize("geometry", ["contour", "polygon", "hull", "box"])
def test_region_geometry_and_resolution_scaling(geometry):
    native = scene()
    large = cv2.resize(native, (1024, 768), interpolation=cv2.INTER_NEAREST)
    mask = contour_mask(large, resolution=256, geometry=geometry, shape="circle")
    assert mask.shape == (768, 1024) and mask[200, 520] == 255
    assert not mask[160, 160] and set(np.unique(mask)) <= {0, 255}


@pytest.mark.parametrize("method", ["canny", "sobel", "scharr"])
@pytest.mark.parametrize("output", ["overlay", "mask", "cutout"])
def test_edge_methods_outputs_and_non_mutation(method, output):
    original = scene()
    before = original.copy()
    result = apply_features(
        original, "edges", params("edges", method=method, output=output, min_pixels=1)
    )
    assert result.shape == original.shape and result.dtype == np.float32
    np.testing.assert_array_equal(original, before)
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255
    if output == "mask":
        assert result.any() and set(np.unique(result)) <= {0, 255}
    assert apply_features(original, "edges", params("edges", mix=0)) is original


def test_horizontal_vertical_edges_and_strength_threshold():
    original = scene()
    base = {"method": "sobel", "blur": 0, "min_pixels": 1, "output": "mask"}
    horizontal = apply_features(original, "edges", params("edges", direction="horizontal", **base))
    vertical = apply_features(original, "edges", params("edges", direction="vertical", **base))
    assert horizontal[20, 40, 0] and not horizontal[40, 20, 0]
    assert vertical[40, 20, 0] and not vertical[20, 40, 0]
    weak = apply_features(original, "edges", params("edges", strength=10, **base))
    strong = apply_features(original, "edges", params("edges", strength=250, **base))
    assert np.count_nonzero(weak) > np.count_nonzero(strong)


def test_connected_edge_length_and_rank_selection():
    image = np.zeros((192, 256, 3), np.float32)
    cv2.rectangle(image, (20, 20), (90, 75), (200, 200, 200), -1)
    cv2.rectangle(image, (120, 90), (135, 105), (200, 200, 200), -1)
    base = {"blur": 0, "output": "mask", "min_pixels": 1, "max_count": 1}
    longest = apply_features(image, "edges", params("edges", rank="longest", **base))[..., 0]
    central = apply_features(image, "edges", params("edges", rank="center", **base))[..., 0]
    assert longest[:80, :100].any() and not longest[85:110, 115:140].any()
    assert central[85:110, 115:140].any() and not central[:80, :100].any()
    rejected = apply_features(image, "edges", params("edges", min_pixels=10000, output="mask"))
    assert not rejected.any()


@pytest.mark.parametrize("kind", ["edges", "contours"])
def test_empty_image_and_pixel_dimensions(kind):
    for shape in [(1, 1, 3), (1, 64, 3), (64, 1, 3), (192, 256, 3)]:
        image = np.zeros(shape, np.float32)
        result = apply_features(image, kind, params(kind, output="mask"))
        assert result.shape == shape and not result.any()


@pytest.mark.parametrize(
    "kind,values",
    [
        ("edges", {"auto_threshold": False, "lower": 200, "upper": 100}),
        ("contours", {"segmentation": "canny", "lower": 200, "upper": 100}),
        ("contours", {"min_area": 50, "max_area": 20}),
        ("contours", {"min_aspect": 5, "max_aspect": 2}),
        ("edges", {"resolution": 4096}),
        ("contours", {"max_count": 1000}),
        ("contours", {"solidity": float("nan")}),
    ],
)
def test_invalid_configuration_rejected_atomically(kind, values):
    saved = default_pipeline()
    saved["software"].insert(1, node("software", kind, **values))
    before = deepcopy(saved)
    with pytest.raises(ValueError):
        validate_pipeline(saved)
    assert saved == before


def test_feature_nodes_repeat_export_and_persist(tmp_path, monkeypatch):
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    saved = default_pipeline()
    saved["software"].insert(1, node("software", "edges", method="scharr", direction="vertical"))
    saved["software"].insert(2, node("software", "edges", output="mask"))
    saved["branches"]["B"].append(
        node("software", "contours", shape="rectangle", max_count=1, output="mask")
    )
    assert validate_pipeline(json.loads(json.dumps(saved))) == saved
    save_settings({"pipeline": saved})
    assert load_settings()["pipeline"] == saved


def test_real_pipeline_feature_mask_can_drive_combine_and_preserves_radiometry():
    from test_render import frame_with_preview

    from topdon_duo.camera import decode_duo_frame
    from topdon_duo.pipeline_processing import PipelineProcessor

    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    snapshot = raw.copy()
    saved = default_pipeline()
    saved["software"] = [
        saved["software"][0],
        node("software", "brightness", amount=50),
        node("software", "combine", tab="preview", mask_source="B", opacity=1),
        saved["software"][-1],
    ]
    saved["branches"]["B"].append(
        node(
            "software",
            "contours",
            segmentation="threshold",
            threshold=0,
            exclude_border=False,
            max_area=100,
            output="mask",
            blur=0,
            close_kernel=0,
        )
    )
    processor = PipelineProcessor()
    try:
        result = processor.process(frame, raw.astype(np.float32), saved, scale=1)[0]
        assert result.shape == (192, 256, 3)
        np.testing.assert_array_equal(raw, snapshot)
        assert set(processor.branches) == {"A", "B"}
        saved["software"][-2]["bypass"] = True
        unmasked = processor.process(frame, raw.astype(np.float32), saved, scale=1)[0]
        assert not np.array_equal(result, unmasked)
    finally:
        processor.close()


def test_editor_hides_irrelevant_feature_fields_and_locks_inputs():
    from test_capture_panel import popup_environment

    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import default_pipeline, node
from topdon_duo.feature_processing import feature_fields
app = QApplication([])
window = ViewWindow(lambda message: None)
window.show()
editor = window.pipeline_editor
saved = default_pipeline()
saved["software"] = [saved["software"][0], saved["software"][-1]]
for kind in ("edges", "contours"):
    item = node("software", kind)
    item["expanded"] = True
    saved["software"].insert(-1, item)
editor.update_state({"pipeline": saved, "pipeline_serial": 1}, False)
for index, cases in [(1, [("method", "sobel"), ("method", "canny"), ("auto_threshold", False), ("output", "mask"), ("region", "center")]), (2, [("segmentation", "adaptive"), ("segmentation", "threshold"), ("segmentation", "canny"), ("output", "mean")])]:
    for key, value in cases:
        item = editor.document["software"][index]
        editor.change(item, key, value)
        editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, False)
        item = editor.document["software"][index]
        for name, row in editor.widgets[item["id"]][1].items():
            assert row.isHidden() == (name not in feature_fields(item["type"], item["params"]))
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, True)
for index in (1,2):
    for row in editor.widgets[editor.document["software"][index]["id"]][1].values():
        assert not row.input.isEnabled()
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_auto_canny_detects_a_straight_step_without_corner_seeds():
    image = np.zeros((128, 128, 3), np.float32)
    image[:, 64:] = 200
    mask = apply_features(
        image, "edges", params("edges", output="mask", blur=0, exclude_border=False)
    )
    assert mask[:, 63:65].any() and not mask[:, :60].any()


def test_closing_gaps_recovers_an_open_boundary():
    image = np.zeros((100, 100, 3), np.float32)
    cv2.rectangle(image, (20, 20), (80, 80), (220, 220, 220), 1)
    image[20, 49:52] = 0
    open_mask = contour_mask(image, close_kernel=0)
    closed_mask = contour_mask(image, close_kernel=5)
    assert open_mask[50, 50] == 0 and closed_mask[50, 50] == 255


def test_rotated_rectangles_are_selected_without_axis_aligned_aspect_bias():
    image = np.zeros((192, 256, 3), np.float32)
    points = cv2.boxPoints(((128, 96), (80, 20), 35)).round().astype(np.int32)
    cv2.fillPoly(image, [points], (230, 230, 230))
    mask = contour_mask(image, shape="rectangle", min_aspect=3, max_aspect=5)
    assert mask[96, 128] == 255
