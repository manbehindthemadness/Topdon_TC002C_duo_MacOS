"""Classic filters, backwards-compatible migration and editor controls."""

import json
import subprocess
import sys
from copy import deepcopy

import cv2
import numpy as np
import pytest

from topdon_duo.image_filters import FILTER_MODES, apply_filter, filter_fields
from topdon_duo.pipeline import default_pipeline, migrate_pipeline, node, validate_pipeline


def image():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 255, (48, 64, 3)).astype(np.float32)


def params(mode, **settings):
    return node("software", "filter", filter=mode, **settings)["params"]


def document(item):
    result = default_pipeline()
    result["software"].insert(1, item)
    return result


@pytest.mark.parametrize("mode", FILTER_MODES)
def test_filters_preserve_input_shape_and_finite_display_range(mode):
    original = image()
    before = original.copy()
    result = apply_filter(original, params(mode))
    assert result.shape == original.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 255
    np.testing.assert_array_equal(original, before)
    if mode != "none":
        assert not np.allclose(original, result)
    assert apply_filter(original, params(mode, mix=0)) is original


@pytest.mark.parametrize("mode", FILTER_MODES)
def test_blend_amount_interpolates_original_and_filtered(mode):
    original = image()
    full = apply_filter(original, params(mode))
    half = apply_filter(original, params(mode, mix=0.5))
    np.testing.assert_allclose(half, (original + full) / 2, atol=3e-5)


@pytest.mark.parametrize("mode", ["none", "bilateral", "median", "gaussian", "sharpen"])
@pytest.mark.parametrize("amount", [0, 0.7, 2.5])
def test_version_three_filter_migration_preserves_original_pixels(mode, amount):
    original = image()
    old = node("software", "filter")
    old["params"] = {"filter": mode, "amount": amount}
    saved = document(old)
    saved["version"] = 3
    before = deepcopy(saved)
    migrated = validate_pipeline(saved)
    assert saved == before and migrated["version"] == 4
    result = apply_filter(original, migrated["software"][1]["params"])
    expected = original
    if mode == "bilateral":
        expected = cv2.bilateralFilter(original, 5, 30 * amount, 3)
    elif mode == "median":
        expected = cv2.medianBlur(original.round().astype(np.uint8), 3).astype(np.float32)
    elif mode in ("gaussian", "sharpen") and amount:
        blurred = cv2.GaussianBlur(original, (0, 0), 0.8 if mode == "sharpen" else max(0.1, amount))
        expected = (
            np.clip(original * (1 + amount) - blurred * amount, 0, 255)
            if mode == "sharpen"
            else blurred
        )
    np.testing.assert_allclose(result, expected, atol=3e-5)


def test_legacy_display_gaussian_keeps_sigma():
    saved = migrate_pipeline({"display": {"image_filter": "gaussian"}})
    selected = next(n for n in saved["software"] if n["type"] == "filter")
    assert selected["params"]["sigma"] == 0.7


@pytest.mark.parametrize(
    "settings",
    [
        {"filter": "canny", "kernel": 9},
        {"filter": "median", "kernel": 11},
        {"filter": "sobel", "kernel": 15},
        {"filter": "laplacian", "kernel": 21},
        {"filter": "canny", "edge_low": 150, "edge_high": 150},
        {"filter": "gaussian", "sigma": 0},
        {"filter": "adaptive", "block_size": 4},
        {"filter": "morphology", "iterations": 6},
    ],
)
def test_invalid_filter_configuration_rejected(settings):
    with pytest.raises(ValueError):
        validate_pipeline(document(node("software", "filter", **settings)))


@pytest.mark.parametrize(
    "operation,code",
    [
        ("erode", cv2.MORPH_ERODE),
        ("dilate", cv2.MORPH_DILATE),
        ("open", cv2.MORPH_OPEN),
        ("close", cv2.MORPH_CLOSE),
        ("gradient", cv2.MORPH_GRADIENT),
        ("tophat", cv2.MORPH_TOPHAT),
        ("blackhat", cv2.MORPH_BLACKHAT),
    ],
)
def test_morphology_operation_and_configuration(operation, code):
    original = image()
    configured = params(
        "morphology",
        morph_operation=operation,
        shape="cross",
        kernel=5,
        iterations=2,
        border="replicate",
    )
    expected = cv2.morphologyEx(
        original,
        code,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
        iterations=2,
        borderType=cv2.BORDER_REPLICATE,
    )
    np.testing.assert_allclose(apply_filter(original, configured), expected, atol=1e-5)


def test_threshold_and_canny_match_opencv():
    original = image()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)[..., 0].round().astype(np.uint8)
    for mode, settings, expected in [
        (
            "threshold",
            {"threshold": 91, "maximum": 201},
            cv2.threshold(gray, 91, 201, cv2.THRESH_BINARY)[1],
        ),
        (
            "threshold",
            {"threshold_type": "otsu"},
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
        ),
        (
            "canny",
            {"kernel": 5, "edge_low": 30, "edge_high": 110, "l2_gradient": True},
            cv2.Canny(gray, 30, 110, apertureSize=5, L2gradient=True),
        ),
        (
            "adaptive",
            {"adaptive_method": "mean", "invert": True, "block_size": 7, "adaptive_c": 4},
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 4
            ),
        ),
    ]:
        np.testing.assert_array_equal(
            apply_filter(original, params(mode, **settings)),
            np.repeat(expected[..., None], 3, axis=2),
        )


def test_filter_configuration_roundtrip_and_repetition():
    saved = document(
        node("software", "filter", filter="clahe", clip_limit=7, tile_size=12, mix=0.4)
    )
    saved["software"].insert(
        2,
        node(
            "software",
            "filter",
            filter="morphology",
            kernel=7,
            iterations=3,
            shape="cross",
            morph_operation="gradient",
        ),
    )
    assert validate_pipeline(json.loads(json.dumps(saved))) == saved
    original = image()
    first = apply_filter(original, saved["software"][1]["params"])
    second = apply_filter(first, saved["software"][2]["params"])
    reversed_order = apply_filter(
        apply_filter(original, saved["software"][2]["params"]), saved["software"][1]["params"]
    )
    assert not np.allclose(second, reversed_order)
    assert "threshold" not in filter_fields(params("threshold", threshold_type="otsu"))


def test_editor_filter_rows_follow_mode_and_logging_lock():
    from test_capture_panel import popup_environment

    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import default_pipeline, node
from topdon_duo.image_filters import FILTER_MODES, filter_fields
app = QApplication([])
window = ViewWindow(lambda message: None)
window.show()
editor = window.pipeline_editor
saved = default_pipeline()
item = node("software", "filter", filter="gaussian", kernel=15)
item["expanded"] = True
saved["software"].insert(1, item)
editor.update_state({"pipeline": saved, "pipeline_serial": 1}, False)
for mode in FILTER_MODES:
    editor.change(editor.document["software"][1], "filter", mode)
    editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, False)
    current = editor.document["software"][1]
    rows = editor.widgets[item["id"]][1]
    for key, row in rows.items():
        assert row.isHidden() == (key not in filter_fields(current["params"])), (mode, key)
assert editor.document["software"][1]["params"]["kernel"] == 0
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial}, True)
for row in editor.widgets[item["id"]][1].values():
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


@pytest.mark.parametrize(
    "mode", ["box", "median", "bilateral", "gaussian", "sharpen", "highpass", "morphology"]
)
@pytest.mark.parametrize("channels", ["color", "luminance"])
def test_largest_supported_kernels_and_channel_options(mode, channels):
    from topdon_duo.image_filters import allowed_kernels

    original = image()
    result = apply_filter(
        original,
        params(
            mode,
            kernel=max(allowed_kernels(mode)),
            channels=channels,
            border="constant",
            sigma=10,
            sigma_color=150,
            sigma_space=10,
            iterations=5,
        ),
    )
    assert result.shape == original.shape and np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 255


@pytest.mark.parametrize("mode", ["sobel", "scharr", "laplacian"])
@pytest.mark.parametrize("direction", ["x", "y", "magnitude"])
def test_derivative_options_and_large_aperture(mode, direction):
    original = image()
    result = apply_filter(
        original, params(mode, kernel=7, direction=direction, gain=0.1, border="replicate")
    )
    assert result.shape == original.shape and np.isfinite(result).all()
    np.testing.assert_array_equal(result[..., 0], result[..., 1])
    np.testing.assert_array_equal(result[..., 0], result[..., 2])


def test_box_kernel_and_blur_sigma_change_result():
    original = image()
    a = apply_filter(original, params("box", kernel=3))
    b = apply_filter(original, params("box", kernel=11))
    assert not np.allclose(a, b)
    np.testing.assert_allclose(b, cv2.blur(original, (11, 11)), atol=1e-5)
    assert not np.allclose(
        apply_filter(original, params("gaussian", sigma=0.3)),
        apply_filter(original, params("gaussian", sigma=3)),
    )
