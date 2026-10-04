from pathlib import Path

import cv2
import numpy as np
import pytest
from test_camera import make_frame
from test_render import frame_with_preview

from topdon_duo.anime4k09 import upscale_anime4k09
from topdon_duo.render import ThermalRenderer
from topdon_duo.upsampling import VisionUpsampler


@pytest.mark.parametrize("level", range(4))
def test_onnx_matches_independent_upstream_cpu_reference(level, capfd):
    # Created with pinned upstream ac_cli, not with this ONNX exporter/runtime.
    with np.load(Path(__file__).parent / "fixtures/acnet-native-reference.npz") as reference:
        image = reference["input"]
        original = image.copy()
        upsampler = VisionUpsampler()
        result = upsampler.apply(image, f"acnet-legacy-hdn{level}")
        assert not upsampler.error
        # Allow a uint8 rounding boundary across CPU architectures/OpenCV builds.
        assert np.abs(result.astype(int) - reference[f"hdn{level}"].astype(int)).max() <= 1
        assert np.array_equal(image, original)
        assert upsampler.elapsed_ms > 0
        assert "Targets are not supported" not in capfd.readouterr().err


@pytest.mark.parametrize(
    "rotation,source", [(0, "preview"), (90, "raw"), (180, "preview"), (270, "raw")]
)
def test_upsampling_preserves_measurements_orientation_and_display_geometry(rotation, source):
    frame, _ = frame_with_preview(preview_scale=1)
    renderer = ThermalRenderer(scale=3, rotation=rotation, image_source=source)
    original = renderer.render_detailed(frame)
    renderer.set_view_setting("upsampling", "acnet-legacy-hdn2")
    enhanced = renderer.render_detailed(frame)
    assert not renderer.upsampler.error
    assert enhanced.image.shape == original.image.shape
    assert enhanced.stats == original.stats
    assert np.array_equal(enhanced.raw_counts, original.raw_counts)
    assert np.array_equal(enhanced.temperatures_celsius, original.temperatures_celsius)
    assert enhanced.display_settings["upsampling"] == "acnet-legacy-hdn2"
    renderer.set_view_setting("upsampling", "off")
    restored = renderer.render_detailed(frame)
    assert np.array_equal(restored.image, original.image)


def test_model_is_reused_and_switching_off_releases_it(monkeypatch):
    read = cv2.dnn.readNetFromONNX
    calls = []

    def load(data):
        calls.append(1)
        return read(data)

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", load)
    renderer = ThermalRenderer(scale=2)
    renderer.set_view_setting("upsampling", "acnet-legacy-hdn1")
    renderer.render_detailed(make_frame())
    renderer.render_detailed(make_frame())
    assert len(calls) == 1
    renderer.set_view_setting("upsampling", "acnet-legacy-hdn3")
    renderer.render_detailed(make_frame())
    assert len(calls) == 2
    renderer.set_view_setting("upsampling", "off")
    assert renderer.upsampler._net is None


def test_failed_model_keeps_live_image_and_does_not_retry_every_frame(monkeypatch):
    calls = []

    def load(_data):
        calls.append(1)
        raise cv2.error("unavailable model")

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", load)
    frame, _ = frame_with_preview()
    renderer = ThermalRenderer(scale=2)
    original = renderer.render_detailed(frame)
    renderer.set_view_setting("upsampling", "acnet-legacy-hdn2")
    for _ in range(2):
        fallback = renderer.render_detailed(frame)
        assert np.array_equal(fallback.image, original.image)
        assert fallback.stats == original.stats
    assert len(calls) == 1
    assert "unavailable model" in renderer.upsampler.error
    renderer.set_view_setting("upsampling", "off")
    assert not renderer.upsampler.error


def test_hardware_color_uses_enhanced_luminance_and_keeps_bgr_output():
    upsampler = VisionUpsampler()
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :8] = (10, 60, 180)
    image[:, 8:] = (100, 90, 20)
    original = image.copy()
    result = upsampler.apply(image, "acnet-legacy-hdn0")
    assert not upsampler.error
    assert result.shape == (24, 32, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(image, original)


def test_anime4k09_matches_independent_upstream_opencl_kernel_rules():
    with np.load(Path(__file__).parent / "fixtures/anime4k09-kernel-reference.npz") as reference:
        result = upscale_anime4k09(reference["input"])
        assert np.abs(result.astype(int) - reference["output"].astype(int)).max() <= 1


@pytest.mark.parametrize("algorithm", ["anime4k09", "acnet-legacy-hdn2"])
@pytest.mark.parametrize("input_size", ["native", "preview"])
def test_enhancement_options_change_image_and_zero_amount_preserves_original(algorithm, input_size):
    frame, _ = frame_with_preview()
    renderer = ThermalRenderer(scale=3, rotation=90)
    original = renderer.render_detailed(frame)
    renderer.set_view_setting("upsampling", algorithm)
    renderer.set_view_setting("enhancement_input", input_size)
    enhanced = renderer.render_detailed(frame)
    assert not renderer.upsampler.error
    assert enhanced.image.shape == original.image.shape
    assert not np.array_equal(enhanced.image, original.image)
    assert enhanced.stats == original.stats
    assert np.array_equal(enhanced.temperatures_celsius, original.temperatures_celsius)
    renderer.set_view_setting("enhancement_amount", 0)
    assert np.array_equal(renderer.render_detailed(frame).image, original.image)


def test_anime4k09_passes_and_strength_control_edge_processing():
    with np.load(Path(__file__).parent / "fixtures/anime4k09-kernel-reference.npz") as reference:
        image = reference["input"]
        one = upscale_anime4k09(image, passes=1)
        three = upscale_anime4k09(image, passes=3)
        weak = upscale_anime4k09(image, passes=3, strength=0.25)
        assert not np.array_equal(one, three)
        assert not np.array_equal(weak, three)
        flat = np.full((8, 12, 3), 120, np.uint8)
        assert np.all(upscale_anime4k09(flat) == 120)


@pytest.mark.parametrize("algorithm", ["anime4k09", "acnet-legacy-hdn2"])
def test_input_size_option_controls_actual_processing_resolution(algorithm, monkeypatch):
    frame, _ = frame_with_preview()
    renderer = ThermalRenderer(scale=3, rotation=90)
    renderer.set_view_setting("upsampling", algorithm)
    seen = []

    def apply(image, _model, *args, **kwargs):
        seen.append(image.shape[:2])
        return cv2.resize(image, None, fx=2, fy=2)

    monkeypatch.setattr(renderer.upsampler, "apply", apply)
    renderer.render_detailed(frame)
    renderer.set_view_setting("enhancement_input", "preview")
    renderer.render_detailed(frame)
    assert seen == [(256, 192), (512, 384)]
