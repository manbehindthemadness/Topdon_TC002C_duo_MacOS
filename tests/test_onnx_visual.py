import sys
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from support.onnx_visual import fake_runtime

from topdon_duo import onnx_models
from topdon_duo.onnx_upsampling import ONNXRuntime, ONNXUpsampler
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_titles import node_title


@pytest.mark.parametrize(
    "model,factor",
    [
        (name, spec["factor"])
        for name, spec in onnx_models.MODELS.items()
        if spec.get("output_channels", 3) != 1
    ],
)
@pytest.mark.parametrize("color", [False, True])
def test_inference_shapes_color_order_session_reuse_and_input_ownership(
    monkeypatch: pytest.MonkeyPatch, model: Any, factor: Any, color: Any
) -> None:
    calls = fake_runtime(monkeypatch)
    image = np.full((4, 6, 3) if color else (4, 6), 100, dtype=np.uint8)
    if color:
        image[:] = [10, 70, 200]
    original = image.copy()
    runtime = ONNXRuntime()
    result = runtime.apply(image, model, amount=0.5)
    assert result.shape == ((4 * factor, 6 * factor, 3) if color else (4 * factor, 6 * factor))
    assert result.dtype == np.uint8
    assert np.array_equal(image, original)
    assert (
        np.max(np.abs(result.astype(int) - cv2.resize(image, (6 * factor, 4 * factor)).astype(int)))
        <= 2
    )
    if onnx_models.MODELS[model]["rgb"] and color:
        pixel_range = onnx_models.MODELS[model].get("pixel_range", 1)
        pixel = (
            calls[1][0, 0, 0, :]
            if onnx_models.MODELS[model].get("layout") == "nhwc"
            else calls[1][0, :, 0, 0]
        )
        assert np.allclose(
            pixel,
            np.array([200, 70, 10]) * pixel_range / 255.0
            + onnx_models.MODELS[model].get("input_offset", 0),
        )
    runtime.apply(image, model)
    assert len(calls) == 3  # One session, two inferences.


def test_fixed_espcn_tiles_preserve_rectangular_geometry_and_cover_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_runtime(monkeypatch, fixed_shape=True)
    image = np.random.default_rng(10).integers(0, 256, (192, 256), dtype=np.uint8)
    output = ONNXRuntime().apply(image, "espcn")
    assert np.array_equal(output, np.repeat(np.repeat(image, 3, axis=0), 3, axis=1))
    assert len(calls) == 3  # Session plus two overlapping 224x224 tiles.
    assert all(blob.shape == (1, 1, 224, 224) for blob in calls[1:])


@pytest.mark.parametrize(
    "model",
    ["style-mosaic", "style-candy", "style-rain-princess", "style-udnie", "style-pointillism"],
)
@pytest.mark.parametrize("shape", [(168, 224), (224, 168)])
def test_style_rgb_range_letterboxing_and_rectangular_geometry(
    monkeypatch: pytest.MonkeyPatch, model: Any, shape: Any
) -> None:
    calls = fake_runtime(monkeypatch)
    height, width = shape
    image = np.zeros((height, width, 3), np.uint8)
    image[: height // 2, : width // 2] = (10, 40, 200)
    image[height // 2 :, width // 2 :] = (170, 20, 50)
    original = image.copy()
    runtime = ONNXRuntime()
    result = runtime.apply(image, model)
    assert result.shape == image.shape
    assert np.array_equal(result, original)  # Identity inference must not shift/crop/stretch.
    assert np.array_equal(image, original)
    blob = calls[1]
    assert blob.shape == (1, 3, 224, 224)
    top, left = (224 - height) // 2, (224 - width) // 2
    assert np.array_equal(blob[0, :, top, left], [200, 40, 10])
    assert blob.max() == 200  # Style models consume raw RGB 0..255, not normalized 0..1.
    fake_runtime(monkeypatch, bad_output=True)
    with pytest.raises(ValueError, match="style output"):
        ONNXRuntime().apply(image, model)


def test_style_blend_and_zero_amount_do_not_change_baseline_or_load_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_runtime(monkeypatch, invert_style=True)
    image = np.full((192, 256, 3), (10, 40, 200), np.uint8)
    runtime = ONNXRuntime()
    assert runtime.apply(image, "style-candy", amount=0) is image
    assert calls == []
    styled = runtime.apply(image, "style-candy", amount=1)
    assert np.array_equal(styled, 255 - image)
    blended = runtime.apply(image, "style-candy", amount=0.35)
    assert np.array_equal(blended, cv2.addWeighted(255 - image, 0.35, image, 0.65, 0))


def test_line_art_normalized_rgb_input_and_single_channel_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_runtime(monkeypatch)
    image = np.full((192, 256, 3), (30, 90, 180), np.uint8)
    original = image.copy()
    output = ONNXRuntime().apply(image, "style-line-art")
    assert calls[1].shape == (1, 3, 256, 256)
    assert np.allclose(calls[1][0, :, 32, 0], np.array([180, 90, 30]) / 255)
    assert output.shape == image.shape
    assert np.all(output == 100)  # Fake one-channel output is replicated to RGB.
    assert np.array_equal(image, original)
    fake_runtime(monkeypatch, bad_output=True)
    with pytest.raises(ValueError, match="style output"):
        ONNXRuntime().apply(image, "style-line-art")


def test_animegan_signed_nhwc_letterboxing_and_inverse_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_runtime(monkeypatch)
    image = np.zeros((384, 512, 3), np.uint8)
    image[:192, :256] = (0, 127, 255)
    image[192:, 256:] = (255, 40, 0)
    original = image.copy()
    output = ONNXRuntime().apply(image, "style-animegan-sketch")
    assert calls[1].shape == (1, 512, 512, 3)
    assert np.allclose(calls[1][0, 64, 0], [1, 127 / 127.5 - 1, -1], atol=1e-7)
    assert np.array_equal(output, original)
    assert np.array_equal(image, original)
    fake_runtime(monkeypatch, bad_output=True)
    with pytest.raises(ValueError, match="style output"):
        ONNXRuntime().apply(image, "style-animegan-sketch")


@pytest.mark.parametrize(
    "model", [name for name, spec in onnx_models.MODELS.items() if spec.get("task") == "style"]
)
def test_every_style_is_selectable_and_round_trips_saved_settings(model: Any) -> None:
    from test_pipeline import raw_pipeline

    style = node("software", "onnx_style", model=model, backend="coreml", amount=0.75)
    document = raw_pipeline(style)
    assert validate_pipeline(deepcopy(document)) == document


def test_ffdnet_odd_geometry_noise_normalization_and_session_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_runtime(monkeypatch)
    image = np.random.default_rng(8).integers(0, 256, (5, 7), dtype=np.uint8)
    runtime = ONNXRuntime()
    assert np.array_equal(runtime.apply(image, "ffdnet-gray", noise=15), image)
    assert calls[1][0].shape == (1, 1, 6, 8)
    assert np.allclose(calls[1][1], 15 / 255)
    assert np.allclose(calls[1][0][0, 0, -1], calls[1][0][0, 0, -2])
    runtime.apply(image, "ffdnet-gray", noise=75)
    assert len(calls) == 3
    assert np.allclose(calls[2][1], 75 / 255)
    for invalid in (-1, 76, np.nan):
        with pytest.raises(ValueError, match="sigma"):
            runtime.apply(image, "ffdnet-gray", noise=invalid)


@pytest.mark.parametrize("model", ["dncnn-25", "ffdnet-gray", "realesr-general-x4v3"])
def test_new_models_apple_shape_specialization(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    ONNXRuntime().apply(np.zeros((5, 7, 3), np.uint8), model, "coreml")
    shape = (6, 8) if model == "ffdnet-gray" else (5, 7)
    assert calls[0]["sess_options"].overrides == {"batch": 1, "height": shape[0], "width": shape[1]}
    assert calls[0]["providers"][0][1]["RequireStaticInputShapes"] == "1"


@pytest.mark.parametrize(
    "model", [name for name, spec in onnx_models.MODELS.items() if spec.get("task") != "style"]
)
@pytest.mark.parametrize("apple_available", [False, True])
def test_regular_ai_dispatch_single_pass_fallback_measurements_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, model: Any, apple_available: Any
) -> None:
    from test_pipeline import raw_pipeline
    from test_render import frame_with_preview

    from topdon_duo.camera import decode_duo_frame
    from topdon_duo.pipeline_processing import PipelineProcessor

    calls, engines = [], []
    factor = onnx_models.MODELS[model]["factor"]

    class Engine:
        def __init__(self) -> None:
            """
            Init.
            """
            self.closed = False
            engines.append(self)

        def apply(
            self,
            image: Any,
            selected: Any,
            backend: Any,
            compute: Any,
            amount: Any,
            noise: Any = 15,
        ) -> Any:
            """
            Apply.
            """
            calls.append((image.shape, selected, backend, compute, amount, noise))
            return cv2.resize(image, (image.shape[1] * factor, image.shape[0] * factor))

        def close(self) -> None:
            """
            Close.
            """
            self.closed = True

    monkeypatch.setattr("topdon_duo.processing.branch.ONNXUpsampler", Engine)
    ai = node(
        "software",
        "enhance",
        model=model,
        backend="coreml",
        noise=32,
        passes=5,
        input="preview" if factor == 1 else "native",
    )
    document = raw_pipeline(ai)
    original = deepcopy(document)
    assert validate_pipeline(document) == original
    assert "pass" not in node_title("software", ai).lower()
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    measurements = raw.astype(np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=apple_available)
    try:
        output, _ = processor.process(frame, measurements, document, scale=factor)
        assert output.shape == (192 * factor, 256 * factor, 3)
        assert calls[0][0][:2] == (192, 256)
        assert calls[0][1] == model
        assert calls[0][2] == ("coreml" if apple_available else "cpu")
        assert calls[0][-1] == (32 if factor == 1 else 15)
        processor.process(frame, measurements, document, scale=1)
        assert len(processor.onnx_models) == 1
        assert document == original
        assert np.array_equal(measurements, before)
        ai["params"]["model"] = "off"
        processor.process(frame, measurements, document)
        assert engines[0].closed
        assert len(calls) == 2
    finally:
        processor.close()


def test_apple_provider_configuration_and_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    ONNXRuntime().apply(np.zeros((4, 6), np.uint8), "espcn", "coreml", "ALL")
    assert calls[0]["providers"][0][1]["MLComputeUnits"] == "ALL"
    fake_runtime(monkeypatch, bad_output=True)
    with pytest.raises(ValueError, match="non-finite"):
        ONNXRuntime().apply(np.zeros((4, 6), np.uint8), "mewzoom")


@pytest.mark.parametrize("model", ["mewzoom-v1-2x", "mewzoom-v1-4x"])
def test_v1_apple_shape_specialization_and_size_change_reload(
    monkeypatch: pytest.MonkeyPatch, model: Any
) -> None:
    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    runtime = ONNXRuntime()
    runtime.apply(np.zeros((4, 6, 3), np.uint8), model, "coreml")
    assert calls[0]["providers"][0][1]["RequireStaticInputShapes"] == "1"
    assert calls[0]["sess_options"].overrides == {"batch": 1, "height": 4, "width": 6}
    runtime.apply(np.zeros((4, 6, 3), np.uint8), model, "coreml")
    assert len(calls) == 3  # Reuse one session for matching dimensions.
    runtime.apply(np.zeros((8, 10, 3), np.uint8), model, "coreml")
    assert calls[3]["sess_options"].overrides == {"batch": 1, "height": 8, "width": 10}
    assert len(calls) == 5


def test_v1_cpu_keeps_dynamic_shapes_and_no_apple_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_runtime(monkeypatch)
    runtime = ONNXRuntime()
    runtime.apply(np.zeros((4, 6), np.uint8), "mewzoom-v1-2x")
    runtime.apply(np.zeros((8, 10), np.uint8), "mewzoom-v1-2x")
    assert len(calls) == 3
    assert calls[0]["sess_options"].overrides == {}
    assert calls[0]["providers"] == ["CPUExecutionProvider"]


def test_zero_amount_and_pixel_budget_do_not_load_models(monkeypatch: pytest.MonkeyPatch) -> None:
    load = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr("topdon_duo.onnx_upsampling.verified_model", load)
    image = np.zeros((768, 1024), np.uint8)
    assert ONNXRuntime().apply(image, "mewzoom", amount=0) is image
    assert ONNXUpsampler().apply(image, "mewzoom", amount=0) is image
    with pytest.raises(ValueError, match="4 megapixels"):
        ONNXRuntime().apply(image, "mewzoom")
    load.assert_not_called()


def test_helper_crash_is_contained_and_configuration_change_clears_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "topdon_duo.onnx_upsampling.MODEL_DOWNLOADS.request", lambda *args, **kwargs: True
    )
    process = Mock()
    process.is_alive.return_value = False
    connection = Mock()
    connection.poll.return_value = True
    connection.recv.side_effect = EOFError
    engine = ONNXUpsampler()
    engine.key = "espcn", "cpu", "CPUAndGPU"
    engine.process, engine.connection = process, connection
    with pytest.raises(ValueError, match="helper exited"):
        engine.apply(np.zeros((4, 4), np.uint8), "espcn")
    assert engine.process is None
    process.close.assert_called_once()
