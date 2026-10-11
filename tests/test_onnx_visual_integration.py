import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from topdon_duo import onnx_models
from topdon_duo.enhancement_limits import enhancement_pass_limits
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_titles import node_title


def assert_scaled_call(output: np.ndarray, call: tuple, backend: str) -> None:
    """
    Check the common native input geometry, viewer scale and effective backend.
    """
    assert output.shape == (768, 1024, 3)
    assert call[0][:2] == (192, 256)
    assert call[2] == backend


@pytest.mark.parametrize("apple_available", [False, True])
def test_style_pipeline_preserves_size_radiometry_settings_and_releases_helper(
    monkeypatch: pytest.MonkeyPatch, apple_available: Any
) -> None:
    from test_pipeline import raw_pipeline
    from test_render import frame_with_preview

    from topdon_duo.camera import decode_duo_frame
    from topdon_duo.pipeline_processing import PipelineProcessor

    calls, engines = [], []

    class Engine:
        def __init__(self) -> None:
            """
            Init.
            """
            self.closed = False
            engines.append(self)

        def apply(self, image: Any, model: Any, backend: Any, compute: Any, amount: Any) -> Any:
            """
            Apply.
            """
            assert not self.closed
            calls.append((image.shape, model, backend, compute, amount))
            return image.copy()

        def close(self) -> None:
            """
            Close.
            """
            self.closed = True

    monkeypatch.setattr("topdon_duo.processing.branch.ONNXUpsampler", Engine)
    style = node("software", "onnx_style", model="style-candy", backend="coreml", amount=0.35)
    document = raw_pipeline(style)
    original = deepcopy(document)
    assert validate_pipeline(document) == document
    assert "Candy" in node_title("software", style)
    assert "35%" in node_title("software", style)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    measurements = raw.astype(np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=apple_available)
    try:
        output, _ = processor.process(frame, measurements, document, scale=4)
        assert output.shape == (768, 1024, 3)
        assert calls[0] == (
            (192, 256, 3),
            "style-candy",
            "coreml" if apple_available else "cpu",
            "CPUAndGPU",
            0.35,
        )
        assert np.array_equal(before, measurements)
        assert document == original
        processor.process(frame, measurements, document)
        assert len(processor.onnx_models) == 1
        style["params"]["amount"] = 0
        processor.process(frame, measurements, document)
        assert engines[0].closed
        assert len(calls) == 2
    finally:
        processor.close()


def test_old_ai_preset_migrates_noise_without_changing_saved_choices() -> None:
    from test_pipeline import raw_pipeline

    ai = node("software", "enhance", model="acnet", passes=2, backend="coreml")
    del ai["params"]["noise"]
    original = deepcopy(ai)
    restored = validate_pipeline(raw_pipeline(ai))["software"][2]
    assert restored["params"].pop("noise") == 15
    assert restored == original


@pytest.mark.parametrize("apple_available,expected", [(False, "cpu"), (True, "coreml")])
def test_denoiser_pipeline_retains_size_preferences_measurements_and_closes(
    monkeypatch: pytest.MonkeyPatch, apple_available: Any, expected: Any
) -> None:
    from test_pipeline import raw_pipeline
    from test_render import frame_with_preview

    from topdon_duo.camera import decode_duo_frame
    from topdon_duo.pipeline_processing import PipelineProcessor

    calls, engines = [], []

    class Engine:
        def __init__(self) -> None:
            """
            Init.
            """
            self.closed = False
            engines.append(self)

        def apply(
            self, image: Any, model: Any, backend: Any, compute: Any, amount: Any, noise: Any = 15
        ) -> Any:
            """
            Apply.
            """
            assert not self.closed
            calls.append((image.shape, model, backend, compute, amount, noise))
            return image.copy()

        def close(self) -> None:
            """
            Close.
            """
            self.closed = True

    monkeypatch.setattr("topdon_duo.processing.branch.ONNXUpsampler", Engine)
    denoiser = node("software", "onnx_denoise", backend="coreml", noise=32)
    document = raw_pipeline(denoiser)
    original = deepcopy(document)
    assert validate_pipeline(document) == document
    assert "sigma 32" in node_title("software", denoiser)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    measurements = raw.astype(np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=apple_available)
    try:
        output, _ = processor.process(frame, measurements, document, scale=4)
        assert_scaled_call(output, calls[0], expected)
        assert calls[0][-1] == 32
        assert document == original
        assert np.array_equal(measurements, before)
        denoiser["bypass"] = True
        processor.process(frame, measurements, document)
        assert engines[0].closed
        assert len(calls) == 1
    finally:
        processor.close()


def test_node_roundtrip_title_and_downstream_acnet_size_limits() -> None:
    from test_pipeline import raw_pipeline

    sr = node("software", "onnx_superresolution", model="mewzoom", backend="coreml")
    ai = node("software", "enhance", model="acnet", passes=5)
    document = raw_pipeline(sr, ai)
    assert validate_pipeline(document) == document
    assert "MewZoom" in node_title("software", sr)
    assert enhancement_pass_limits(document)[ai["id"]] == 1
    sr["bypass"] = True
    assert enhancement_pass_limits(document)[ai["id"]] == 3


def test_v0_2x_node_is_portable_and_retains_existing_v0_4x_selection() -> None:
    from test_pipeline import raw_pipeline

    sr = node("software", "onnx_superresolution", model="mewzoom-v0-2x", backend="coreml")
    ai = node("software", "enhance", model="acnet", passes=5)
    document = raw_pipeline(sr, ai)
    assert validate_pipeline(document) == document
    assert "MewZoom V0 2×" in node_title("software", sr)
    assert enhancement_pass_limits(document)[ai["id"]] == 2
    assert node("software", "onnx_superresolution", model="mewzoom")["params"]["model"] == "mewzoom"


@pytest.mark.parametrize("model,maximum", [("mewzoom-v1-2x", 2), ("mewzoom-v1-4x", 1)])
def test_v1_node_roundtrip_and_factor_dependent_limits(model: Any, maximum: Any) -> None:
    from test_pipeline import raw_pipeline

    sr = node("software", "onnx_superresolution", model=model, backend="coreml")
    ai = node("software", "enhance", model="acnet", passes=5)
    document = raw_pipeline(sr, ai)
    assert validate_pipeline(document) == document
    assert "MewZoom V1" in node_title("software", sr)
    assert enhancement_pass_limits(document)[ai["id"]] == maximum


@pytest.mark.parametrize("apple_available,expected", [(False, "cpu"), (True, "coreml")])
def test_pipeline_cpu_fallback_retains_preferences_and_measurements(
    monkeypatch: pytest.MonkeyPatch, apple_available: Any, expected: Any
) -> None:
    from test_pipeline import raw_pipeline
    from test_render import frame_with_preview

    from topdon_duo.camera import decode_duo_frame
    from topdon_duo.pipeline_processing import PipelineProcessor

    calls = []
    engines = []

    class Engine:
        def __init__(self) -> None:
            """
            Init.
            """
            self.closed = False
            engines.append(self)

        def apply(self, image: Any, model: Any, backend: Any, compute: Any, amount: Any) -> Any:
            """
            Apply.
            """
            assert not self.closed
            calls.append((image.shape, model, backend, compute, amount))
            factor = onnx_models.MODELS[model]["factor"]
            return cv2.resize(image, (image.shape[1] * factor, image.shape[0] * factor))

        def close(self) -> None:
            """
            Close.
            """
            self.closed = True

    monkeypatch.setattr("topdon_duo.processing.branch.ONNXUpsampler", Engine)
    sr = node("software", "onnx_superresolution", model="mewzoom", backend="coreml")
    document = raw_pipeline(sr)
    original = deepcopy(document)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    measurements = raw.astype(np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=apple_available)
    try:
        output, _ = processor.process(frame, measurements, document, scale=4)
        assert_scaled_call(output, calls[0], expected)
        assert document == original
        assert np.array_equal(measurements, before)
        sr["bypass"] = True
        processor.process(frame, measurements, document)
        assert len(calls) == 1
        assert engines[0].closed
    finally:
        processor.close()


def test_popup_apple_controls_and_cpu_preference_retention(tmp_path: Path) -> None:
    import subprocess

    from test_capture_panel import popup_environment

    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
app = QApplication([])
window = ViewWindow(lambda message: None)
editor = window.pipeline_editor
sr = node("software", "onnx_superresolution", model="mewzoom", backend="coreml")
editor.document["software"].insert(-1, sr)
editor.rebuild()
editor.update_state({"pipeline": editor.document, "apple_acceleration": {"available": True}}, False)
rows = editor.widgets[sr["id"]][1]
assert not rows["backend"].isHidden() and rows["backend"].input.isEnabled()
editor.update_state({"apple_acceleration": {"available": False}}, False)
assert rows["backend"].isHidden()
assert sr["params"]["backend"] == "coreml"
assert sr["params"]["model"] == "mewzoom"
assert "passes" not in rows
denoise = node("software", "onnx_denoise", model="ffdnet-gray", backend="coreml", noise=42)
editor.document["software"].insert(-1, denoise)
editor.rebuild()
editor.update_state({"pipeline": editor.document, "apple_acceleration": {"available": True}}, False)
rows = editor.widgets[denoise["id"]][1]
assert not rows["backend"].isHidden() and rows["backend"].input.isEnabled()
assert not rows["noise"].isHidden()
denoise["params"]["model"] = "dncnn-25"
editor.update_state({"apple_acceleration": {"available": False}}, False)
assert rows["noise"].isHidden()
assert rows["backend"].isHidden()
assert denoise["params"]["backend"] == "coreml"
assert denoise["params"]["noise"] == 42
ai = node("software", "enhance", model="realesr-general-x4v3", backend="coreml", input="native", noise=37)
editor.document["software"].insert(-1, ai)
editor.rebuild()
editor.update_state({"pipeline": editor.document, "apple_acceleration": {"available": True}}, False)
rows = editor.widgets[ai["id"]][1]
assert not rows["backend"].isHidden() and rows["backend"].input.isEnabled()
assert rows["passes"].isHidden() and rows["noise"].isHidden()
assert rows["denoise"].isHidden() and not rows["input"].isHidden()
editor.change(ai, "backend", "cpu")
assert ai["params"]["model"] == "realesr-general-x4v3"
editor.change(ai, "model", "ffdnet-gray")
editor.update_state({"pipeline_serial": editor.edit_serial}, False)
assert not rows["noise"].isHidden() and rows["noise"].input.isEnabled()
assert rows["input"].isHidden() and rows["passes"].isHidden()
editor.change(ai, "backend", "coreml")
editor.update_state({"apple_acceleration": {"available": False}}, False)
assert rows["backend"].isHidden() and ai["params"]["backend"] == "coreml"
assert ai["params"]["noise"] == 37
editor.change(ai, "model", "dncnn-25")
editor.update_state({"pipeline_serial": editor.edit_serial}, False)
assert rows["noise"].isHidden()
editor.change(ai, "model", "acnet")
editor.update_state({"pipeline_serial": editor.edit_serial}, False)
assert not rows["passes"].isHidden() and not rows["denoise"].isHidden()
editor.change(ai, "model", "anime4k09")
editor.update_state({"apple_acceleration": {"available": True}, "pipeline_serial": editor.edit_serial}, False)
assert rows["backend"].isHidden()
assert rows["passes"].input.maximum() == 5
assert ai["params"]["noise"] == 37
style = node("software", "onnx_style", model="style-candy", backend="coreml", amount=0.35)
editor.document["software"].insert(-1, style)
editor.rebuild()
editor.update_state({"pipeline": editor.document, "apple_acceleration": {"available": True}}, False)
rows = editor.widgets[style["id"]][1]
assert not rows["backend"].isHidden() and rows["backend"].input.isEnabled()
assert "passes" not in rows and "noise" not in rows and "input" not in rows
editor.update_state({"apple_acceleration": {"available": False}}, False)
assert rows["backend"].isHidden()
assert style["params"] == {"model": "style-candy", "backend": "coreml", "apple_compute": "CPUAndGPU", "amount": 0.35}
window.close()
"""
    env = popup_environment()
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
