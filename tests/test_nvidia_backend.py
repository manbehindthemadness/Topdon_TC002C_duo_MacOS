"""
Verify CUDA dispatch, provider failures, portable preferences, and worker ownership.
"""

import subprocess
import sys
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from support.onnx_visual import fake_runtime
from test_capture_panel import popup_environment
from test_pipeline import raw_pipeline
from test_render import frame_with_preview

from topdon_duo.camera import decode_duo_frame
from topdon_duo.onnx_upsampling import ONNXRuntime, ONNXUpsampler
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_processing import PipelineProcessor


def test_cuda_provider_order_reuse_and_unchanged_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Preserve model bytes and dynamic dimensions with explicit CUDA then CPU providers.
    """
    calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    sys.modules["onnxruntime"].get_available_providers = lambda: ["CUDAExecutionProvider"]
    runtime = ONNXRuntime()
    image = np.full((4, 6), 120, np.uint8)
    runtime.apply(image, "espcn", "cuda")
    runtime.apply(np.full((8, 10), 120, np.uint8), "espcn", "cuda")
    assert calls[0]["providers"][0][0] == "CUDAExecutionProvider"
    assert calls[0]["providers"][0][1]["use_tf32"] == "0"
    assert calls[0]["providers"][1] == "CPUExecutionProvider"
    assert not calls[0]["sess_options"].overrides
    assert len(calls) == 3


@pytest.mark.parametrize("advertised", [False, True])
def test_cuda_does_not_silently_initialize_cpu_only_session(
    monkeypatch: pytest.MonkeyPatch,
    advertised: bool,
) -> None:
    """
    Report missing CUDA libraries even if ORT falls back during session construction.
    """
    fake_runtime(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    ort = sys.modules["onnxruntime"]
    ort.get_available_providers = lambda: ["CUDAExecutionProvider"] if advertised else []
    monkeypatch.setattr(
        ort.InferenceSession, "get_providers", lambda self: ["CPUExecutionProvider"]
    )
    with pytest.raises(ValueError, match="CUDA"):
        ONNXRuntime().apply(np.zeros((4, 6), np.uint8), "espcn", "cuda")


@pytest.mark.parametrize("model", ["acnet", "espcn", "dncnn-25", "style-candy"])
@pytest.mark.parametrize("detected", [False, True])
def test_cuda_dispatch_preserves_measurements_preferences_and_releases_workers(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    detected: bool,
) -> None:
    """
    Exercise scheduler routing and cache pruning for bundled and downloadable models.
    """
    calls, engines = [], []

    class Engine:
        """
        Simulate isolated inference without USB, GPU, or network access.
        """

        error = ""

        def __init__(self) -> None:
            """
            Track cache lifetime.
            """
            self.closed = False
            engines.append(self)

        def apply(
            self,
            image: np.ndarray,
            selected: str,
            backend: str,
            compute: str,
            amount: float,
            **kwargs: Any,
        ) -> np.ndarray:
            """
            Record execution and emulate the selected model's output geometry.
            """
            calls.append((image.shape[:2], selected, backend, amount))
            factor = 2 if selected.startswith("acnet") else 3 if selected == "espcn" else 1
            return cv2.resize(image, None, fx=factor, fy=factor)

        def close(self) -> None:
            """
            Record release of a disconnected worker.
            """
            self.closed = True

    monkeypatch.setattr("topdon_duo.processing.branch.ONNXUpsampler", Engine)
    item = (
        node("software", "onnx_style", model=model, backend="cuda")
        if model.startswith("style-")
        else node("software", "enhance", model=model, backend="cuda", denoise=2, passes=2)
    )
    document = raw_pipeline(item)
    original = deepcopy(document)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    measurements = raw.astype(np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=False, nvidia_available=detected)
    try:
        processor.process(frame, measurements, document, scale=1)
        if model != "acnet" or detected:
            assert calls[0][2] == ("cuda" if detected else "cpu")
            assert calls[0][1] == ("acnet-legacy-hdn2" if model == "acnet" else model)
            assert len(calls) == (2 if model == "acnet" else 1)
            assert len(processor.onnx_models) == 1
        else:
            assert not calls and processor.models
        assert document == original and validate_pipeline(document) == original
        assert np.array_equal(measurements, before)
        processor.process(frame, measurements, document, scale=1)
        if detected:
            previous = engines[0]
            item["params"]["backend"] = "cpu"
            processor.process(frame, measurements, document, scale=1)
            assert previous.closed
        item["bypass"] = True
        processor.process(frame, measurements, document, scale=1)
        assert not processor.onnx_models
        assert all(engine.closed for engine in engines)
    finally:
        processor.close()


def test_bundled_acnet_never_requests_a_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Reject oversized bundled inference before spawning or consulting downloads.
    """

    def unexpected(*args: Any, **kwargs: Any) -> None:
        """
        Fail on network preparation.
        """
        raise AssertionError("bundled weights must not download")

    monkeypatch.setattr("topdon_duo.onnx_upsampling.MODEL_DOWNLOADS.request", unexpected)
    with pytest.raises(ValueError, match="4 megapixels"):
        ONNXUpsampler().apply(np.zeros((1024, 1024), np.uint8), "acnet-legacy-hdn0", "cuda")


def test_bundled_acnet_reuses_worker_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Send bundled ACNet directly to the worker without any model installation job.
    """
    request = Mock(side_effect=AssertionError("bundled model download"))
    monkeypatch.setattr("topdon_duo.onnx_upsampling.MODEL_DOWNLOADS.request", request)
    image = np.zeros((8, 8), np.uint8)
    output = np.zeros((16, 16), np.uint8)
    engine = ONNXUpsampler()
    engine.key = ("acnet-legacy-hdn0", "cuda", "CPUAndGPU")
    engine.connection = Mock()
    engine.connection.poll.return_value = True
    engine.connection.recv.return_value = (output, 1.0, "")
    engine.process = Mock()
    engine.process.is_alive.return_value = False
    try:
        assert engine.apply(image, "acnet-legacy-hdn0", "cuda") is output
        engine.connection.send.assert_called_once_with((image, 1.0, 15))
        request.assert_not_called()
    finally:
        engine.close()


def test_cuda_controls_and_saved_preferences_in_offscreen_ui() -> None:
    """
    Show CUDA only for supported models and keep Apple settings saved on Jetson.
    """
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
item = node("software", "enhance", model="acnet", backend="cuda", apple_compute="ALL")
editor.document["software"].insert(1, item)
original = deepcopy(editor.document)
editor.rebuild()
for detected in (True, False, True):
    state = {"pipeline": original, "apple_acceleration": {"available": False},
             "nvidia_acceleration": {"available": detected, "reason": "CUDA ready"}}
    editor.update_state(state, False)
    rows = editor.widgets[item["id"]][1]
    assert rows["backend"].isHidden() is (not detected)
    assert rows["apple_compute"].isHidden()
    combo = rows["backend"].input
    assert combo.model().item(combo.findData("cuda")).isEnabled() is detected
    assert not combo.model().item(combo.findData("coreml")).isEnabled()
    assert editor.document == original
    badge = editor.widgets[item["id"]][4]
    assert bool(badge.text()) is (not detected)
editor.change(item, "model", "anime4k09")
editor.update_state({"pipeline_serial": editor.edit_serial,
                     "nvidia_acceleration": {"available": True}}, False)
rows = editor.widgets[item["id"]][1]
assert rows["backend"].isHidden() and not rows["backend"].input.isEnabled()
assert item["params"]["backend"] == "cuda"
assert item["params"]["apple_compute"] == "ALL"
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
