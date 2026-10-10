"""
Verify reciprocal GPU fallback without changing saved preferences or measurements.
"""

import subprocess
import sys
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import pytest
from test_capture_panel import popup_environment
from test_pipeline import raw_pipeline
from test_render import frame_with_preview

from topdon_duo.acceleration import select_backend
from topdon_duo.pipeline import node
from topdon_duo.pipeline_processing import PipelineProcessor


@pytest.mark.parametrize(
    "apple,nvidia", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("requested", ["cpu", "coreml", "cuda"])
def test_gpu_preference_order_and_explicit_cpu(requested: str, apple: bool, nvidia: bool) -> None:
    """
    Keep an available preferred GPU, try its alternate, and honor explicit CPU.
    """
    expected = {
        "cpu": "cpu",
        "coreml": "coreml" if apple else "cuda" if nvidia else "cpu",
        "cuda": "cuda" if nvidia else "coreml" if apple else "cpu",
    }
    assert select_backend(requested, apple, nvidia) == expected[requested]


@pytest.mark.parametrize(
    "requested,apple,nvidia,expected",
    [
        ("coreml", False, True, "cuda"),
        ("cuda", True, False, "coreml"),
    ],
)
@pytest.mark.parametrize("model", ["acnet", "espcn", "style-candy"])
def test_alternate_gpu_dispatch_reuse_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    apple: bool,
    nvidia: bool,
    expected: str,
    model: str,
) -> None:
    """
    Route ACNet and visual ONNX models through the alternate GPU and retain caches.
    """
    calls = []
    factor = 2 if model == "acnet" else 3 if model == "espcn" else 1

    class Engine:
        """
        Simulate both spawned GPU engines without hardware access.
        """

        error = ""

        def __init__(self) -> None:
            """
            Record the cached engine's lifetime.
            """
            self.closed = False

        def apply(self, image: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
            """
            Capture backend or denoising arguments and emulate output geometry.
            """
            calls.append(args)
            result = cv2.resize(image, None, fx=factor, fy=factor)
            return result

        def close(self) -> None:
            """
            Release the disconnected engine.
            """
            self.closed = True

    monkeypatch.setattr("topdon_duo.processing.branch.ONNXUpsampler", Engine)
    monkeypatch.setattr("topdon_duo.processing.branch.CoreMLUpsampler", Engine)
    item = (
        node("software", "onnx_style", model=model, backend=requested)
        if model.startswith("style-")
        else node(
            "software",
            "enhance",
            model=model,
            backend=requested,
            passes=2,
            denoise=2,
            apple_compute="ALL",
        )
    )
    document = raw_pipeline(item)
    original = deepcopy(document)
    frame, _ = frame_with_preview()
    measurements = np.ones((192, 256), np.float32)
    before = measurements.copy()
    processor = PipelineProcessor(apple_available=apple, nvidia_available=nvidia)
    try:
        processor.process(frame, measurements, document, scale=1)
        cache = (
            processor.coreml_models
            if model == "acnet" and expected == "coreml"
            else processor.onnx_models
        )
        assert len(cache) == 1
        engine = next(iter(cache.values()))
        assert not engine.closed
        if model == "acnet" and expected == "coreml":
            assert calls == [(2, "ALL", 1.0), (2, "ALL", 1.0)]
        else:
            assert calls[0][1] == expected
            assert len(calls) == (2 if model == "acnet" else 1)
        processor.process(frame, measurements, document, scale=1)
        assert not engine.closed
        assert document == original
        np.testing.assert_array_equal(measurements, before)
        item["params"]["backend"] = "cpu"
        processor.process(frame, measurements, document, scale=1)
        assert engine.closed
        assert not processor.coreml_models
    finally:
        processor.close()


def test_ui_reports_alternate_gpu_and_restores_apple_compute_devices() -> None:
    """
    Display effective fallback and allow Apple device selection on a saved CUDA node.
    """
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
item = node("software", "enhance", model="acnet", backend="coreml", apple_compute="ALL")
editor.document["software"].insert(1, item)
for requested, apple, nvidia, effective, label in (
    ("coreml", False, True, "cuda", "NVIDIA CUDA"),
    ("cuda", True, False, "coreml", "Apple Core ML"),
    ("coreml", False, False, "cpu", "CPU"),
):
    item["params"]["backend"] = requested
    original = deepcopy(editor.document)
    state = {"pipeline": original, "apple_acceleration": {"available": apple},
             "nvidia_acceleration": {"available": nvidia}}
    editor.rebuild()
    editor.update_state(state, False)
    rows = editor.widgets[item["id"]][1]
    badge = editor.widgets[item["id"]][4]
    assert badge.text().startswith(label + " execution")
    assert rows["backend"].value() == effective
    assert rows["apple_compute"].isHidden() is (not apple)
    if apple:
        assert rows["apple_compute"].input.isEnabled()
        assert rows["apple_compute"].value() == "ALL"
    assert editor.document == original
    editor.rebuild()
    rows = editor.widgets[item["id"]][1]
    assert rows["backend"].value() == effective
    assert editor.document == original
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
