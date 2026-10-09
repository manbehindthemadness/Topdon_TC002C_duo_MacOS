"""Contract tests do not imply that real Core ML hardware was exercised."""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from topdon_duo.coreml_upsampling import CoreMLUpsampler, _CoreMLRuntime
from topdon_duo.pipeline import node
from topdon_duo.pipeline_titles import node_title


def fake_runtime(monkeypatch, providers=None, tensor=None):
    calls = []

    class Session:
        def __init__(self, data, **kwargs):
            calls.append(kwargs)

        def disable_fallback(self):
            calls.append("disable_fallback")

        def get_providers(self):
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, outputs, feed):
            if tensor is not None:
                return [tensor]
            gray = feed["input"][0, 0]
            return [cv2.resize(gray, None, fx=2, fy=2)[None, None]]

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: providers or ["CoreMLExecutionProvider"],
            SessionOptions=SimpleNamespace,
            InferenceSession=Session,
        ),
    )
    return calls


@pytest.mark.parametrize("color", [False, True])
def test_isolated_runtime_reuses_session_and_preserves_input(monkeypatch, color):
    calls = fake_runtime(monkeypatch)
    shape = (12, 16, 3) if color else (12, 16)
    image = np.full(shape, 100, dtype=np.uint8)
    original = image.copy()
    engine = _CoreMLRuntime()
    output = engine.apply(image, amount=0.5)
    assert output.shape[:2] == (24, 32)
    assert output.dtype == np.uint8
    assert np.array_equal(image, original)
    engine.apply(image)
    assert len(calls) == 2
    assert calls[0]["providers"][0][1]["MLComputeUnits"] == "CPUAndGPU"
    engine.apply(image, denoise=1, compute="ALL")
    assert len(calls) == 4


def test_linux_node_fails_cleanly_but_zero_amount_is_noop(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    engine = CoreMLUpsampler()
    image = np.zeros((4, 4), dtype=np.uint8)
    assert engine.apply(image, amount=0) is image
    with pytest.raises(ValueError, match="requires macOS"):
        engine.apply(image)


def test_missing_runtime_is_actionable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ValueError, match="uv sync"):
        _CoreMLRuntime().apply(np.zeros((4, 4), dtype=np.uint8))


def test_no_coreml_provider_does_not_silently_select_cpu(monkeypatch):
    calls = fake_runtime(monkeypatch, providers=["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="does not provide"):
        _CoreMLRuntime().apply(np.zeros((4, 4), dtype=np.uint8))
    assert not calls


def test_invalid_model_output_is_reported(monkeypatch):
    fake_runtime(monkeypatch, tensor=np.full((1, 1, 8, 8), np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        _CoreMLRuntime().apply(np.zeros((4, 4), dtype=np.uint8))


def test_new_node_has_independent_defaults_and_title():
    item = node("software", "coreml_acnet")
    assert item["params"]["input"] == "native"
    assert "CPU + GPU" in node_title("software", item)
    assert node("software", "enhance")["params"]["model"] == "anime4k09"


def test_pipeline_node_executes_independently_and_can_be_bypassed(monkeypatch):
    from test_pipeline import process, raw_pipeline

    fake_runtime(monkeypatch)
    monkeypatch.setattr("topdon_duo.processing.branch.CoreMLUpsampler", _CoreMLRuntime)
    item = node("software", "coreml_acnet")
    document = raw_pipeline(item)
    assert process(document).shape == (192, 256, 3)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError, match="requires macOS"):
        process(document)
    item["bypass"] = True
    assert process(document).shape == (192, 256, 3)


@pytest.mark.parametrize("level", range(4))
def test_real_coreml_matches_upstream_reference_when_runtime_is_installed(level):
    if sys.platform != "darwin":
        pytest.skip("Core ML requires macOS")
    ort = pytest.importorskip("onnxruntime")
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        pytest.skip("Core ML provider not installed")
    with np.load(Path(__file__).parent / "fixtures/acnet-native-reference.npz") as reference:
        image = reference["input"]
        original = image.copy()
        engine = CoreMLUpsampler()
        try:
            result = engine.apply(image, denoise=level)
        except ValueError as exc:
            if "Failed to create a working directory appropriate for URL" in str(exc):
                pytest.skip("Core ML compilation is blocked by this sandbox's temp directory")
            raise
        finally:
            engine.close()
        # Core ML may use reduced precision internally, unlike the CPU reference.
        assert np.abs(result.astype(int) - reference[f"hdn{level}"].astype(int)).max() <= 2
        assert np.array_equal(image, original)


def _exiting_worker(connection, denoise, compute):
    import os

    connection.recv()
    os._exit(17)


def _echo_worker(connection, denoise, compute):
    try:
        while True:
            image, _amount = connection.recv()
            connection.send((image.copy(), 1.5, ""))
    except EOFError:
        pass


def test_native_helper_death_does_not_kill_or_retry_viewer(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("topdon_duo.coreml_upsampling._coreml_worker", _exiting_worker)
    engine = CoreMLUpsampler()
    image = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="exited unexpectedly.*17"):
        engine.apply(image)
    assert engine._process is None
    with pytest.raises(ValueError, match="exited unexpectedly"):
        engine.apply(image)


def test_helper_reused_and_closed_without_changing_image(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("topdon_duo.coreml_upsampling._coreml_worker", _echo_worker)
    engine = CoreMLUpsampler()
    image = np.zeros((4, 4), dtype=np.uint8)
    try:
        assert np.array_equal(engine.apply(image), image)
        process = engine._process
        engine.apply(image)
        assert engine._process is process
        assert engine.elapsed_ms == 1.5
    finally:
        engine.close()
    assert engine._process is None
