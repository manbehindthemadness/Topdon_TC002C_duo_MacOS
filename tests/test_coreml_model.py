from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from topdon_duo.coreml_model import _fields, explicit_coreml_padding


@pytest.mark.parametrize("level", range(4))
def test_only_convolution_nodes_change_and_padding_is_idempotent(level):
    data = files("topdon_duo").joinpath("models", f"acnet-legacy-hdn{level}.onnx").read_bytes()
    result = explicit_coreml_padding(data)
    assert len(result) == len(data) + 19 * 10
    assert explicit_coreml_padding(result) == result
    graph = next(payload for n, payload, _ in _fields(data) if n == 7)
    padded = next(payload for n, payload, _ in _fields(result) if n == 7)
    old = list(_fields(graph))
    new = list(_fields(padded))
    changed = [(a, b) for a, b in zip(old, new, strict=True) if a != b]
    assert len(changed) == 10  # Nine Conv plus one ConvTranspose.
    for before, after in changed:
        assert before[0] == after[0] == 1  # Graph.node; weights/other fields untouched.
        assert any(
            n == 4 and payload in (b"Conv", b"ConvTranspose")
            for n, payload, _ in _fields(before[1])
        )
        attrs = [payload for n, payload, _ in _fields(after[1]) if n == 5]
        assert attrs[-1] == b"\x0a\x04pads\x40\x00\x40\x00\x40\x00\x40\x00\xa0\x01\x07"
        assert after[1] == before[1] + b"\x2a\x11" + attrs[-1]


@pytest.mark.parametrize("level", range(4))
def test_padded_model_matches_unmodified_model_and_upstream_reference(level):
    ort = pytest.importorskip("onnxruntime")
    data = files("topdon_duo").joinpath("models", f"acnet-legacy-hdn{level}.onnx").read_bytes()
    with np.load(Path(__file__).parent / "fixtures/acnet-native-reference.npz") as reference:
        blob = reference["input"][None, None].astype(np.float32) / 255
        original = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        padded = ort.InferenceSession(
            explicit_coreml_padding(data), providers=["CPUExecutionProvider"]
        )
        baseline = original.run(None, {"input": blob})[0]
        output = padded.run(None, {"input": blob})[0]
        assert np.array_equal(output, baseline)
        result = np.clip(output[0, 0] * 255, 0, 255).round().astype(np.uint8)
        assert np.abs(result.astype(int) - reference[f"hdn{level}"].astype(int)).max() <= 1


def test_truncated_model_is_rejected():
    with pytest.raises(ValueError, match="Truncated"):
        explicit_coreml_padding(b"\x3a\x0aabc")
