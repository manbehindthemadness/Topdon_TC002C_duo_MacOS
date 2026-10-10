"""
DDColor portable package contract with independent fake ONNX inference.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import cv2
import numpy as np
import pytest

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import default_pipeline, node, validate_pipeline

FOLDER = Path(__file__).parents[1] / "examples" / "custom_nodes" / "DDColor"


class FakeSession:
    """
    Predict constant Lab chroma independently of input and record model tensors.
    """

    instances: ClassVar[list[Any]] = []
    bad_output = False
    bad_schema = False
    spatial_chroma = False

    def __init__(self, path: str, options: Any, providers: list[Any]) -> None:
        """
        Record effective providers without opening an ONNX file.
        """
        self.path = path
        assert options.intra_op_num_threads == 4
        self.providers = [p if isinstance(p, str) else p[0] for p in providers]
        self.tensor = None
        self.instances.append(self)

    def disable_fallback(self) -> None:
        """
        Match the runtime session API.
        """

    def get_providers(self) -> list[str]:
        """
        Report the providers selected by the worker.
        """
        return self.providers

    @staticmethod
    def get_inputs() -> list[Any]:
        """
        Describe one static batch of normalized RGB pixels.
        """
        return [SimpleNamespace(name="input", shape=[1, 3, 32, 32], type="tensor(float)")]

    def get_outputs(self) -> list[Any]:
        """
        Describe two chroma planes or an unsupported full-image export.
        """
        channels = 3 if self.bad_schema else 2
        return [SimpleNamespace(name="output", shape=[1, channels, 32, 32],
                                type="tensor(float)")]

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Emit modest chroma values, optionally with invalid inference data.
        """
        assert names == ["output"]
        self.tensor = feed["input"].copy()
        output = np.empty((1, 2, 32, 32), dtype=np.float32)
        output[:, 0] = 10
        output[:, 1] = -5
        if self.spatial_chroma:
            output[:, 0] = np.linspace(-8, 8, 32, dtype=np.float32)[None, None, :]
            output[:, 1] = 0
        if self.bad_output:
            output[0, 0, 0, 0] = np.nan
        return [output]


@pytest.fixture
def setup_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[Any, ...]]:
    """
    Validate folder metadata and a portable JSON pipeline before executing the worker.
    """
    FakeSession.instances = []
    FakeSession.bad_output = FakeSession.bad_schema = False
    FakeSession.spatial_chroma = False
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)
    monkeypatch.setattr("onnxruntime.get_available_providers",
                        lambda: ["CPUExecutionProvider", "CoreMLExecutionProvider"])
    model = tmp_path / "model.onnx"
    model.write_bytes(b"mocked ONNX boundary")
    params = load_folder(FOLDER)
    config = json.loads(params["config"])
    config["model_path"] = str(model)
    config["device"]["backend"] = "cpu"
    params["config"] = json.dumps(config)
    item = node("software", "custom", **params)
    pipeline = default_pipeline()
    pipeline["software"] = [pipeline["software"][0], item, node("software", "output")]
    document = validate_pipeline(json.loads(json.dumps(pipeline)))
    processor = CustomProcessor(apple_available=True, nvidia_available=False)
    try:
        yield processor, document["software"][1], config
    finally:
        processor.close()


def test_preserves_lightness_and_native_geometry(setup_node: tuple[Any, ...]) -> None:
    """
    Add color while retaining source lightness and full-resolution image structure.
    """
    processor, item, _ = setup_node
    gray = np.linspace(64, 192, 11 * 17, dtype=np.float32).reshape(11, 17)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    before = image.copy()
    result = processor.apply(image, item)
    assert result.shape == image.shape and result.dtype == np.float32
    assert np.array_equal(image, before)
    normalized = np.asarray(image / np.float32(255), dtype=np.float32)
    original_lab = np.asarray(cv2.cvtColor(normalized, cv2.COLOR_BGR2Lab))
    result_lab = cv2.cvtColor(result / 255, cv2.COLOR_BGR2Lab)
    np.testing.assert_allclose(result_lab[:, :, 0], original_lab[:, :, 0], atol=0.25)
    np.testing.assert_allclose(result_lab[:, :, 1], 10, atol=0.3)
    np.testing.assert_allclose(result_lab[:, :, 2], -5, atol=0.3)
    tensor = FakeSession.instances[0].tensor
    assert tensor.shape == (1, 3, 32, 32) and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous and 0 <= tensor.min() <= tensor.max() <= 1
    np.testing.assert_allclose(tensor[:, 0], tensor[:, 1], atol=0.0001)
    np.testing.assert_allclose(tensor[:, 1], tensor[:, 2], atol=0.0001)


def test_input_controls_strength_and_session_reuse(setup_node: tuple[Any, ...]) -> None:
    """
    Change model input polarity/orientation while restoring original output geometry.
    """
    processor, item, config = setup_node
    image = np.full((9, 15, 3), 80, dtype=np.float32)
    processor.apply(image, item)
    initial = FakeSession.instances[0].tensor.copy()
    config.update(invert_input=True, input_rotation="90", strength=0.5)
    item["params"]["config"] = json.dumps(config)
    result = processor.apply(image, item)
    assert result.shape == image.shape
    assert len(FakeSession.instances) == 1
    assert FakeSession.instances[0].tensor.mean() > initial.mean()
    lab = cv2.cvtColor(result / 255, cv2.COLOR_BGR2Lab)
    np.testing.assert_allclose(lab[:, :, 1], 5, atol=0.3)
    np.testing.assert_allclose(lab[:, :, 2], -2.5, atol=0.3)
    config["strength"] = 0
    config["model_path"] = "/nonexistent/model.onnx"
    item["params"]["config"] = json.dumps(config)
    assert np.array_equal(processor.apply(image, item), image)


def test_device_model_changes_and_private_sessions(setup_node: tuple[Any, ...]) -> None:
    """
    Rebuild for effective device/model changes and keep each node's session private.
    """
    processor, item, config = setup_node
    image = np.full((8, 8, 3), 100, dtype=np.float32)
    processor.apply(image, item)
    config["device"]["backend"] = "coreml"
    item["params"]["config"] = json.dumps(config)
    processor.apply(image, item)
    assert "CoreMLExecutionProvider" in FakeSession.instances[-1].providers
    another = node("software", "custom", **item["params"])
    processor.apply(image, another)
    assert len(FakeSession.instances) == 3
    model = Path(config["model_path"]).with_name("another.onnx")
    model.write_bytes(b"another mocked boundary")
    config["model_path"] = str(model)
    item["params"]["config"] = json.dumps(config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 4


def test_rotated_chroma_returns_to_source_coordinates(setup_node: tuple[Any, ...]) -> None:
    """
    A model horizontal chroma ramp becomes a downward ramp after undoing CCW input rotation.
    """
    processor, item, config = setup_node
    FakeSession.spatial_chroma = True
    config["input_rotation"] = "90"
    item["params"]["config"] = json.dumps(config)
    image = np.full((9, 15, 3), 128, dtype=np.float32)
    result = processor.apply(image, item)
    lab = cv2.cvtColor(result / 255, cv2.COLOR_BGR2Lab)
    assert lab[0, :, 1].mean() < -6
    assert lab[-1, :, 1].mean() > 6
    assert lab[:, :, 1].std(axis=1).max() < 0.3


def test_pending_default_download_passes_through(
    setup_node: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep input visible while the checksum-pinned default model downloads.
    """
    processor, item, config = setup_node
    config["model_path"] = ""
    item["params"]["config"] = json.dumps(config)

    def pending(_self: Any, source: Any) -> None:
        """
        Assert the package uses its published, version-pinned download descriptor.
        """
        assert source == config["model"]
        assert source["max_bytes"] == 150_000_000

    monkeypatch.setattr("topdon_duo.custom_nodes.models.ModelResolver.resolve", pending)
    image = np.full((8, 8, 3), 100, dtype=np.float32)
    assert np.array_equal(processor.apply(image, item), image)
    assert not FakeSession.instances


@pytest.mark.parametrize("kind, message", [
    ("schema", "Use a DDColor export"), ("output", "invalid chroma"),
    ("missing", "does not exist"), ("strength", "range/value: strength"),
    ("rotation", "choice options/value: input_rotation"),
])
def test_invalid_models_and_settings_report_errors(
    setup_node: tuple[Any, ...], kind: str, message: str,
) -> None:
    """
    Reject unsupported models and malformed inference/configuration through worker errors.
    """
    processor, item, config = setup_node
    FakeSession.bad_schema = kind == "schema"
    FakeSession.bad_output = kind == "output"
    if kind == "missing":
        config["model_path"] = "/nonexistent/model.onnx"
    if kind == "strength":
        config["strength"] = -0.5
    if kind == "rotation":
        config["input_rotation"] = "45"
    item["params"]["config"] = json.dumps(config)
    with pytest.raises(ValueError, match=message):
        processor.apply(np.zeros((8, 8, 3), dtype=np.float32), item)
