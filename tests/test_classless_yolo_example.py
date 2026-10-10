"""
Verify the example package with fake ONNX outputs and synthetic pipeline images.
"""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.pipeline import raw_pipeline

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.runtime import CustomPackage, CustomProcessor
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_ui.transfer.documents import read_pipelines, write_pipelines

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "examples/custom_nodes/Classless YOLO"


@pytest.fixture
def package() -> Iterator[CustomPackage]:
    """
    Load the example under its actual temporary package namespace.
    """
    package = CustomPackage(load_folder(FOLDER)["package"])
    try:
        yield package
    finally:
        package.close()


def detector_module(package: CustomPackage) -> ModuleType:
    """
    Access helpers from the loaded package without relying on filesystem import names.
    """
    return sys.modules[package.name + ".detector"]


def model_outputs() -> list[np.ndarray]:
    """
    Provide independently specified boxes, classes, scores and valid count.
    """
    return [
        np.array([[[0.25, 0.25, 0.75, 0.75], [0, 0, 1, 1], [0, 0, 0.5, 0.5]]]),
        np.array([[14, 42, 99]]),
        np.array([[0.9, 0.1, 0.5]]),
        np.array([3]),
    ]


def test_example_full_inference_preprocessing_overlay_and_cached_config(
    package: CustomPackage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Preserve geometry and input ownership; reuse the session when overlay settings change.
    """
    model = tmp_path / "model.onnx"
    model.touch()
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="image", shape=[1, 3, 192, 192], type="tensor(float)")
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name=name) for name in ("boxes", "classes", "scores", "count")
    ]
    session.run.return_value = model_outputs()
    factory = Mock(return_value=session)
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr("onnxruntime.InferenceSession", factory)
    image = np.full((80, 160, 3), [10, 20, 30], dtype=np.float32)
    config: dict[str, Any] = {"model_path": str(model), "show_scores": False}
    result = package.process(image, config)
    assert result.shape == image.shape
    assert np.array_equal(image[20, 40], [10, 20, 30])
    assert np.array_equal(result[20, 40], [0, 255, 0])
    assert np.array_equal(result[0, 159], [10, 20, 30])
    tensor = session.run.call_args.args[1]["image"]
    assert tensor.shape == (1, 3, 192, 192) and tensor.dtype == np.float32
    assert np.array_equal(tensor[0, :, 100, 100], [30, 20, 10])
    assert tensor.flags.c_contiguous
    # OpenCV 5 requires byte pixels for putText; exercise confidence labels too.
    config.update(score_threshold=0.95, show_scores=True)
    assert np.array_equal(package.process(image, config), image)
    config["score_threshold"] = 0.2
    assert package.process(image, config).shape == image.shape
    factory.assert_called_once()
    replacement = tmp_path / "replacement.onnx"
    replacement.touch()
    config["model_path"] = str(replacement)
    package.process(image, config)
    assert factory.call_count == 2
    downloader = Mock(return_value=replacement)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr("topdon_duo.custom_nodes.models.ModelResolver.resolve", downloader)
    config["model_path"] = ""
    assert package.process(image, config).shape == image.shape
    assert factory.call_count == 2
    assert downloader.call_args.args[0]["archive_member"] == "saved_model/model_float32.onnx"


def test_decode_ignores_class_labels_thresholds_orders_and_limits(package: CustomPackage) -> None:
    """
    Treat all detections as objects regardless of their class tensor values.
    """
    helper = detector_module(package)
    detections = helper.decode_outputs(model_outputs(), 0.2, 1)
    assert detections == [(0.25, 0.25, 0.75, 0.75, 0.9)]
    assert helper.decode_outputs(model_outputs(), 1, 20) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("score_threshold", True),
        ("score_threshold", float("nan")),
        ("score_threshold", 1.1),
        ("max_detections", 1.5),
        ("max_detections", False),
        ("line_thickness", 0),
        ("providers", []),
        ("providers", "CPUExecutionProvider"),
        ("show_scores", 1),
    ],
)
def test_invalid_configuration_is_reported(
    package: CustomPackage, tmp_path: Path, field: str, value: object
) -> None:
    """
    Fail before model initialization for invalid user settings.
    """
    model = tmp_path / "model.onnx"
    model.touch()
    with pytest.raises(ValueError, match=field):
        package.process(np.zeros((2, 2, 3)), {"model_path": str(model), field: value})


def test_missing_model_and_provider_have_clear_errors(
    package: CustomPackage, tmp_path: Path
) -> None:
    """
    Keep invalid local overrides and unavailable providers explicit.
    """
    image = np.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="absolute"):
        package.process(image, {"model_path": "model.onnx"})
    with pytest.raises(ValueError, match="Model not found"):
        package.process(image, {"model_path": str(tmp_path / "absent.onnx")})
    model = tmp_path / "model.onnx"
    model.touch()
    with pytest.raises(ValueError, match="providers unavailable"):
        package.process(image, {"model_path": str(model), "providers": ["NoSuchExecutionProvider"]})


@pytest.mark.parametrize("bad_input", [True, False])
def test_incompatible_model_metadata_is_rejected(
    package: CustomPackage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_input: bool
) -> None:
    """
    Refuse input/layout mismatches and the newer two-output export before execution.
    """
    model = tmp_path / "model.onnx"
    model.touch()
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(
            name="image",
            shape=[1, 192, 192, 3] if bad_input else [1, 3, 192, 192],
            type="tensor(float)",
        )
    ]
    session.get_outputs.return_value = [SimpleNamespace(name=name) for name in ("scores", "boxes")]
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *args, **kwargs: session)
    with pytest.raises(ValueError, match="NCHW" if bad_input else "four-output"):
        package.process(np.zeros((2, 2, 3)), {"model_path": str(model)})
    session.run.assert_not_called()


@pytest.mark.parametrize("mutation", ["count", "shape", "nonfinite", "confidence", "export"])
def test_malformed_model_outputs_are_rejected(package: CustomPackage, mutation: str) -> None:
    """
    Reject incompatible export variants and invalid tensor data.
    """
    outputs = model_outputs()
    if mutation == "count":
        outputs[3] = np.array([4])
    elif mutation == "shape":
        outputs[2] = np.zeros((3,))
    elif mutation == "nonfinite":
        outputs[0][0, 0, 0] = np.nan
    elif mutation == "confidence":
        outputs[2][0, 0] = 2
    else:
        outputs = outputs[:2]
    with pytest.raises(ValueError):
        detector_module(package).decode_outputs(outputs, 0.2, 10)


def test_example_exports_without_model_weights_and_loads_with_display_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keep the download manifest portable and pass frames through during installation.
    """
    params = load_folder(FOLDER)
    assert params["name"] == "Classless YOLO"
    item = node("software", "custom", **params)
    document = validate_pipeline(raw_pipeline(item))
    path = tmp_path / "example.pipeline.json"
    write_pipelines(path, {"Example": document}, single=False)
    imported = read_pipelines(path)[0].document
    assert imported == document
    assert all(name.endswith((".py", ".json")) for name in json.loads(params["package"]))
    processor = CustomProcessor()
    requests = Mock(return_value=False)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr("topdon_duo.custom_nodes.models.MODEL_DOWNLOADS.request", requests)
    try:
        image = np.zeros((2, 2, 3))
        assert np.array_equal(processor.apply(image, imported["software"][2]), image)
        assert requests.call_args.kwargs["spec"]["name"] == "Classless YOLO"
    finally:
        processor.close()
