"""
Classless YOLO device controls route inference and invalidate sessions appropriately.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.classless import FOLDER, helper
from support.classless import package as package  # noqa: PLC0414 - pytest fixture re-export
from support.qt_process import run_popup

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.runtime import CustomPackage, CustomProcessor
from topdon_duo.pipeline import node


def fake_session(providers: list[Any]) -> Mock:
    """
    Return dense empty detections with the independently specified localizer schema.
    """
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="image", shape=[1, 3, 192, 192], type="tensor(float)"),
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name=name) for name in ("boxes", "classes", "scores", "count")
    ]
    session.get_providers.return_value = [
        value if isinstance(value, str) else value[0] for value in providers
    ]
    session.run.return_value = [
        np.zeros((1, 1, 4), np.float32), np.zeros((1, 1), np.float32),
        np.zeros((1, 1), np.float32), np.zeros((1,), np.float32),
    ]
    return session


def test_yolo_device_sessions_rebuild_for_execution_and_reuse_for_display_edits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Route real package processing through CPU, Apple units, CUDA and portable fallback.
    """
    model = tmp_path / "localizer.onnx"
    model.touch()
    params = load_folder(FOLDER)
    config = json.loads(params["config"])
    assert config["device"] == {"backend": "cpu", "apple_compute": "CPUAndGPU"}
    config["model_path"] = str(model)
    params["config"] = json.dumps(config)
    item = node("software", "custom", **params)
    sessions: list[Mock] = []

    def initialize(_model: str, *, sess_options: object, providers: list[Any]) -> Mock:
        """
        Capture provider configuration without creating native inference sessions.
        """
        assert sess_options is not None
        initialized = fake_session(providers)
        sessions.append(initialized)
        return initialized

    factory = Mock(side_effect=initialize)
    monkeypatch.setattr("onnxruntime.InferenceSession", factory)
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: [
        "CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider",
    ])
    cuda = Mock(return_value=[("CUDAExecutionProvider", {"device_id": "0"}), "CPUExecutionProvider"])
    monkeypatch.setattr("topdon_duo.nvidia_acceleration.cuda_providers", cuda)
    processor = CustomProcessor(apple_available=True, nvidia_available=True)
    image = np.full((12, 20, 3), 25, np.float32)

    def apply() -> None:
        """
        Apply the latest saved configuration while preserving geometry and input ownership.
        """
        item["params"]["config"] = json.dumps(config)
        result = processor.apply(image, item)
        assert np.array_equal(result, image) and result.dtype == np.float32

    def check_sessions(expected: int) -> None:
        """
        Check cumulative session initialization after each configuration transition.
        """
        assert factory.call_count == expected

    try:
        apply()
        assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
        check_sessions(1)
        config["device"]["backend"] = "coreml"
        apply()
        check_sessions(2)
        provider, options = factory.call_args.kwargs["providers"][0]
        assert provider == "CoreMLExecutionProvider" and options["MLComputeUnits"] == "CPUAndGPU"
        config["device"]["apple_compute"] = "ALL"
        apply()
        check_sessions(3)
        assert factory.call_args.kwargs["providers"][0][1]["MLComputeUnits"] == "ALL"
        config.update(score_threshold=0.5, border_color="dynamic", show_scores=True)
        apply()
        check_sessions(3)
        config["device"]["backend"] = "cuda"
        apply()
        check_sessions(4)
        assert factory.call_args.kwargs["providers"] == cuda.return_value
        apply()
        check_sessions(4)
        cuda.assert_called_once_with()
        processor.apple_available = processor.nvidia_available = False
        apply()
        check_sessions(5)
        assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
        assert json.loads(item["params"]["config"])["device"] == {"backend": "cuda", "apple_compute": "ALL"}
        processor.apple_available = True
        apply()
        check_sessions(6)
        assert factory.call_args.kwargs["providers"][0][1]["MLComputeUnits"] == "ALL"
        for session in sessions:
            session.disable_fallback.assert_called_once_with()
    finally:
        processor.close()


@pytest.mark.parametrize("device", [
    None, "cpu", {}, {"backend": "coreml", "apple_compute": "GPUOnly"},
])
def test_yolo_rejects_invalid_device_settings(package: CustomPackage, device: object) -> None:
    """
    Validate advanced JSON device edits before loading a model.
    """
    with pytest.raises(ValueError, match="device|compute"):
        helper(package, "settings").Settings.from_config({"device": device})


def test_yolo_device_overrides_legacy_providers_and_rejects_failed_gpu_session(
    package: CustomPackage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Ignore retired provider fields when a device is supplied and refuse silent GPU fallback.
    """
    model = tmp_path / "localizer.onnx"
    model.touch()
    session = fake_session(["CPUExecutionProvider"])
    factory = Mock(return_value=session)
    monkeypatch.setattr("onnxruntime.InferenceSession", factory)
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: [
        "CPUExecutionProvider", "CoreMLExecutionProvider",
    ])
    config = {"model_path": str(model), "providers": ["NoSuchExecutionProvider"],
              "device": {"backend": "cpu", "apple_compute": "CPUOnly"}}
    assert package.process(np.zeros((2, 3, 3)), config).shape == (2, 3, 3)
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    config["device"] = {"backend": "coreml", "apple_compute": "CPUAndGPU"}
    with pytest.raises(ValueError, match="Selected GPU session failed"):
        package.process(np.zeros((2, 3, 3)), config)
    assert session.run.call_count == 1


def test_real_yolo_editor_exposes_device_and_persists_selection(tmp_path: Path) -> None:
    """
    Load the actual example and exercise its optional control without initializing inference.
    """
    result = run_popup(
        '''
import json
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
from topdon_duo.view_window import ViewWindow

app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.insert("software", "custom", 1)
item = editor.document["software"][1]
controls = editor.custom_widgets[item["id"]]
QFileDialog.getExistingDirectory = lambda *args: str(Path.cwd() / "examples/custom_nodes/Classless YOLO")
controls.open_folder()
editor.update_state({"pipeline_serial": editor.edit_serial,
                     "apple_acceleration": {"available": True}}, False)
picker = controls.fields.fields["device"]
assert picker.backend.currentData() == "cpu"
assert not picker.compute.isEnabled()
picker.backend.setCurrentIndex(picker.backend.findData("coreml"))
assert picker.compute.currentData() == "CPUAndGPU" and picker.compute.isEnabled()
picker.compute.setCurrentIndex(picker.compute.findData("ALL"))
assert json.loads(item["params"]["config"])["device"] == {"backend": "coreml", "apple_compute": "ALL"}
editor.rebuild()
assert editor.custom_widgets[item["id"]].fields.fields["device"].compute.currentData() == "ALL"
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_yolo_coreml_decodes_count_with_extra_batch_axis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Draw a GPU proposal when the count has the reported (1, 1) runtime shape.
    """
    model = tmp_path / "localizer.onnx"
    model.touch()
    params = load_folder(FOLDER)
    config = json.loads(params["config"])
    config.update(model_path=str(model), device={"backend": "coreml", "apple_compute": "CPUAndGPU"})
    params["config"] = json.dumps(config)
    session = fake_session(["CoreMLExecutionProvider", "CPUExecutionProvider"])
    outputs = [
        np.array([[[0.25, 0.25, 0.75, 0.75]]], np.float32),
        np.array([[14]], np.float32), np.array([[0.9]], np.float32),
        np.array([[1]], np.float32),
    ]
    session.run.return_value = outputs
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *args, **kwargs: session)
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: [
        "CPUExecutionProvider", "CoreMLExecutionProvider",
    ])
    processor = CustomProcessor(apple_available=True, nvidia_available=False)
    image = np.zeros((80, 160, 3), np.float32)
    try:
        result = processor.apply(image, node("software", "custom", **params))
        assert result.shape == image.shape and result.dtype == np.float32
        assert np.array_equal(result[20, 40], [0, 255, 0])
        assert not image.any()
        assert outputs[3].shape == (1, 1)
        tensor = session.run.call_args.args[1]["image"]
        assert tensor.shape == (1, 3, 192, 192) and tensor.dtype == np.float32
    finally:
        processor.close()


@pytest.mark.parametrize("count", [
    np.array(1), np.array([]), np.array([1, 1]), np.array([[[1]]]),
    np.array([[float("nan")]]), np.array([[float("inf")]]),
    np.array([[0.5]]), np.array([[-1]]), np.array([[2]]),
])
def test_yolo_rejects_malformed_counts_after_batch_normalization(
    package: CustomPackage, count: np.ndarray,
) -> None:
    """
    Preserve shape, finite, integer and capacity checks for the count tensor.
    """
    outputs = [
        np.array([[[0.25, 0.25, 0.75, 0.75]]]),
        np.array([[14]]), np.array([[0.9]]), count,
    ]
    with pytest.raises(ValueError, match="output shapes|detection count"):
        helper(package, "detector").decode_outputs(outputs, 0.2)
