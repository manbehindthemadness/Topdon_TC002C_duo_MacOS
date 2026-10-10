"""
Custom device schemas, portable fallback and worker-only provider configuration.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.frames import make_frame
from support.pipeline import raw_pipeline

from topdon_duo import apple_acceleration, nvidia_acceleration
from topdon_duo.custom_nodes import devices
from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.configuration import configuration
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_model.catalog import SOFTWARE_NODES
from topdon_duo.processing.branch import BranchProcessor


def device_document(value: object) -> dict[str, Any]:
    """
    Declare a device control without requiring a special configuration key name.
    """
    document = {"defaults": {"inference": value}, "controls": [
        {"key": "inference", "label": "Inference device", "type": "device"},
    ]}
    return document


@pytest.mark.parametrize("value", [
    None, [], "cpu", {}, {"backend": "cpu"},
    {"backend": "cpu", "apple_compute": "CPUAndGPU", "extra": True},
    {"backend": "automatic", "apple_compute": "CPUAndGPU"},
    {"backend": [], "apple_compute": "CPUAndGPU"},
    {"backend": "cpu", "apple_compute": None},
    {"backend": "cpu", "apple_compute": "GPUOnly"},
])
def test_invalid_device_values_are_rejected_before_execution(value: object) -> None:
    """
    Refuse malformed external values before creating widgets or importing packages.
    """
    with pytest.raises(ValueError, match="device|compute"):
        configuration(json.dumps(device_document(value)))


@pytest.mark.parametrize("requested,apple,nvidia,expected", [
    ("cpu", True, True, "cpu"),
    ("coreml", True, True, "coreml"),
    ("coreml", False, True, "cuda"),
    ("coreml", False, False, "cpu"),
    ("cuda", True, True, "cuda"),
    ("cuda", True, False, "coreml"),
    ("cuda", False, False, "cpu"),
])
def test_custom_devices_match_builtin_fallback_without_mutating_saved_settings(
    requested: str, apple: bool, nvidia: bool, expected: str,
) -> None:
    """
    Honor explicit CPU and retain saved Apple units through another GPU or CPU fallback.
    """
    saved = {"backend": requested, "apple_compute": "ALL"}
    effective = devices.resolve_device(saved, apple_available=apple, nvidia_available=nvidia)
    assert effective == {"backend": expected, "apple_compute": "ALL" if expected == "coreml"
                         else "CPUOnly"}
    assert saved == {"backend": requested, "apple_compute": "ALL"}


def test_device_validation_and_cpu_requests_never_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep hardware probes out of schema validation, plain CPU execution and UI imports.
    """
    apple = Mock(side_effect=AssertionError("unexpected Apple probe"))
    nvidia = Mock(side_effect=AssertionError("unexpected NVIDIA probe"))
    monkeypatch.setattr(apple_acceleration, "apple_acceleration", apple)
    monkeypatch.setattr(nvidia_acceleration, "nvidia_acceleration", nvidia)
    saved = {"backend": "cpu", "apple_compute": "CPUAndGPU"}
    assert configuration(json.dumps(device_document(saved)))[0]["inference"] == saved
    assert devices.resolve_device(saved)["backend"] == "cpu"
    assert devices.onnx_providers(devices.resolve_device(saved)) == ["CPUExecutionProvider"]
    apple.assert_not_called()
    nvidia.assert_not_called()


def test_standalone_gpu_resolution_reuses_capability_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Standalone CustomProcessor requests use the same bounded capability probes.
    """
    apple, nvidia = Mock(return_value={"available": False}), Mock(return_value={"available": True})
    monkeypatch.setattr(apple_acceleration, "apple_acceleration", apple)
    monkeypatch.setattr(nvidia_acceleration, "nvidia_acceleration", nvidia)
    resolved = devices.resolve_device({"backend": "coreml", "apple_compute": "ALL"})
    assert resolved == {"backend": "cuda", "apple_compute": "CPUOnly"}
    apple.assert_called_once_with()
    nvidia.assert_called_once_with()


def test_provider_options_and_device_choices_match_builtin_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reuse CUDA preloading and expose the same Apple units as the existing AI nodes.
    """
    params = SOFTWARE_NODES["onnx_superresolution"][1]
    assert {key for key, _ in devices.BACKENDS} == {key for key, _ in params["backend"].options}
    assert devices.APPLE_COMPUTE == params["apple_compute"].options
    providers = devices.onnx_providers({"backend": "coreml", "apple_compute": "CPUAndNeuralEngine"})
    assert providers == [("CoreMLExecutionProvider", {
        "ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndNeuralEngine",
        "RequireStaticInputShapes": "0", "EnableOnSubgraphs": "0",
    }), "CPUExecutionProvider"]
    cuda = Mock(return_value=[("CUDAExecutionProvider", {"device_id": "0"}), "CPUExecutionProvider"])
    monkeypatch.setattr(nvidia_acceleration, "cuda_providers", cuda)
    assert devices.onnx_providers({"backend": "cuda", "apple_compute": "ALL"}) == cuda.return_value
    cuda.assert_called_once_with()


def test_device_folder_round_trip_and_worker_configuration_changes(tmp_path: Path) -> None:
    """
    Resolve devices per callback without changing saved data or reimporting cached packages.
    """
    saved = {"backend": "coreml", "apple_compute": "ALL"}
    marker = tmp_path / "imported"
    source = (
        f"CONFIG_JSON = {device_document(saved)!r}\n"
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "def process(image, config):\n"
        " device = config['inference']\n"
        " image[:] = {'cpu': 1, 'coreml': 2, 'cuda': 3}[device['backend']]\n"
        " if device['backend'] == 'coreml':\n"
        "  assert device['apple_compute'] == 'ALL'\n"
        " else:\n"
        "  assert device['apple_compute'] == 'CPUOnly'\n"
        " device['backend'] = 'mutated by callback'\n"
        " return image\n"
    )
    (tmp_path / "__init__.py").write_text(source)
    item = node("software", "custom", **load_folder(tmp_path))
    document = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))
    assert not marker.exists()
    item = document["software"][2]
    original = item["params"]["config"]
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    image = np.zeros((2, 3, 3), np.float32)
    try:
        assert np.all(processor.apply(image, item) == 1)
        package = processor.packages[item["id"]]
        assert marker.exists() and item["params"]["config"] == original
        processor.nvidia_available = True
        assert np.all(processor.apply(image, item) == 3)
        processor.apple_available = True
        assert np.all(processor.apply(image, item) == 2)
        assert processor.packages[item["id"]] is package
        item["params"]["config"] = json.dumps({"inference": {"backend": "cpu", "apple_compute": "ALL"}})
        assert np.all(processor.apply(image, item) == 1)
        assert not image.any()
    finally:
        processor.close()


def test_undeclared_device_keys_keep_legacy_configuration() -> None:
    """
    Plain JSON keys are never interpreted as device controls without an explicit declaration.
    """
    source = "def process(image, config): return image + (10 if config['device'] == 'custom' else 0)"
    item = node("software", "custom", package=json.dumps({"__init__.py": source}),
                config='{"device": "custom"}')
    processor = CustomProcessor(apple_available=False, nvidia_available=False)
    try:
        assert np.all(processor.apply(np.zeros((1, 1, 3)), item) == 10)
    finally:
        processor.close()


def test_custom_branch_devices_reuse_startup_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Use the owning branch's capability results without probing again or touching radiometry.
    """
    probe = Mock(side_effect=AssertionError("branch capabilities must be reused"))
    monkeypatch.setattr(apple_acceleration, "apple_acceleration", probe)
    monkeypatch.setattr(nvidia_acceleration, "nvidia_acceleration", probe)
    saved = {"backend": "cuda", "apple_compute": "ALL"}
    item = node("software", "custom", package=json.dumps({"__init__.py":
        "def process(image, config):\n"
        " assert config['inference'] == {'backend': 'coreml', 'apple_compute': 'ALL'}\n"
        " image[:] = 77\n return image\n",
    }), config=json.dumps({"inference": saved}),
        controls=json.dumps(device_document(saved)["controls"]))
    frame = make_frame()
    processor = BranchProcessor(apple_available=True, nvidia_available=False)
    try:
        output, source = processor.process(frame, None, raw_pipeline(item))
        assert source == "raw" and np.all(output == 77)
        assert frame == make_frame()
        assert json.loads(item["params"]["config"])["inference"] == saved
        probe.assert_not_called()
    finally:
        processor.close()
