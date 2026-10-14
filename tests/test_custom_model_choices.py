"""
Validate optional model-picker names and discovery hints as portable JSON metadata.
"""

from typing import Any

import pytest

from based_thermals.custom_nodes.configuration import validate_controls


@pytest.mark.parametrize("models", [
    None, [], {}, [{"name": "pose"}], [{"name": "", "sha256": "a" * 64}],
    [{"name": "pose", "sha256": "bad"}], [{"name": "pose", "sha256": None}],
    [{"name": "pose", "filenames": []}], [{"name": "pose", "filenames": [None]}],
    [{"name": "pose", "filenames": ["../pose.onnx"]}],
    [{"name": "pose", "filenames": ["pose.txt"]}],
    [{"name": "pose", "filenames": ["pose.onnx"], "unexpected": True}],
    [{"name": "pose", "sha256": "a" * 64}] * 2,
])
def test_invalid_model_discovery_metadata_is_rejected(models: Any) -> None:
    """
    Reject malformed naming/filter metadata before UI construction or worker import.
    """
    controls = [{"key": "weights", "label": "Weights", "type": "model", "models": models}]
    with pytest.raises(ValueError, match="model choice"):
        validate_controls(controls, {"weights": ""})


def test_optional_model_discovery_hints_preserve_legacy_controls() -> None:
    """
    Retain model controls without hints and allow readable names with either matching method.
    """
    controls: list[dict[str, Any]] = [{"key": "weights", "label": "Weights", "type": "model"}]
    assert validate_controls(controls, {"weights": ""}) == controls
    controls[0]["models"] = [
        {"name": "Detector", "sha256": "a" * 64},
        {"name": "Pose", "filenames": ["pose.onnx"]},
    ]
    assert validate_controls(controls, {"weights": "/missing/saved.onnx"}) == controls
