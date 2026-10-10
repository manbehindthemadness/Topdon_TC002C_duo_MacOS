"""
Check embedded declarations, control schemas and portable configuration state.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.pipeline import raw_pipeline

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.configuration import configuration
from topdon_duo.custom_nodes.metadata import embedded_metadata
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import node, validate_pipeline


def document() -> dict[str, Any]:
    """
    Supply a control schema with independent defaults for all supported types.
    """
    return {"defaults": {"gain": 1.5, "count": 2, "enabled": True,
                         "mode": "A", "name": "hello"}, "controls": [
        {"key": "gain", "label": "Gain", "type": "number", "min": 0, "max": 3, "step": 0.1},
        {"key": "count", "label": "Count", "type": "integer", "min": 1, "max": 5, "step": 1},
        {"key": "enabled", "label": "Enabled", "type": "boolean"},
        {"key": "mode", "label": "Mode", "type": "choice", "options": ["A", "B"]},
        {"key": "name", "label": "Name", "type": "text"},
    ]}


def test_embedded_metadata_precedence_safe_loading_and_round_trip(tmp_path: Path) -> None:
    """
    Read literal metadata without side effects and persist independently edited values.
    """
    marker = tmp_path / "executed"
    config = document()
    model = {"url": "https://example.com/weights", "sha256": "a" * 64}
    (tmp_path / "__init__.py").write_text(
        f"CONFIG_JSON = {json.dumps(config)!r}\nMODEL_SOURCE = {model!r}\n"
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "def process(image, config): return image + config['gain']\n"
    )
    (tmp_path / "config.json").write_text('{"gain": 99}')
    params = load_folder(tmp_path)
    assert not marker.exists()
    assert json.loads(params["config"])["gain"] == 1.5
    assert json.loads(params["config"])["model"] == model
    assert len(json.loads(params["controls"])) == 5
    item = node("software", "custom", **params)
    values = json.loads(item["params"]["config"])
    values["gain"] = 2.5
    item["params"]["config"] = json.dumps(values)
    saved = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))
    processor = CustomProcessor()
    try:
        assert np.all(processor.apply(np.zeros((2, 2, 3)), saved["software"][2]) == 2.5)
        assert marker.exists()
    finally:
        processor.close()


@pytest.mark.parametrize("source", [
    "CONFIG_JSON = build_config()", "CONFIG_JSON = '{}'; CONFIG_JSON = '{}'",
    "CONFIG_JSON = '[]'", "MODEL_SOURCE = {'url': 'https://example.com/a'}",
    "MODEL_SOURCE = read_model()", "CONFIG_JSON: str",
])
def test_invalid_embedded_declarations_are_rejected(source: str) -> None:
    """
    Never execute computed declarations or accept unsupported metadata.
    """
    with pytest.raises(ValueError, match="CONFIG_JSON|MODEL_SOURCE"):
        embedded_metadata(source)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bool", "range", "step",
                                      "type", "options", "integer", "extra", "label", "huge"])
def test_invalid_control_schema_and_values_are_rejected(mutation: str) -> None:
    """
    Reject invalid controls before widget creation or pipeline execution.
    """
    data = document()
    controls, defaults = data["controls"], data["defaults"]
    if mutation == "duplicate":
        controls.append(controls[0])
    elif mutation == "missing":
        del defaults["gain"]
    elif mutation == "bool":
        defaults["enabled"] = 1
    elif mutation == "range":
        defaults["gain"] = 4
    elif mutation == "step":
        controls[0]["step"] = 0
    elif mutation == "type":
        controls[0]["type"] = "execute"
    elif mutation == "options":
        controls[3]["options"] = ["A", "A"]
    elif mutation == "integer":
        defaults["count"] = 1.5
    elif mutation == "extra":
        controls[0]["callback"] = "anything"
    elif mutation == "huge":
        defaults["gain"] = 10 ** 400
    else:
        controls[0]["label"] = None
    with pytest.raises(ValueError):
        configuration(json.dumps(data))


def test_plain_and_external_configs_and_legacy_custom_parameters(tmp_path: Path) -> None:
    """
    Existing plain configs work and old saved nodes acquire the controls default.
    """
    assert configuration('{"gain": 2}') == ({"gain": 2}, [])
    (tmp_path / "__init__.py").write_text("def process(image, config): return image")
    (tmp_path / "config.json").write_text(json.dumps(document()))
    params = load_folder(tmp_path)
    assert len(json.loads(params["controls"])) == 5
    old = node("software", "custom")
    del old["params"]["controls"]
    migrated = validate_pipeline(raw_pipeline(old))
    assert migrated["software"][2]["params"]["controls"] == "[]"


@pytest.mark.parametrize("kind,value", [("color", "red"), ("color", "#12345"),
                                        ("color", None), ("model", 123)])
def test_invalid_picker_values_rejected(kind: str, value: Any) -> None:
    """
    Refuse invalid picker state before creating native widgets.
    """
    data = {"defaults": {"value": value}, "controls": [
        {"key": "value", "label": "Value", "type": kind},
    ]}
    with pytest.raises(ValueError):
        configuration(json.dumps(data))
