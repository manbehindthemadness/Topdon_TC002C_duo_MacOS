"""
Keep humidity in permanent calibration controls and invert only display color values.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.pipeline import fake_hardware
from support.qt_process import run_popup
from test_render import frame_with_preview

from topdon_duo.pipeline import (
    HARDWARE_NODES,
    default_pipeline,
    migrate_pipeline,
    node,
    validate_pipeline,
)
from topdon_duo.pipeline_hardware import PIPELINE_FIELDS, PipelineHardware
from topdon_duo.pipeline_processing import PipelineProcessor
from topdon_duo.processing.images import invert_colors
from topdon_duo.render import ThermalRenderer
from topdon_duo.settings_preferences import load_settings, save_settings


def legacy_humidity(value: float = 45.5, bypass: bool = False) -> dict[str, Any]:
    """
    Represent an old saved node without reintroducing it into the current catalog.
    """
    return {"id": "old-humidity", "type": "humidity", "params": {"value": value},
            "bypass": bypass, "expanded": True}


@pytest.mark.parametrize("bypass", [False, True])
def test_old_humidity_migrates_to_local_calibration_preferences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass: bool,
) -> None:
    """
    Preserve active saved values while removing legacy nodes from usable pipelines.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = default_pipeline()
    document["hardware"].append(legacy_humidity(bypass=bypass))
    save_settings({"pipeline": document})
    loaded = load_settings()
    assert loaded["hardware"].get("humidity") == (None if bypass else 45.5)
    assert loaded["pipeline"]["hardware"] == []
    assert "humidity" not in HARDWARE_NODES and "humidity" not in PIPELINE_FIELDS
    assert validate_pipeline(document)["hardware"] == []
    assert len(document["hardware"]) == 1  # Import validation must not mutate the source.


def test_explicit_humidity_preference_takes_precedence_over_old_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep the user's newer standalone preference when a stale node remains in saved data.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = default_pipeline()
    document["hardware"].append(legacy_humidity())
    save_settings({"pipeline": document, "hardware": {"humidity": 60}})
    assert load_settings()["hardware"]["humidity"] == 60
    assert migrate_pipeline({"hardware": {"humidity": 60}})["hardware"] == []


@pytest.mark.parametrize("value", [10.123, -1, 101, True, float("nan")])
def test_removed_humidity_nodes_are_validated_before_omission(value: Any) -> None:
    """
    Reject malformed legacy calibration values instead of silently trusting old documents.
    """
    document = default_pipeline()
    document["hardware"].append(legacy_humidity(value))
    with pytest.raises((ValueError, TypeError)):
        validate_pipeline(document)


def test_pipeline_edits_do_not_write_or_restore_standalone_humidity() -> None:
    """
    Preserve calibration values through source switches and pipeline control removal.
    """
    hardware = fake_hardware()
    hardware.state.return_value["humidity"].update(value=45.5, enabled=True)
    controller = PipelineHardware(hardware)
    document = default_pipeline()
    document["hardware"].append(node("hardware", "brightness", value=60))
    controller.apply(document)
    document["software"][0]["params"]["source"] = "raw"
    document["hardware"] = []
    controller.apply(document)
    assert hardware.state.return_value["humidity"] == {
        "value": 45.5, "enabled": True, "available": True,
    }
    assert all(call.args[0] != "humidity" for call in hardware.set.call_args_list)


def test_invert_replaces_color_values_and_preserves_measurements() -> None:
    """
    Invert each channel at its existing pixel without changing native temperature planes.
    """
    frame, _ = frame_with_preview()
    document = default_pipeline()
    renderer = ThermalRenderer(scale=1, rotation=90)
    renderer.set_pipeline(document)
    try:
        original = renderer.render_detailed(frame)
        inversion = node("software", "invert")
        document["software"].insert(-1, inversion)
        renderer.set_pipeline(document)
        inverted = renderer.render_detailed(frame)
        np.testing.assert_array_equal(inverted.image, 255 - original.image)
        np.testing.assert_array_equal(inverted.temperatures_celsius, original.temperatures_celsius)
        np.testing.assert_array_equal(inverted.raw_counts, original.raw_counts)
        assert inverted.stats == original.stats
        inversion["bypass"] = True
        renderer.set_pipeline(document)
        np.testing.assert_array_equal(renderer.render_detailed(frame).image, original.image)
    finally:
        renderer.pipeline_processor.close()


def test_two_invert_nodes_cancel_and_saved_documents_keep_them() -> None:
    """
    A parameter-free inversion can repeat, round-trip validation and cancel itself.
    """
    document = default_pipeline()
    frame, _ = frame_with_preview()
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        original, _ = processor.process(frame, None, document, scale=1)
        document["software"][-1:-1] = [node("software", "invert"), node("software", "invert")]
        assert validate_pipeline(deepcopy(document)) == document
        result, _ = processor.process(frame, None, document, scale=1)
        np.testing.assert_array_equal(result, original)
    finally:
        processor.close()


def test_invert_preserves_fractional_color_values_and_input_ownership() -> None:
    """
    Complement all three channels without byte overflow or premature quantization.
    """
    image = np.array([[[0, 128, 255], [10.25, 20.5, 30.75]]], np.float32)
    snapshot = image.copy()
    expected = np.array([[[255, 127, 0], [244.75, 234.5, 224.25]]], np.float32)
    np.testing.assert_array_equal(invert_colors(image), expected)
    np.testing.assert_array_equal(image, snapshot)


def test_humidity_calibration_row_and_invert_node_in_offscreen_ui(tmp_path: Path) -> None:
    """
    Verify calibration placement, percent units, edit messages, locks and node insertion.
    """
    script = '''
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import default_pipeline, HARDWARE_NODES
from topdon_duo.view_settings import VIEW_DEFAULTS
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
humidity = window.hardware_rows["humidity"]
layout = humidity.parentWidget().layout()
assert layout.indexOf(humidity) == layout.indexOf(window.hardware_rows["transmission"]) + 1
state = {**VIEW_DEFAULTS, "pipeline": default_pipeline(), "image_source": "raw",
         "actual_image_source": "raw", "hardware": {
             "humidity": {"value": 45.5, "available": True}}}
window.update_state(state)
assert humidity.input.isEnabled() and humidity.input.value() == 45.5
assert humidity.input.suffix() == " %"
humidity.input.setValue(50)
humidity._emit()
assert messages[-1] == {"action": "hardware", "name": "humidity", "value": 50, "enabled": True}
assert "humidity" not in HARDWARE_NODES
editor = window.pipeline_editor
editor.insert("software", "invert", 1)
assert editor.document["software"][1]["type"] == "invert"
assert editor.document["software"][1]["params"] == {}
window.update_state({**state, "settings_locked": True})
assert not humidity.input.isEnabled() and editor.locked
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
