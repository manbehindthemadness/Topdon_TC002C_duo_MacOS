"""Pipeline behavior: ordered image operations, safe ownership, and portable state."""

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture
from support.pipeline import fake_hardware, raw_pipeline

from topdon_duo.camera import (
    CameraError,
)
from topdon_duo.pipeline import (
    default_pipeline,
    node,
    validate_pipeline,
)
from topdon_duo.pipeline_hardware import PipelineHardware


def test_hardware_reorder_makes_no_usb_writes_and_bypass_restores_only_owned_fields() -> None:
    hw = fake_hardware()
    hw.state.return_value["emissivity"].update(value=0.95, enabled=True)
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"] += [
        node("hardware", "brightness", value=60),
        node("hardware", "contrast", value=70),
    ]
    controller.apply(document)
    hw.reset_mock()
    document["hardware"].reverse()
    controller.apply(document)
    assert not hw.mock_calls
    document["hardware"][0]["bypass"] = True
    controller.apply(document)
    assert hw.set.call_args.args[0] == "contrast" and hw.set.call_args.args[2] is False
    assert hw.state.return_value["emissivity"]["value"] == 0.95
    hw.restore.assert_not_called()


def test_hardware_failure_rolls_back_without_clearing_calibration_fields() -> None:
    hw = fake_hardware()
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"].append(node("hardware", "brightness", value=60))
    controller.apply(document)
    candidate = deepcopy(document)
    candidate["hardware"][0]["params"]["value"] = 80
    original = hw.set.side_effect

    def fail(name: Any, value: Any, enabled: Any) -> None:
        """
        Fail.
        """
        if value == 80:
            raise CameraError("failed")
        original(name, value, enabled)

    hw.set.side_effect = fail
    with pytest.raises(CameraError, match="failed"):
        controller.apply(candidate)
    assert controller.document == document
    assert hw.state.return_value["brightness"]["value"] == 60
    assert not any(
        call.args[0] in ("ambient", "reflected", "emissivity", "distance")
        for call in hw.set.call_args_list
    )


def test_pipeline_fields_preserve_sdk_calibration_bytes_and_restore_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_hardware_controls import Device

    from topdon_duo.hardware_controls import HardwareControls

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    device = Device()
    hw = HardwareControls(Mock(device=device))
    hw.load()
    hw.set("ambient", 22.0, True)
    hw.set("reflected", 24.2, True)
    hw.set("emissivity", 0.95, True)
    calibration = hw.read(3, 1)
    original_preview = {key: hw.read(*key) for key in ((2, 1), (2, 2), (2, 5))}
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"] += [
        node("hardware", "contrast", value=75),
        node("hardware", "detail", amount=40, enabled=True),
        node("hardware", "camera_colors", palette=22),
    ]
    controller.apply(document)
    assert hw.read(3, 1) == calibration
    assert hw.state()["detail_enabled"]["value"] == 1
    for item in document["hardware"]:
        item["bypass"] = True
    controller.apply(document)
    assert hw.read(3, 1) == calibration
    assert {key: hw.read(*key) for key in original_preview} == original_preview


def test_software_pipeline_works_when_sdk_controls_are_unavailable() -> None:
    hw = fake_hardware()
    hw.original = {}
    hw.load.side_effect = CameraError("Unsupported control layout")
    document = raw_pipeline(node("software", "gamma", amount=2.0))
    controller = PipelineHardware(hw)
    controller.apply(document)
    hw.load.assert_not_called()
    assert controller.document == document
    document["software"][0]["params"]["source"] = "preview"
    document["software"] = [document["software"][0], document["software"][-1]]
    document["hardware"].append(node("hardware", "brightness", value=40))
    with pytest.raises(CameraError, match="Unsupported control layout"):
        controller.apply(document)


def test_source_toggle_preserves_unavailable_hardware_without_retrying_sdk() -> None:
    hw = fake_hardware()
    hw.original = {}
    hw.load.side_effect = CameraError("Unsupported control layout")
    controller = PipelineHardware(hw)
    current = default_pipeline()
    current["hardware"].append(node("hardware", "brightness", value=40))
    for source in ("raw", "preview", "raw"):
        candidate = deepcopy(current)
        candidate["software"][0]["params"]["source"] = source
        controller.apply(candidate, previous_document=current)
        assert controller.document == candidate
        hw.load.assert_not_called()
        current = candidate
    candidate = deepcopy(current)
    candidate["hardware"][0]["params"]["value"] = 50
    candidate["software"][0]["params"]["source"] = "preview"
    with pytest.raises(CameraError, match="Unsupported control layout"):
        controller.apply(candidate, previous_document=current)


@pytest.mark.parametrize("bypass", (False, True))
def test_old_transmission_node_moves_to_standalone_preferences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass: Any
) -> None:
    from topdon_duo.pipeline import HARDWARE_NODES
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = default_pipeline()
    document["hardware"].append(
        {
            "id": "old-transmission",
            "type": "transmission",
            "params": {"value": 95},
            "bypass": bypass,
            "expanded": True,
        }
    )
    save_settings({"pipeline": document})
    saved = load_settings()
    assert "transmission" not in HARDWARE_NODES
    assert all(item["type"] != "transmission" for item in saved["pipeline"]["hardware"])
    assert saved["hardware"].get("transmission") == (None if bypass else 95)
    # Importing a pipeline never changes the standalone correction controls.
    assert all(item["type"] != "transmission" for item in validate_pipeline(document)["hardware"])


def test_transmission_is_not_owned_or_restored_by_pipeline_nodes() -> None:
    hw = fake_hardware()
    hw.state.return_value["transmission"].update(value=95, enabled=True)
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"].append(node("hardware", "brightness", value=60))
    controller.apply(document)
    document["hardware"] = []
    controller.apply(document)
    assert hw.state.return_value["transmission"] == {
        "value": 95,
        "enabled": True,
        "available": True,
    }
    assert all(call.args[0] != "transmission" for call in hw.set.call_args_list)


__all__ = ["viewer_fixture"]
