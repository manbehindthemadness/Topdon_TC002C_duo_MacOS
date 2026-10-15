"""
App display defaults replace inactive pipeline nodes without changing restoration snapshots.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from support.pipeline import fake_hardware
from support.qt_process import run_popup
from test_hardware_controls import Device

from based_thermals.camera_backends.contracts import CameraProfile, ControlSpec
from based_thermals.camera_backends.controls import DeviceControls
from based_thermals.camera_backends.duo import hardware_controls as duo_controls
from based_thermals.camera_backends.errors import CameraError
from based_thermals.hardware_controls import HARDWARE_CONTROLS, HardwareControls
from based_thermals.pipeline import default_pipeline, node
from based_thermals.pipeline_hardware import PIPELINE_FIELDS, PipelineHardware


@pytest.fixture
def duo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[HardwareControls, Device]:
    """
    Connect to an independent fake camera that already has colored, enhanced preview settings.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(duo_controls.time, "sleep", lambda _seconds: None)
    device = Device()
    device.blocks[2, 5][5] = 22
    device.blocks[2, 5][7] = 40
    device.blocks[2, 5][-1] = 137
    hardware = HardwareControls(Mock(device=device))
    hardware.load()
    return hardware, device


@pytest.mark.parametrize("action", ["new", "clear", "bypass", "delete", "preset", "raw"])
def test_inactive_nodes_reset_device_bytes_and_report_defaults(
    duo: tuple[HardwareControls, Device], action: str,
) -> None:
    """
    Every removal path neutralizes display controls, preserves calibration and restores on exit.
    """
    hw, device = duo
    snapshot = deepcopy(hw.original)
    calibration = bytes(device.blocks[3, 1])
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"] = [node("hardware", "camera_colors", palette=12),
                            node("hardware", "contrast", value=75),
                            node("hardware", "detail", amount=60)]
    controller.apply(document)
    candidate = deepcopy(document)
    if action in ("new", "preset"):
        candidate = default_pipeline()
    elif action in ("clear", "delete"):
        candidate["hardware"] = []
    elif action == "bypass":
        for item in candidate["hardware"]:
            item["bypass"] = True
    else:
        candidate["software"][0]["params"]["source"] = "raw"
    controller.apply(candidate)
    for name in PIPELINE_FIELDS:
        spec = HARDWARE_CONTROLS[name]
        assert spec.decode(device.blocks[spec.selector, spec.command]) == hw.specs[name].default
        assert hw.state()[name]["value"] == hw.specs[name].default
        assert not hw.state()[name]["enabled"]
    assert bytes(device.blocks[3, 1]) == calibration and hw.original == snapshot
    assert device.blocks[2, 5][-1] == 137
    writes = len(device.writes)
    controller.apply(candidate)
    assert len(device.writes) == writes
    hw.restore()
    assert {key: bytes(value) for key, value in device.blocks.items()} == snapshot


def test_empty_stack_and_external_edit_reapply_defaults_without_cache_invalidation(
    duo: tuple[HardwareControls, Device],
) -> None:
    """
    Initial empty stacks and same-document refreshes cannot retain an external palette override.
    """
    hw, device = duo
    controller = PipelineHardware(hw)
    document = default_pipeline()
    controller.apply(document)
    assert device.blocks[2, 5][5] == 1
    hw.set("palette", 12, True)
    controller.apply(document)
    assert device.blocks[2, 5][5] == 1
    assert not hw.state()["palette"]["enabled"]


def test_shared_block_default_retains_active_noise_and_reenable_saved_palette(
    duo: tuple[HardwareControls, Device],
) -> None:
    """
    Resetting a palette leaves neighboring active fields intact and retains saved node values.
    """
    hw, device = duo
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"] = [node("hardware", "camera_colors", palette=12),
                            node("hardware", "noise", noise_mode=2, noise_spatial=65)]
    controller.apply(document)
    document["hardware"][0]["bypass"] = True
    controller.apply(document)
    assert device.blocks[2, 5][5] == 1 and device.blocks[2, 5][3] == 65
    assert hw.state()["noise_spatial"]["enabled"]
    document["hardware"][0]["bypass"] = False
    controller.apply(document)
    assert device.blocks[2, 5][5] == 12 and device.blocks[2, 5][3] == 65


def device_controls(values: dict[str, Any], write: Any) -> DeviceControls:
    """
    Supply backend-specific defaults that differ from both Duo and the connection baseline.
    """
    specs = {"palette": ControlSpec("Palette", "choice", options=((1, "White"), (12, "Color")),
                                    default=1),
             "brightness": ControlSpec("Brightness", default=45),
             "emissivity": ControlSpec("Emissivity", minimum=0, maximum=1, step=.01,
                                        effect="measurement", default=1)}
    hw = DeviceControls(CameraProfile("test", "Test", (256, 192), 25), specs,
                        lambda name, _spot: values[name], write)
    hw.load()
    return hw


def test_backend_defaults_and_failed_clear_restore_previous_effective_state() -> None:
    """
    Clear transactions roll back defaults and active overrides, preserving the original snapshot.
    """
    values = {"palette": 12, "brightness": 60, "emissivity": .95}
    fail = False

    def write(name: str, value: Any, _spot: str | None) -> None:
        """
        Reject one palette reset after another default has already been written.
        """
        nonlocal fail
        if fail and name == "brightness":
            fail = False
            raise CameraError("reset failed")
        values[name] = value

    hw = device_controls(values, write)
    original = hw.original.copy()
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"] = [node("hardware", "device_control", control="palette", value=12),
                            node("hardware", "device_control", control="brightness", value=80)]
    controller.apply(document)
    previous = hw.state()
    candidate = default_pipeline()
    fail = True
    with pytest.raises(CameraError, match="reset failed"):
        controller.apply(candidate)
    assert hw.state() == previous and controller.document == document
    controller.apply(candidate)
    assert values == {"palette": 1, "brightness": 45, "emissivity": .95}
    assert not hw.enabled and hw.preview_active
    with pytest.raises(CameraError, match="Restore owned"):
        hw.invalidate_device()
    assert hw.original == original
    hw.restore()
    assert values == original and not hw.preview_active


def test_defaults_apply_after_temporarily_unavailable_controls_become_ready() -> None:
    """
    An empty document accepted without SDK controls must still reset a later loaded baseline.
    """
    hw = fake_hardware()
    hw.original = {}
    for setting in hw.state.return_value.values():
        setting["available"] = False
    controller = PipelineHardware(hw)
    document = default_pipeline()
    controller.apply(document)
    hw.load.assert_not_called()
    hw.set_pipeline.assert_not_called()
    hw.original = {"baseline": True}
    for setting in hw.state.return_value.values():
        setting["available"] = True
    hw.state.return_value["palette"]["value"] = 22
    controller.apply(document)
    assert hw.state()["palette"]["value"] == 1
    assert not hw.state()["palette"]["enabled"]


def test_inactive_defaults_remain_owned_after_failed_shutdown_restore() -> None:
    """
    Keep inactive display defaults eligible for another restoration attempt after a failed write.
    """
    values = {"palette": 12, "brightness": 60, "emissivity": .95}
    fail = False

    def write(name: str, value: Any, _spot: str | None) -> None:
        """
        Fail the first attempt to restore the connection-time palette.
        """
        nonlocal fail
        if fail and name == "palette" and value == 12:
            fail = False
            raise CameraError("restore failed")
        values[name] = value

    hw = device_controls(values, write)
    PipelineHardware(hw).apply(default_pipeline())
    assert not hw.enabled
    fail = True
    with pytest.raises(CameraError, match="restoration failed"):
        hw.restore()
    assert hw.preview_active and hw.original["palette"] == 12
    hw.restore()
    assert values == hw.original and not hw.preview_active


def test_duo_failed_first_application_restores_inactive_connection_values(
    duo: tuple[HardwareControls, Device], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A failed initial reset rolls back partial block writes without requiring preset support.
    """
    hw, device = duo
    snapshot = deepcopy(hw.original)
    original_write = hw.write
    fail = True

    def write(key: tuple[int, int], payload: bytes) -> None:
        """
        Reject the contrast write after brightness has been reset successfully.
        """
        nonlocal fail
        if fail and key == (2, 2):
            fail = False
            raise CameraError("reset failed")
        original_write(key, payload)

    monkeypatch.setattr(hw, "write", write)
    controller = PipelineHardware(hw)
    with pytest.raises(CameraError, match="reset failed"):
        controller.apply(default_pipeline())
    assert {key: bytes(value) for key, value in device.blocks.items()} == snapshot
    assert controller.document is None and not hw.enabled
    controller.apply(default_pipeline())
    assert device.blocks[2, 5][5] == 1


def test_editor_new_clear_bypass_and_software_only_clear_reach_hardware(
    duo: tuple[HardwareControls, Device], tmp_path: Path,
) -> None:
    """
    Replay real isolated editor messages through the hardware owner and check each palette.
    """
    result = run_popup('''
import json, sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
from based_thermals.pipeline import node
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
editor.create_new_pipeline()
editor.document["hardware"] = [node("hardware", "camera_colors", palette=12)]
editor.rebuild()
editor.publish()
editor.clear("software")
editor.bypass(editor.document["hardware"][0], True)
saved = json.loads(json.dumps(editor.document))
editor.update_state({"pipeline": saved, "pipeline_serial": editor.edit_serial}, False)
assert editor.document["hardware"][0]["bypass"]
editor.bypass(editor.document["hardware"][0], False)
editor.clear("hardware")
editor.document["hardware"] = [node("hardware", "camera_colors", palette=12)]
editor.rebuild()
editor.publish()
editor.clear()
editor.document["hardware"] = [node("hardware", "camera_colors", palette=12)]
editor.rebuild()
editor.publish()
editor.create_new_pipeline()
Path(sys.argv[1], "messages.json").write_text(json.dumps([
    message for message in messages if message["action"] == "pipeline"]))
window.close()
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    hw, device = duo
    controller = PipelineHardware(hw)
    messages = json.loads((tmp_path / "messages.json").read_text())
    expected = [1, 12, 12, 1, 12, 1, 12, 1, 12, 1]
    assert len(messages) == len(expected)
    for message, palette in zip(messages, expected, strict=True):
        controller.apply(message["document"])
        assert device.blocks[2, 5][5] == palette
