import json
import struct
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from test_render import frame_with_preview

from topdon_duo import hardware_controls as module
from topdon_duo.camera import IMAGE_OFFSET, CameraError
from topdon_duo.hardware_controls import (
    BLOCK_LENGTHS,
    HardwareControls,
    validate_fixed_range_bounds,
)
from topdon_duo.render import ThermalRenderer


def test_processing_preset_wire_modes_and_bank_getter(controls):
    device = Mock()
    packets = []
    bank = 1

    def transfer(kind, request, value, index, data, timeout):
        nonlocal bank
        assert value == 0 and index == 0xA00
        if kind == 0x41:
            packet = bytes(data)
            packets.append(packet)
            if packet[4] == 0:
                bank = {1: 1, 2: 2, 0: 3}[int.from_bytes(packet[5:7], "big")]
                body = b"\x36\x23\x03\x01"
            else:
                body = b"\x36\x23\x03" + bank.to_bytes(2, "big")
            device.reply = bytes((0xF0, len(body))) + body + bytes((sum(body) & 255, 0xFF))
            return len(data)
        if request == 0x85:
            return len(device.reply).to_bytes(2, "little")
        return device.reply

    device.ctrl_transfer.side_effect = transfer
    controls.camera.device = device
    controls._fixed_range_baseline = Mock(return_value=(1000, 2800))
    original = bytearray(controls.original[2, 5])
    original[23] = 1
    controls.original[2, 5] = bytes(original)
    controls.set_processing_preset("shadow")
    assert controls.preview_active
    controls.set_processing_preset("soft")
    controls.restore_processing_preset()
    setters = [p for p in packets if p[4] == 0]
    assert [int.from_bytes(p[5:7], "big") for p in setters] == [2, 0, 1]
    assert all(p[-2] == sum(p[2:-2]) & 255 for p in packets)
    assert controls.processing_preset == "balanced" and bank == 1
    assert not controls._processing_preset_owned


def test_preset_failure_restores_balanced_and_keeps_cleanup_on_restore_failure(controls):
    controls._fixed_range_baseline = Mock()
    original = bytearray(controls.original[2, 5])
    original[23] = 1
    controls.original[2, 5] = bytes(original)
    controls._processing_command = Mock(return_value=1)
    controls._apply_processing_preset = Mock(side_effect=[CameraError("failed"), None])
    with pytest.raises(CameraError, match="failed"):
        controls.set_processing_preset("shadow")
    assert controls.processing_preset == "balanced"
    assert not controls._processing_preset_owned
    controls._processing_preset_owned = True
    controls._apply_processing_preset = Mock(side_effect=CameraError("restore failed"))
    with pytest.raises(CameraError, match="restore failed"):
        controls.restore_processing_preset()
    assert controls._processing_preset_owned


def test_preset_survives_camera_adjustments_and_is_restored_on_exit(controls):
    controls._processing_preset_owned = True
    controls.processing_preset = "shadow"
    controls._apply_processing_preset = Mock()
    controls.set("contrast", 70, True)
    controls._apply_processing_preset.assert_called_once_with("shadow")
    with pytest.raises(CameraError, match="Balanced"):
        controls.set_fixed_range(True)
    controls.restore()
    assert controls.processing_preset == "balanced"
    assert not controls._processing_preset_owned
    assert controls._apply_processing_preset.call_args_list[-1].args == ("balanced",)


def test_unknown_preset_and_fixed_mode_reject_writes(controls):
    controls._processing_command = Mock()
    for value in ([], True, 1, "manual"):
        with pytest.raises(ValueError):
            controls.set_processing_preset(value)
    controls._fixed_range_owned = True
    with pytest.raises(CameraError, match="fixed mode"):
        controls.set_processing_preset("shadow")
    controls._processing_command.assert_not_called()


def test_fixed_range_packets_and_restore_order(controls):
    controls._fixed_range_bounds = (1000, 2800)
    controls._fixed_range_baseline = Mock(return_value=(1000, 2800))
    device = Mock()
    packets = []

    def transfer(kind, request, value, index, data, timeout):
        assert value == 0 and index == 0xA00
        if kind == 0x41:
            assert request == 1
            packets.append(bytes(data))
            return len(data)
        if request == 0x85:
            assert kind == 0xC1 and data == 4
            return (11).to_bytes(2, "little")
        assert kind == 0xC1 and request == 0x81 and data == 11
        return bytes.fromhex("f00736fe030000000138ff")

    device.ctrl_transfer.side_effect = transfer
    controls.camera.device = device
    controls.set_fixed_range(True)
    assert controls.fixed_range
    with pytest.raises(CameraError, match="Turn off fixed range"):
        controls.set("contrast", 50, True)
    controls.set_fixed_range(False)
    assert not controls.fixed_range and not controls._fixed_range_owned
    for packet, bounds in zip(packets, [(0, 16383), (0, 16383), (1000, 2800), (0, 0)], strict=True):
        body = b"\x36\xfe\x00" + struct.pack(">IHH", 0xF113, *bounds)
        assert packet == bytes((0xF0, len(body))) + body + bytes((sum(body) & 255, 0xFF))
    controls.restore_fixed_range()
    assert len(packets) == 4


def test_fixed_range_failed_enable_restores_bounds_and_disables(controls):
    controls._fixed_range_bounds = (1000, 2800)
    controls._fixed_range_baseline = Mock(return_value=(1000, 2800))
    controls._fixed_range_command = Mock(side_effect=[None, CameraError("ack failed"), None, None])
    with pytest.raises(CameraError, match="ack failed"):
        controls.set_fixed_range(True)
    assert [call.args for call in controls._fixed_range_command.call_args_list] == [
        (0, 16383), (0, 16383), (1000, 2800), (0, 0)
    ]
    assert not controls.fixed_range and not controls._fixed_range_owned


def test_fixed_range_failed_restore_still_disables_and_retains_cleanup(controls):
    controls._fixed_range_bounds = (1000, 2800)
    controls._fixed_range_owned = controls.fixed_range = True
    controls._fixed_range_command = Mock(side_effect=[CameraError("restore failed"), None])
    with pytest.raises(CameraError, match="restore failed"):
        controls.restore_fixed_range()
    assert [call.args for call in controls._fixed_range_command.call_args_list] == [(1000, 2800), (0, 0)]
    assert controls._fixed_range_owned


def test_fixed_range_rejects_unknown_preset_before_serial_writes(controls):
    device = Mock()
    device.ctrl_transfer.side_effect = [2, (5).to_bytes(4, "little"), b"\x01\x00\x00\x00\x00"]
    controls.camera.device = device
    with pytest.raises(CameraError, match="tested Duo ISP layout"):
        controls.set_fixed_range(True)
    assert not controls._fixed_range_owned
    assert all(call.args[0] != 0x41 for call in device.ctrl_transfer.call_args_list)


def test_fixed_range_requires_boolean(controls):
    with pytest.raises(ValueError, match="boolean"):
        controls.set_fixed_range(1)


def test_fixed_mode_requires_detail_but_detail_is_independent(controls):
    controls._fixed_range_command = Mock()
    controls.set("detail_enabled", 0, True)
    with pytest.raises(CameraError, match="Enable detail enhancement"):
        controls.set_fixed_range(True)
    controls._fixed_range_command.assert_not_called()
    controls.set("detail_enabled", 1, True)
    assert not controls.fixed_range
    controls._fixed_range_command.assert_not_called()


def test_disabling_detail_restores_fixed_mode_before_writing_control(controls):
    controls._fixed_range_bounds = (1000, 2800)
    controls._fixed_range_baseline = Mock(return_value=(1000, 2800))
    controls._fixed_range_command = Mock()
    controls.set_fixed_range(True)
    assert controls.fixed_range
    controls.set("detail_enabled", 0, True)
    assert not controls.fixed_range and not controls._fixed_range_owned
    assert controls.state()["detail_enabled"]["value"] == 0
    assert [call.args for call in controls._fixed_range_command.call_args_list] == [
        (0, 16383), (0, 16383), (1000, 2800), (0, 0)
    ]


def test_detail_strength_can_change_while_fixed_mode_stays_enabled(controls):
    controls._fixed_range_bounds = (1000, 2800)
    controls._fixed_range_baseline = Mock(return_value=(1000, 2800))
    controls._fixed_range_command = Mock()
    controls.set_fixed_range(True)
    controls.set("detail", 60, True)
    assert controls.fixed_range and controls.state()["detail_enabled"]["value"] == 1
    assert controls.state()["detail"]["value"] == 60
    assert [call.args for call in controls._fixed_range_command.call_args_list] == [
        (0, 16383), (0, 16383), (1000, 2800), (0, 0), (0, 16383), (0, 16383)
    ]


@pytest.mark.parametrize("bounds", [(0, 0), (100, 99), (-1, 100), (0, 16384), (True, 100), (0, 1.5), None])
def test_fixed_range_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        validate_fixed_range_bounds(bounds)


def test_fixed_range_rejects_modified_isp_export(controls):
    export = bytes(3956)
    packets = [
        b"\x02" + sequence.to_bytes(4, "little") + export[start:start + 507]
        for sequence, start in enumerate(range(0, len(export), 507), 1)
    ]
    device = Mock()
    device.ctrl_transfer.side_effect = [
        2, (5).to_bytes(4, "little"), b"\x01\x74\x0f\x00\x00", *packets
    ]
    controls.camera.device = device
    with pytest.raises(CameraError, match="verified factory ISP preset"):
        controls.set_fixed_range(True)
    assert not controls._fixed_range_owned
    assert all(call.args[0] != 0x41 for call in device.ctrl_transfer.call_args_list)


class Device:
    def __init__(self):
        self.blocks = {key: bytearray(size) for key, size in BLOCK_LENGTHS.items()}
        self.blocks[2, 1][:] = b"\x01\x32"
        self.blocks[2, 2][:] = b"\x01\x32"
        self.blocks[2, 5][1] = 2
        self.blocks[2, 5][5] = 1
        self.blocks[2, 5][6] = 1
        thermometry = self.blocks[3, 1]
        thermometry[75] = 2
        struct.pack_into("<I", thermometry, 16, 80)
        struct.pack_into("<I", thermometry, 21, 100)
        struct.pack_into("<I", thermometry, 26, 1200)
        struct.pack_into("<I", thermometry, 40, 100)
        struct.pack_into("<I", thermometry, 76, 13000)
        self.selected = None
        self.writes = []
        self.ignore_next_write = False
        self.last_request = None

    def ctrl_transfer(self, kind, request, value, index, data, timeout):
        assert index == 0x0A00
        selector = value >> 8
        if kind == 0x21 and selector == 5:
            self.selected = tuple(data)
            self.last_request = "select"
            return 2
        assert selector == self.selected[0]
        block = self.blocks[self.selected]
        if request == 0x85:
            assert data == 4
            self.last_request = "length"
            return len(block).to_bytes(2, "little")
        if kind == 0xA1:
            assert request == 0x81 and data == len(block)
            self.last_request = "read"
            return bytes(block)
        assert kind == 0x21 and request == 1
        assert self.last_request == "length", "GET_CUR must not precede SET_CUR"
        self.writes.append((self.selected, bytes(data)))
        if self.ignore_next_write:
            self.ignore_next_write = False
        else:
            block[:] = data
        self.last_request = "write"
        return len(data)


@pytest.fixture
def controls(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    clock = iter(np.arange(0, 1000, 0.25))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    camera = type("Camera", (), {"device": Device()})()
    controls = HardwareControls(camera)
    controls.load()
    return controls


def test_controls_preserve_snapshot_and_other_enabled_fields(controls):
    original = {key: bytes(data) for key, data in controls.camera.device.blocks.items()}
    snapshot = json.loads(controls.snapshot_path.read_text())
    assert snapshot["values"]["ambient"] == 30
    assert snapshot["blocks"]["3:1"] == original[3, 1].hex()
    controls.camera.device.blocks[3, 1][8] = 123  # unrelated field changed independently
    controls.set("ambient", 27.5, True)
    controls.set("emissivity", 0.95, True)
    controls.set("distance", 0.3, True)
    controls.set("ambient", 27.5, False)
    data = controls.camera.device.blocks[3, 1]
    assert struct.unpack_from("<I", data, 76)[0] == 13000
    assert struct.unpack_from("<I", data, 16)[0] == 95
    assert struct.unpack_from("<I", data, 21)[0] == 30
    assert data[8] == 123
    assert controls.state()["ambient"]["value"] == 30
    assert controls.values["ambient"] == 27.5
    controls.restore()
    expected = bytearray(original[3, 1])
    expected[8] = 123
    assert data == expected
    assert not controls.enabled


def test_ignored_write_rolls_back_without_enabling_row(controls):
    original = bytes(controls.camera.device.blocks[2, 1])
    controls.camera.device.ignore_next_write = True
    with pytest.raises(CameraError, match="did not apply"):
        controls.set("brightness", 30, True)
    assert bytes(controls.camera.device.blocks[2, 1]) == original
    assert "brightness" not in controls.enabled
    assert controls.values["brightness"] == 50
    assert controls.camera.device.writes[-1] == ((2, 1), original)


@pytest.mark.parametrize(
    "name,value",
    [
        ("ambient", float("nan")),
        ("distance", 0.25),
        ("emissivity", 1.1),
        ("palette", 3),
        ("brightness", 30.5),
        ("brightness", "30"),
    ],
)
def test_invalid_controls_never_write(controls, name, value):
    with pytest.raises((ValueError, TypeError)):
        controls.set(name, value, True)
    assert not controls.camera.device.writes


def test_camera_temperature_conversion_does_not_reanchor_hardware_output():
    frame, _ = frame_with_preview()
    renderer = ThermalRenderer(scale=1)
    renderer.native_temperatures = True
    first = renderer.render_detailed(frame)
    renderer.ambient_celsius = 40
    second = renderer.render_detailed(frame)
    np.testing.assert_allclose(first.temperatures_celsius, first.raw_counts / 64 - 50)
    np.testing.assert_array_equal(first.temperatures_celsius, second.temperatures_celsius)


def test_hardware_brightness_and_yuyv_color_are_not_normalized_away():
    frame, _ = frame_with_preview()
    yuyv = np.frombuffer(frame, dtype=np.uint8, offset=IMAGE_OFFSET * 2).copy()
    yuyv[0::2] = 80
    yuyv[1::4] = 80
    yuyv[3::4] = 190
    colored = frame[: IMAGE_OFFSET * 2] + yuyv.tobytes()
    renderer = ThermalRenderer(scale=1)
    renderer.camera_preview = True
    renderer.camera_color = True
    result = renderer.render_detailed(colored)
    expected = cv2.cvtColor(yuyv.reshape(384, 512, 2), cv2.COLOR_YUV2BGR_YUY2)
    np.testing.assert_array_equal(
        result.image, cv2.resize(expected, (256, 192), interpolation=cv2.INTER_AREA)
    )
    renderer.camera_color = False
    renderer.color_palette = "white_hot"
    first = renderer.render_detailed(colored)
    yuyv[0::2] = 110
    brighter = frame[: IMAGE_OFFSET * 2] + yuyv.tobytes()
    second = renderer.render_detailed(brighter)
    assert first.image.mean() == 80
    assert second.image.mean() == 110
    assert first.stats == second.stats


def test_unsupported_layout_leaves_hardware_controls_unavailable(controls):
    controls.camera.device.blocks[2, 5] = bytearray(78)
    unloaded = HardwareControls(controls.camera)
    with pytest.raises(CameraError, match="unsupported control layout"):
        unloaded.load()
    assert not unloaded.original and unloaded.snapshot_path is None
    assert not controls.camera.device.writes


def test_zero_brightness_keeps_camera_preview_selected():
    frame, _ = frame_with_preview()
    dark = frame[: IMAGE_OFFSET * 2] + bytes(len(frame) - IMAGE_OFFSET * 2)
    renderer = ThermalRenderer(scale=1)
    renderer.camera_preview = True
    renderer.color_palette = "white_hot"
    result = renderer.render_detailed(dark)
    assert result.image_source == "preview"
    assert not np.any(result.image)


class CalibrationDevice(Device):
    def __init__(self):
        super().__init__()
        self.blocks.update({(1, 24): bytearray(11), (2, 4): bytearray(1)})
        self.statuses = [b"\x00"]
        self.status_selection = None

    def ctrl_transfer(self, kind, request, value, index, data, timeout):
        if value == 0x0600:
            assert kind == 0xA1 and index == 0x0A00
            self.status_selection = self.selected
            if request == 0x85:
                assert data == 4
                return b"\x01\x00"
            assert request == 0x81 and data == 1
            return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return super().ctrl_transfer(kind, request, value, index, data, timeout)


def test_calibration_commands_poll_direct_status_and_restore_automatic_operation(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    device = CalibrationDevice()
    control = HardwareControls(SimpleNamespace(device=device))
    device.statuses = [b"\x01", b"\x00"]
    control.set_auto_calibrate(False)
    assert control.auto_calibrate is False
    assert device.writes == [((1, 24), bytes.fromhex("0200000120000000000000"))]
    assert device.status_selection == (1, 24)
    control.calibrate_now()
    assert device.writes[-1] == ((2, 4), b"\x01")
    assert device.status_selection == (2, 4)
    assert control.auto_calibrate is False
    control.restore_auto_calibrate()
    assert device.writes[-1] == ((1, 24), bytes.fromhex("0200000120000001000000"))
    assert control.auto_calibrate is True
    count = len(device.writes)
    control.restore_auto_calibrate()
    assert len(device.writes) == count


def test_failed_auto_command_keeps_previous_state_and_can_restore(monkeypatch):
    from types import SimpleNamespace

    device = CalibrationDevice()
    control = HardwareControls(SimpleNamespace(device=device))
    control.set_auto_calibrate(True)
    device.statuses = [b"\x09"]
    with pytest.raises(CameraError, match="rejected"):
        control.set_auto_calibrate(False)
    assert control.auto_calibrate is True
    device.statuses = [b"\x00"]
    control.restore_auto_calibrate()
    assert control.auto_calibrate is True
    with pytest.raises(ValueError, match="boolean"):
        control.set_auto_calibrate(1)


def test_busy_calibration_command_has_bounded_timeout(monkeypatch):
    from types import SimpleNamespace

    clock = iter(np.arange(0, 100, 0.25))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    device = CalibrationDevice()
    device.statuses = [b"\x01"]
    control = HardwareControls(SimpleNamespace(device=device))
    with pytest.raises(CameraError, match="timed out"):
        control.calibrate_now()


def prepare_tone_controls(controls):
    original = bytearray(controls.original[2, 5])
    original[23] = 1
    controls.original[2, 5] = bytes(original)
    controls._fixed_range_baseline = Mock(return_value=(1000, 2800))
    controls._apply_boost = Mock()
    controls._tone_command = Mock()
    return controls


def test_tone_upload_is_incremental_and_preserves_brightness(controls):
    controls = prepare_tone_controls(controls)
    brightness = controls.read(2, 1)
    controls.set_tone(25, 0)
    assert controls.tone_busy and len(controls._tone_queue) == 257
    assert not controls.advance_tone()
    assert controls._tone_command.call_count == 1
    body, replies = controls._tone_command.call_args.args
    assert body[:8] == bytes.fromhex('3674130000206110')
    assert replies == (bytes.fromhex('f0053674130301c1ff'),)
    for _ in range(255):
        assert not controls.advance_tone()
    assert controls.advance_tone()
    assert not controls.tone_busy
    assert controls._tone_command.call_args.args[0][-4:] == bytes(4)
    assert controls.read(2, 1) == brightness
    controls.restore_tone()
    assert (controls.gamma, controls.boost, controls._tone_owned) == (50, 0, False)


def test_neutral_gamma_rebuilds_native_curve_during_pending_upload(controls):
    controls = prepare_tone_controls(controls)
    controls.set_tone(25, 3)
    controls.advance_tone()
    controls._apply_boost.reset_mock()
    controls.set_tone(50, 3)
    controls._apply_boost.assert_called_once_with(3)
    assert not controls.tone_busy and controls.boost


def test_tone_upload_failure_restores_native_processing(controls):
    controls = prepare_tone_controls(controls)
    controls.set_tone(75, 3)
    controls._tone_command.side_effect = CameraError('upload failed')
    with pytest.raises(CameraError, match='upload failed'):
        controls.advance_tone()
    assert not controls.tone_busy and not controls._tone_owned
    controls._apply_boost.assert_called_with(0)


@pytest.mark.parametrize("mode", (0, 1, 2, 3))
def test_boost_applies_twice_and_consumes_both_firmware_replies(controls, mode):
    controls._tone_command = Mock()
    controls._apply_boost(mode)
    assert controls._tone_command.call_count == 2
    for call in controls._tone_command.call_args_list:
        assert call.args == (bytes((0x36, 0x78, 0x31, 0, mode)),
                             (bytes.fromhex('f0053678310301e3ff'),
                              bytes.fromhex('f0053678310400e3ff')))


@pytest.mark.parametrize("mode", (1, 2, 3))
def test_gamma_changes_keep_selected_boost_mode(controls, mode):
    controls = prepare_tone_controls(controls)
    controls.set_tone(25, mode)
    first_curve = controls._tone_queue.copy()
    assert controls.boost == mode
    controls._apply_boost.assert_called_with(mode)
    controls.set_tone(75, mode)
    assert controls.boost == mode and controls._tone_queue != first_curve
    controls.set_tone(50, mode)
    controls._apply_boost.assert_called_with(mode)
    assert controls.boost == mode and not controls.tone_busy


@pytest.mark.parametrize("invalid", (-1, 4, True, 1.5, "1", None))
def test_invalid_boost_mode_is_rejected_before_any_transfer(controls, invalid):
    controls = prepare_tone_controls(controls)
    with pytest.raises(ValueError):
        controls.set_tone(50, invalid)
    controls._apply_boost.assert_not_called()
    controls._tone_command.assert_not_called()
