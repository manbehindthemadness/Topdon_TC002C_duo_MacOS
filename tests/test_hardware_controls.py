import json
import struct

import cv2
import numpy as np
import pytest
from test_render import frame_with_preview

from topdon_duo import hardware_controls as module
from topdon_duo.camera import IMAGE_OFFSET, CameraError
from topdon_duo.hardware_controls import BLOCK_LENGTHS, HardwareControls
from topdon_duo.render import ThermalRenderer


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
