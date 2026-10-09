import struct
from unittest.mock import Mock

import numpy as np
import pytest
import usb.core
from support.frames import make_frame

import topdon_duo.camera as camera_module
from topdon_duo.camera import (
    FRAME_BYTES,
    HEADER_U16,
    SENSOR_HEIGHT,
    SENSOR_PIXELS,
    SENSOR_WIDTH,
    YUY2_FRAME_BYTES,
    CameraAccessError,
    CameraError,
    FrameAssembler,
    LinuxFrameAssembler,
    NegotiatedMode,
    TC002CDuoCamera,
    build_probe,
    decode_duo_frame,
    has_yuy2_preview,
    parse_probe,
    raw_temperatures,
)


def test_probe_round_trip():
    probe = build_probe()
    struct.pack_into("<I", probe, 18, FRAME_BYTES)
    struct.pack_into("<I", probe, 22, 5020)
    mode = parse_probe(probe)
    assert mode.frame_index == 10
    assert mode.fps == 25
    assert mode.max_frame_size == FRAME_BYTES
    assert mode.max_payload_size == 5020


def test_frame_assembler_uses_fid_boundary():
    source = make_frame()
    assembler = FrameAssembler()
    result = None
    for offset in range(0, len(source), 5018):
        result = assembler.feed(bytes([2, 0]) + source[offset : offset + 5018]) or result
    result = assembler.feed(bytes([2, 1]) + b"next") or result
    assert result == source


def test_decode_and_temperature_conversion():
    frame = make_frame()
    telemetry, raw, preview = decode_duo_frame(frame)
    raw.reshape(-1)[:2_000] -= 64
    temperatures = raw_temperatures(raw, ambient_celsius=22.0)
    assert telemetry.shape == (HEADER_U16,)
    assert raw.shape == (SENSOR_HEIGHT, SENSOR_WIDTH)
    assert preview.shape == (SENSOR_HEIGHT, SENSOR_WIDTH)
    assert temperatures[0, 0] == pytest.approx(22.0)
    assert temperatures[100, 100] == pytest.approx(23.0)


def test_decode_rejects_wrong_magic():
    with pytest.raises(ValueError, match="invalid Duo frame magic"):
        decode_duo_frame(bytes(FRAME_BYTES))


def test_linux_larger_preview_keeps_complete_radiometric_frame():
    # This camera advertises a 512x384 preview on Ubuntu, following the
    # same telemetry and 256x192 radiometric plane as the macOS stream.
    source = make_frame() + bytes(512 * 384 * 2 - SENSOR_PIXELS * 2)
    assembler = LinuxFrameAssembler(len(source))
    result = None
    for offset in range(0, len(source), 8134):
        result = assembler.feed(bytes([2, 0x80]) + source[offset:offset + 8134]) or result
    result = assembler.feed(bytes([2, 0x82])) or result
    assert result == source
    _, raw, _ = decode_duo_frame(result)
    assert np.all(raw == 20_000)
    assert has_yuy2_preview(result)
    assert not has_yuy2_preview(make_frame())


def test_linux_rejects_dropped_payload_and_recovers_next_frame():
    source = make_frame() + bytes(294_912)
    assembler = LinuxFrameAssembler(len(source))
    # Reproduce the missing data observed in the captured Ubuntu USB stream:
    # the old assembler accepted 488358 bytes as a 201248-byte frame.
    damaged = source[:10_000] + source[17_802:]
    assert len(damaged) == 488_358
    assert assembler.feed(bytes([2, 0x82]) + damaged) is None
    assert assembler.feed(bytes([2, 0x83]) + source) == source


def test_linux_rejects_stale_midframe_data_even_when_size_matches():
    assembler = LinuxFrameAssembler()
    assert assembler.feed(bytes([2, 0x82]) + bytes(FRAME_BYTES)) is None
    assert assembler.feed(bytes([2, 0x83]) + make_frame()) == make_frame()


def test_rejected_frame_observer_sees_buffer_before_it_is_cleared():
    assembler = LinuxFrameAssembler()
    captures = []
    assembler.rejected_frame_observer = lambda frame: captures.append(bytes(frame))
    assert assembler.feed(b"\x02\x82short") is None
    assert captures == [b"short"]
    assert assembler._data == b""


def test_assembler_diagnostics_distinguish_header_error_size_and_magic():
    assembler = LinuxFrameAssembler()
    assert assembler.feed(b"\x01") is None
    assert assembler.feed(b"\x05\x00") is None
    assert assembler.feed(b"\x02\x40") is None
    assert assembler.feed(b"\x02\x82short") is None
    assert assembler.feed(b"\x02\x83" + bytes(FRAME_BYTES)) is None
    assert assembler.feed(b"\x02\x82" + make_frame()) == make_frame()
    assert assembler.rejected == {
        "invalid_header": 2, "uvc_error": 1, "partial": 0,
        "size_mismatch": 1, "magic_mismatch": 1,
    }


def test_stream_diagnostics_distinguish_timeouts_from_rejected_frames(monkeypatch):
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    clock = iter((0.0, 2.0, 4.0))
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: next(clock))
    read_clock = iter((0.0, 2.0, 2.25, 2.26, 2.27, 2.28))
    monkeypatch.setattr(camera_module.time, "perf_counter", lambda: next(read_clock))
    camera = TC002CDuoCamera()
    camera.usb_queue_depth = 0
    camera.mode = NegotiatedMode(1, 10, 400_000, FRAME_BYTES, 5020)
    camera.device = Mock()
    camera.device.read.side_effect = [
        usb.core.USBTimeoutError("Timed out"),
        b"\x02\x82" + bytes(FRAME_BYTES),
        b"\x02\x83" + make_frame(),
    ]
    camera._running.set()
    events = []
    camera.stream_observer = events.append
    frames = camera.frames()
    try:
        assert next(frames) == make_frame()
    finally:
        frames.close()
    assert events[0]["timeouts"] == 1
    assert events[0]["packets"] == 0
    assert events[0]["longest_read_seconds"] == 2.0
    assert events[0]["packet_lengths"] == {}
    assert events[1]["packets"] == 1
    assert events[1]["rejected"]["magic_mismatch"] == 1
    assert events[1]["longest_host_gap_seconds"] == 0.25
    assert events[1]["longest_read_seconds"] == 0.01
    assert events[1]["packet_lengths"] == {str(FRAME_BYTES + 2): 1}
    assert events[1]["packet_headers"] == {"0282": 1}
    assert events[2]["packets"] == 2
    assert events[2]["frames"] == 1
    assert events[2]["bytes"] == (FRAME_BYTES + 2) * 2
    assert events[2]["packet_headers"] == {"0283": 1}


@pytest.mark.parametrize("platform,expected", [("linux", 32), ("darwin", 32), ("win32", 0)])
def test_supported_platforms_default_to_queued_usb(monkeypatch, platform, expected):
    monkeypatch.setattr(camera_module.sys, "platform", platform)
    assert TC002CDuoCamera().usb_queue_depth == expected


@pytest.mark.parametrize("full_preview", [False, True])
def test_macos_queued_capture_preserves_tolerant_framing(monkeypatch, full_preview):
    monkeypatch.setattr(camera_module.sys, "platform", "darwin")
    expected = make_frame()
    if full_preview:
        expected += bytes(YUY2_FRAME_BYTES - FRAME_BYTES)
    padded_frame = expected + b"padding"
    packets = iter((b"\x02\x82" + padded_frame,))

    class Reader:
        def __init__(self, _device, _endpoint, _size, _timeout, depth):
            assert depth == 32
            self.statuses = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return next(packets)

    monkeypatch.setattr(camera_module, "QueuedBulkReader", Reader)
    camera = TC002CDuoCamera()
    camera.device = Mock()
    camera.mode = NegotiatedMode(1, 10, 400_000, FRAME_BYTES, 5020)
    camera._running.set()
    frames = camera.frames()
    try:
        assert next(frames) == expected
    finally:
        frames.close()
    camera.device.read.assert_not_called()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_usb_setup_and_driver_restoration(monkeypatch, platform):
    monkeypatch.setattr(camera_module.sys, "platform", platform)
    device = Mock()
    device.is_kernel_driver_active.return_value = True
    monkeypatch.setattr(TC002CDuoCamera, "find", Mock(return_value=device))
    claim = Mock()
    release = Mock()
    monkeypatch.setattr(camera_module.usb.util, "claim_interface", claim)
    monkeypatch.setattr(camera_module.usb.util, "release_interface", release)
    monkeypatch.setattr(camera_module.usb.util, "dispose_resources", Mock())
    camera = TC002CDuoCamera()
    mode = NegotiatedMode(1, 10, 400_000, FRAME_BYTES, 5020)
    monkeypatch.setattr(camera, "_negotiate", lambda: mode)
    assert camera.open() == mode
    if platform == "linux":
        device.set_configuration.assert_not_called()
        device.get_active_configuration.assert_called_once()
    else:
        device.set_configuration.assert_called_once()
        device.get_active_configuration.assert_not_called()
    assert [call.args[0] for call in device.detach_kernel_driver.call_args_list] == [0, 1]
    assert [call.args[1] for call in claim.call_args_list] == [0, 1]
    camera.close()
    assert [call.args[1] for call in release.call_args_list] == [1, 0]
    assert [call.args[0] for call in device.attach_kernel_driver.call_args_list] == [1, 0]


def test_linux_unconfigured_device_is_configured(monkeypatch):
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    device = Mock()
    device.get_active_configuration.side_effect = usb.core.USBError("Configuration not set")
    monkeypatch.setattr(TC002CDuoCamera, "find", Mock(return_value=device))
    camera = TC002CDuoCamera()
    monkeypatch.setattr(camera, "_detach_and_claim", Mock())
    monkeypatch.setattr(camera, "_negotiate", lambda: NegotiatedMode(1, 10, 400_000, FRAME_BYTES, 5020))
    monkeypatch.setattr(camera_module.usb.util, "dispose_resources", Mock())
    camera.open()
    device.set_configuration.assert_called_once()
    camera.close()


def test_failed_negotiation_restores_linux_driver(monkeypatch):
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    device = Mock()
    device.is_kernel_driver_active.return_value = True
    monkeypatch.setattr(TC002CDuoCamera, "find", Mock(return_value=device))
    monkeypatch.setattr(camera_module.usb.util, "claim_interface", Mock())
    monkeypatch.setattr(camera_module.usb.util, "release_interface", Mock())
    monkeypatch.setattr(camera_module.usb.util, "dispose_resources", Mock())
    camera = TC002CDuoCamera()
    monkeypatch.setattr(camera, "_negotiate", Mock(side_effect=CameraError("rejected mode")))
    with pytest.raises(CameraError, match="rejected mode"):
        camera.open()
    assert camera.device is None
    assert camera._detached == []
    assert device.attach_kernel_driver.call_count == 2


@pytest.mark.parametrize("platform, message", [("linux", "udev"), ("darwin", "sudo")])
def test_usb_access_error_is_platform_specific(monkeypatch, platform, message):
    monkeypatch.setattr(camera_module.sys, "platform", platform)
    device = Mock()
    denied = usb.core.USBError("Access denied", errno=13)
    device.get_active_configuration.side_effect = denied
    device.set_configuration.side_effect = denied
    monkeypatch.setattr(TC002CDuoCamera, "find", Mock(return_value=device))
    monkeypatch.setattr(camera_module.usb.util, "dispose_resources", Mock())
    with pytest.raises(CameraAccessError, match=message):
        TC002CDuoCamera().open()
