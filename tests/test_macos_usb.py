"""
Exercise the privilege boundary with fake USB and private pipes, without sudo or hardware.
"""

import io
import os
import struct
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock

import pytest
import usb.core

from topdon_duo.camera import FRAME_BYTES, CameraError, NegotiatedMode, TC002CDuoCamera
from topdon_duo.macos_usb import client, protocol, service


class FakeCamera:
    """
    Model one USB owner with deterministic control replies and complete byte frames.
    """

    def __init__(self) -> None:
        """
        Provide a fake device without opening USB.
        """
        self.device = None
        self.timeout_ms = 2000
        self.usb_queue_depth = 32
        self.stream_observer = None
        self.running = threading.Event()
        self.closed = threading.Event()
        self.started = threading.Event()
        self.transfers = Mock(side_effect=self.control)

    @staticmethod
    def control(request_type: int, _request: int, _value: int, _index: int,
                data: bytes | int, *, timeout: int) -> bytes | int:
        """
        Return an independent reply or reject a specific tested control operation.
        """
        assert 0 < timeout <= 2000
        if request_type & 0x80:
            return b"2.0\x00"[:int(data)]
        if data == b"reject":
            raise usb.core.USBError("Access denied", errno=13)
        assert isinstance(data, bytes)
        return len(data)

    def open(self) -> NegotiatedMode:
        """
        Match the tested negotiated frame metadata.
        """
        self.device = Mock(ctrl_transfer=self.transfers)
        self.running.set()
        return NegotiatedMode(1, 10, 400000, FRAME_BYTES, 5020)

    def frames(self) -> Iterator[bytes]:
        """
        Produce a complete binary frame containing all byte values.
        """
        self.started.set()
        yield bytes(range(256)) * (FRAME_BYTES // 256) + bytes(FRAME_BYTES % 256)
        while self.running.is_set():
            self.closed.wait(0.01)

    def stop_stream(self) -> None:
        """
        Wake acquisition without releasing the fake control device.
        """
        self.running.clear()

    def close(self) -> None:
        """
        Record completion of cleanup.
        """
        self.device = None
        self.closed.set()


class FakeProcess:
    """
    Connect the production client to the production service over real local pipes.
    """

    def __init__(self, camera: FakeCamera) -> None:
        """
        Start a USB service thread with no elevated privileges.
        """
        command_read, command_write = os.pipe()
        reply_read, reply_write = os.pipe()
        self.stdin = os.fdopen(command_write, "wb", buffering=0)
        self.stdout = os.fdopen(reply_read, "rb", buffering=0)

        def run() -> None:
            """
            Close both helper ends when the service finishes.
            """
            with (
                os.fdopen(command_read, "rb", buffering=0) as source,
                os.fdopen(reply_write, "wb", buffering=0) as target,
            ):
                service.serve(source, target, camera)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def wait(self, timeout: float) -> int:
        """
        Require the helper to exit after the client closes its input.
        """
        self.thread.join(timeout)
        assert not self.thread.is_alive()
        return 0


def test_helper_round_trip_preserves_frames_controls_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Exercise negotiation, binary frames, control failures, stop and disconnect cleanup.
    """
    physical = FakeCamera()
    process = FakeProcess(physical)
    launch = Mock(return_value=process)
    monkeypatch.setattr(client.subprocess, "Popen", launch)
    camera = client.MacOSCamera()
    camera.usb_queue_depth = 64
    try:
        mode = camera.open()
        assert mode.fps == 25
        assert physical.usb_queue_depth == 64
        device = camera.device
        assert device is not None
        assert device.ctrl_transfer(0xA1, 0x81, 0x400, 0x0A00, 4) == b"2.0\x00"
        assert device.ctrl_transfer(0x21, 1, 0x500, 0x0A00, b"\x02\x05") == 2
        with pytest.raises(usb.core.USBError) as failure:
            device.ctrl_transfer(0x21, 1, 0x500, 0x0A00, b"reject")
        assert failure.value.errno == 13
        frames = camera.frames()
        frame = next(frames)
        assert len(frame) == FRAME_BYTES
        assert frame[:256] == bytes(range(256))
        camera.stop_stream()
        assert list(frames) == []
        assert device.ctrl_transfer(0xA1, 0x81, 0x400, 0x0A00, 4) == b"2.0\x00"
    finally:
        camera.close()
    assert physical.closed.is_set()
    assert camera.device is None
    options = launch.call_args.kwargs
    assert options["cwd"] == "/"
    assert options["env"] == {"PATH": "/usr/bin:/bin"}
    assert "-I" in launch.call_args.args[0]


@pytest.mark.parametrize("changes,payload", [
    ({"index": 0}, b""), ({"length": 5000}, b""), ({"timeout": True}, b""),
    ({"request_type": 0x80}, b""), ({"request": 9}, b""), ({}, b"unwanted"),
    ({"request_type": 0x21}, b"short"),
])
def test_helper_rejects_invalid_transfers_before_usb(changes: dict, payload: bytes) -> None:
    """
    Reject malformed and unrelated USB operations before calling the device.
    """
    camera = FakeCamera()
    camera.open()
    message = {"request_type": 0xA1, "request": 0x81, "value": 0x400,
               "index": 0x0A00, "length": 4, "timeout": 2000, **changes}
    with pytest.raises(ValueError):
        service.transfer(camera, message, payload)
    camera.transfers.assert_not_called()


def test_protocol_handles_fragmented_writes_and_binary_data() -> None:
    """
    Preserve binary data even when the pipe accepts only part of each write.
    """
    class PartialWriter(io.BytesIO):
        """
        Simulate short successful writes.
        """

        def write(self, data: Any) -> int:
            """
            Accept a bounded prefix.
            """
            return super().write(data[:3])

    stream = PartialWriter()
    protocol.send(stream, {"kind": "frame"}, b"\x00\xff" * 10)
    stream.seek(0)
    assert protocol.receive(stream) == ({"kind": "frame"}, b"\x00\xff" * 10)


@pytest.mark.parametrize("data,exception", [
    (struct.pack("!II", protocol.MAX_HEADER + 1, 0), ValueError),
    (struct.pack("!II", 2, protocol.MAX_PAYLOAD + 1), ValueError),
    (struct.pack("!II", 2, 0) + b"[]", TypeError),
    (struct.pack("!II", 2, 1) + b"{}", EOFError),
])
def test_protocol_rejects_malformed_messages(data: bytes, exception: type[Exception]) -> None:
    """
    Bound allocations and reject truncated or non-object messages.
    """
    with pytest.raises(exception):
        protocol.receive(io.BytesIO(data))


def test_helper_disconnect_releases_usb_without_a_close_command() -> None:
    """
    EOF from an exiting viewer releases the helper's resources.
    """
    camera = FakeCamera()
    commands = io.BytesIO()
    protocol.send(commands, {"id": 1, "operation": "open", "timeout": 2000, "depth": 32})
    commands.seek(0)
    service.serve(commands, io.BytesIO(), camera)
    assert camera.closed.is_set()


def test_helper_selection_preserves_linux_root_and_injected_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Change only normal-user macOS sessions with the real Duo factory.
    """
    monkeypatch.setattr(client.sys, "platform", "darwin")
    monkeypatch.setattr(client.os, "geteuid", lambda: 501)
    assert client.use_helper(TC002CDuoCamera) is client.MacOSCamera
    injected = Mock()
    assert client.use_helper(injected) is injected
    monkeypatch.setattr(client.os, "geteuid", lambda: 0)
    assert client.use_helper(TC002CDuoCamera) is TC002CDuoCamera
    monkeypatch.setattr(client.sys, "platform", "linux")
    monkeypatch.setattr(client.os, "geteuid", lambda: 1000)
    assert client.use_helper(TC002CDuoCamera) is TC002CDuoCamera


def test_authorization_failure_is_visible_and_does_not_start_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Explain terminal authorization when sudo exits without a protocol response.
    """
    process = Mock(stdin=io.BytesIO(), stdout=io.BytesIO())
    monkeypatch.setattr(client.subprocess, "Popen", Mock(return_value=process))
    camera = client.MacOSCamera()
    with pytest.raises(CameraError, match="sudo -v"):
        camera.open()
    assert camera.device is None


def test_service_refuses_non_root_or_non_macos(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Refuse helper execution before touching USB on unsupported launches.
    """
    monkeypatch.setattr(service.sys, "platform", "linux")
    assert service.main() == 1
    assert "authorization" in capsys.readouterr().err


def test_helper_keeps_only_the_newest_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Drain a burst while the viewer is idle and discard older complete frames.
    """
    finished = threading.Event()

    class BurstCamera(FakeCamera):
        """
        Produce independently identifiable frames faster than the viewer consumes them.
        """

        def frames(self) -> Iterator[bytes]:
            """
            Signal after all complete messages have been published.
            """
            for number in range(30):
                yield bytes((number,)) * FRAME_BYTES
            finished.set()
            while self.running.is_set():
                self.closed.wait(0.01)

    physical = BurstCamera()
    monkeypatch.setattr(client.subprocess, "Popen", Mock(return_value=FakeProcess(physical)))
    camera = client.MacOSCamera()
    try:
        camera.open()
        camera.request("start")
        assert finished.wait(5)
        # This reply follows the burst on the same pipe, so receiving it proves drainage.
        camera.request("transfer", request_type=0xA1, request=0x81, value=0x400,
                       index=0x0A00, length=4, timeout=2000)
        assert list(camera._frames) == [bytes((29,)) * FRAME_BYTES]
    finally:
        camera.close()
    assert physical.closed.is_set()


def test_stream_failure_wakes_the_viewer_and_releases_the_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Propagate acquisition failure without leaving the viewer or command waiter blocked.
    """
    class FailingCamera(FakeCamera):
        """
        Model a USB disconnect before a complete frame arrives.
        """

        def frames(self) -> Iterator[bytes]:
            """
            Fail immediately at iteration time.
            """
            raise CameraError("USB disconnected")

    physical = FailingCamera()
    monkeypatch.setattr(client.subprocess, "Popen", Mock(return_value=FakeProcess(physical)))
    camera = client.MacOSCamera()
    try:
        with pytest.raises(CameraError, match="USB disconnected"):
            next(camera.frames())
    finally:
        camera.close()
    assert physical.closed.is_set()
