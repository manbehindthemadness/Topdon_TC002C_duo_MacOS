"""
Stalled viewer pipes must not stop the macOS helper's packet/frame consumer.
"""

import io
import os
import threading
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import Mock

import pytest

from based_thermals.camera import NegotiatedMode
from based_thermals.camera_backends.duo.macos_usb import protocol, service


def test_stalled_viewer_does_not_block_acquisition_or_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Drain a 600-frame burst while output is blocked and later publish the newest frame.
    """
    blocked, release, drained, delivered, closed = (threading.Event() for _ in range(5))

    class Camera:
        """
        Produce frames and diagnostic callbacks while a native viewer stall holds output.
        """

        def __init__(self) -> None:
            """
            Supply the small helper contract without USB or elevated authorization.
            """
            self.device = None
            self.timeout_ms = 2000
            self.usb_queue_depth = 32
            self.stream_observer: Callable[[dict[str, Any]], None] = Mock(
                side_effect=AssertionError("Stream observer was not installed"),
            )
            self.running = threading.Event()

        def open(self) -> NegotiatedMode:
            """
            Return a verified-style mode and preserve a usable control endpoint.
            """
            self.device = Mock(ctrl_transfer=Mock(return_value=b"ok"))
            self.running.set()
            return NegotiatedMode(1, 10, 400000, 209920, 5020)

        def frames(self) -> Iterator[bytes]:
            """
            Complete the burst independently of pipe delivery and wait for shutdown.
            """
            yield b"first"
            assert blocked.wait(3)
            observer = self.stream_observer
            for number in range(600):
                observer({"frames": number + 1})
                yield str(number).encode()
            drained.set()
            while self.running.is_set():
                closed.wait(.01)

        def stop_stream(self) -> None:
            """
            Stop the fake acquisition generator.
            """
            self.running.clear()

        def close(self) -> None:
            """
            Record that resources were released after capture stopped.
            """
            self.device = None
            closed.set()

    output = io.BytesIO()

    def send(target: Any, metadata: dict, payload: bytes = b"") -> None:
        """
        Pause the first complete frame's write until the test releases the viewer.
        """
        if metadata.get("kind") == "frame" and payload == b"first":
            blocked.set()
            assert release.wait(5)
        protocol.send(target, metadata, payload)
        if metadata.get("kind") == "frame" and payload == b"599":
            delivered.set()

    monkeypatch.setattr("based_thermals.camera_backends.duo.macos_usb.service.send", send)
    camera = Camera()
    read_fd, write_fd = os.pipe()

    def run() -> None:
        """
        Use the production helper dispatch with a real command pipe.
        """
        with os.fdopen(read_fd, "rb", buffering=0) as source:
            service.serve(source, output, camera)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    with os.fdopen(write_fd, "wb", buffering=0) as commands:
        try:
            protocol.send(commands, {"id": 1, "operation": "open", "timeout": 2000, "depth": 32})
            protocol.send(commands, {"id": 2, "operation": "start"})
            assert blocked.wait(3)
            assert drained.wait(2), "Viewer pipe stall blocked the USB consumer"
            assert not delivered.is_set()
            release.set()
            assert delivered.wait(3)
        finally:
            release.set()
    worker.join(3)
    assert not worker.is_alive() and closed.is_set()
    output.seek(0)
    frames = []
    diagnostics = []
    while output.tell() < len(output.getvalue()):
        header, body = protocol.receive(output)
        if header.get("kind") == "frame":
            frames.append(body)
        elif header.get("kind") == "diagnostics":
            diagnostics.append(header["event"])
    assert frames == [b"first", b"599"]
    assert diagnostics == [{"frames": 600}]
