import threading
from types import SimpleNamespace

import pytest

from topdon_duo.camera import CameraError
from topdon_duo.frame_pump import CameraFramePump


def camera(frames):
    running = threading.Event()
    running.set()
    return SimpleNamespace(frames=frames, _running=running, timeout_ms=100)


def test_capture_keeps_draining_without_a_consumer_and_retains_newest_frame():
    complete = threading.Event()
    captured = []

    def frames():
        for value in range(50):
            captured.append(value)
            yield bytes([value])
        complete.set()

    pump = CameraFramePump(camera(frames))
    try:
        assert complete.wait(2)
        assert captured == list(range(50))
        assert list(pump) == [bytes([49])]
    finally:
        pump.close()


def test_waiting_camera_yields_heartbeat_and_shuts_down():
    holder = camera(None)

    def frames():
        while holder._running.is_set():
            threading.Event().wait(0.01)
        return
        yield  # Make this a generator without producing frames.

    holder.frames = frames
    pump = CameraFramePump(holder)
    try:
        assert next(iter(pump)) is None
    finally:
        pump.close()
    assert not pump._thread.is_alive()


def test_acquisition_errors_reach_viewer():
    def frames():
        raise CameraError("USB disconnected")
        yield

    pump = CameraFramePump(camera(frames))
    try:
        with pytest.raises(CameraError, match="USB disconnected"):
            next(iter(pump))
    finally:
        pump.close()
