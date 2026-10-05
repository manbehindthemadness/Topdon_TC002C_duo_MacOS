"""Continuously drain USB while the desktop consumes the newest complete frame."""

from __future__ import annotations

import threading
from collections import deque

from .camera import CameraError


class CameraFramePump:
    def __init__(self, camera):
        self._camera = camera
        self._condition = threading.Condition()
        self._frames = deque(maxlen=1)
        self._finished = False
        self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-acquisition", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for frame in self._camera.frames():
                if self._stop.is_set():
                    break
                with self._condition:
                    self._frames.append(frame)
                    self._condition.notify()
        except Exception as exc:  # noqa: BLE001 - forward thread failures to the viewer
            if not self._stop.is_set():
                self._error = exc
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def __iter__(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._frames or self._finished, timeout=0.05)
                if self._frames:
                    frame = self._frames.popleft()
                elif self._finished:
                    if self._error is not None:
                        raise CameraError(
                            f"Camera acquisition failed: {self._error}"
                        ) from self._error
                    return
                else:
                    frame = None
            yield frame

    def close(self):
        self._stop.set()
        self._camera._running.clear()
        self._thread.join(timeout=self._camera.timeout_ms / 1000 + 1)
        if self._thread.is_alive():
            raise CameraError("Camera acquisition did not stop within its USB timeout")
