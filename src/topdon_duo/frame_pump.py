"""Continuously drain USB while the desktop consumes the newest complete frame."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any, cast

from .camera import CameraError
from .camera_backends import CameraFrame, CameraProfile


class CameraFramePump:
    def __init__(self, camera: Any) -> None:
        """
        Start the single acquisition owner for a backend or legacy injected camera.
        """
        self._camera = camera
        self._condition = threading.Condition()
        self._frames = deque(maxlen=1)
        self._finished = False
        self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-acquisition", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """
        Retain complete compatible frames and forward failures to the consumer.
        """
        try:
            for frame in self._camera.frames():
                profile = getattr(self._camera, "profile", None)
                if (isinstance(profile, CameraProfile) and profile.id != "duo"
                        and (not isinstance(frame, CameraFrame) or frame.profile != profile)):
                    raise CameraError("Camera backend returned an incompatible decoded frame")
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

    def __iter__(self) -> Iterator[bytes | CameraFrame | None]:
        """
        Yield the latest frame or a heartbeat while waiting for acquisition.
        """
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

    def close(self) -> None:
        """
        Stop and join acquisition before its caller closes backend resources.
        """
        self._stop.set()
        stop = getattr(self._camera, "stop_stream", None)
        if callable(stop):
            cast(Callable[[], None], stop)()
        else:
            # noinspection PyProtectedMember
            self._camera._running.clear()  # Legacy injected camera compatibility.
        self._thread.join(timeout=self._camera.timeout_ms / 1000 + 1)
        if self._thread.is_alive():
            raise CameraError("Camera acquisition did not stop within its USB timeout")
