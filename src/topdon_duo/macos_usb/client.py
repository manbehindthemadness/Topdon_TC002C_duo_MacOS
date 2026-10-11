"""
User-process camera facade for the terminal-authorized macOS USB helper.
"""

import os
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Iterator
from typing import Any, BinaryIO, cast

import usb.core

from ..camera import CameraAccessError, CameraError, NegotiatedMode, TC002CDuoCamera
from .protocol import receive, send


def helper_command() -> list[str]:
    """
    Authorize the configured interpreter; isolated mode ignores Python environment overrides.

    sudo reads authorization from the controlling terminal, never the protocol pipe.
    """
    return ["/usr/bin/sudo", "--", sys.executable, "-I", "-B", "-m",
            "topdon_duo.macos_usb.service"]


class ControlDevice:
    """
    Preserve the audited HardwareControls API without exposing a user-process USB handle.
    """

    def __init__(self, camera: "MacOSCamera") -> None:
        """
        Bind this proxy to one helper session.
        """
        self.camera = camera

    def ctrl_transfer(
        self, request_type: int, request: int, value: int, index: int,
        data: Any, timeout: int = 2000,
    ) -> int | bytes:
        """
        Send bounded control data and reconstruct USB failures for existing callers.
        """
        reading = bool(request_type & 0x80)
        payload = b"" if reading else bytes(data)
        metadata, response = self.camera.request(
            "transfer", payload, request_type=request_type, request=request,
            value=value, index=index, length=data if reading else len(payload), timeout=timeout,
        )
        return response if reading else metadata["written"]


class MacOSCamera(TC002CDuoCamera):
    """
    Retain only the newest helper frame while processing and controls stay unprivileged.
    """

    def __init__(self, timeout_ms: int = 2000) -> None:
        """
        Create synchronization state without authorizing or opening the physical camera.
        """
        super().__init__(timeout_ms)
        self.device: ControlDevice | None = None
        self.mode: NegotiatedMode | None = None
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._rpc_lock = threading.Lock()
        self._condition = threading.Condition()
        self._frames: deque[bytes] = deque(maxlen=1)
        self._replies: queue.Queue = queue.Queue(maxsize=1)
        self._identity = 0
        self._error: str | None = None
        self._closed = True

    def _read_messages(self) -> None:
        """
        Drain the helper continuously, independently of rendering or control waits.
        """
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                metadata, payload = receive(cast(BinaryIO, process.stdout))
                kind = metadata.get("kind")
                if kind == "reply":
                    self._replies.put_nowait((metadata, payload))
                elif kind == "frame":
                    with self._condition:
                        self._frames.append(payload)
                        self._condition.notify_all()
                elif kind == "diagnostics":
                    callback = self.stream_observer
                    if callback is not None:
                        event = metadata.get("event")
                        if not isinstance(event, dict):
                            raise ValueError("Invalid USB helper diagnostics")
                        callback(event)
                elif kind == "failure":
                    raise CameraError(metadata.get("error", "USB capture failed"))
                else:
                    raise ValueError("Unknown USB helper response")
        except (EOFError, OSError, ValueError, TypeError, CameraError, queue.Full) as exc:
            with self._condition:
                if not self._closed:
                    self._error = (
                        f"{exc}. USB authorization runs in the terminal; "
                        "run sudo -v there and launch the viewer without sudo."
                    )
                self._condition.notify_all()
            try:
                self._replies.put_nowait(({"error": self._error or "USB helper closed"}, b""))
            except queue.Full:
                pass  # An outstanding reply already wakes the command waiter.

    def request(self, operation: str, payload: bytes = b"", **fields: Any) -> tuple[dict, bytes]:
        """
        Serialize commands and reject mismatched or failed replies without sharing USB handles.
        """
        with self._rpc_lock:
            process = self._process
            if process is None or process.stdin is None or self._error:
                raise CameraError(self._error or "USB helper is not running")
            self._identity += 1
            identity = self._identity
            try:
                send(cast(BinaryIO, process.stdin),
                     {"id": identity, "operation": operation, **fields}, payload)
                metadata, response = self._replies.get(timeout=90 if operation == "open" else 5)
            except (OSError, ValueError, queue.Empty) as exc:
                with self._condition:
                    self._error = f"USB helper {operation} failed: {exc}"
                    self._condition.notify_all()
                raise CameraError(self._error) from exc
            if "error" in metadata:
                if operation == "transfer":
                    raise usb.core.USBError(metadata["error"], errno=metadata.get("errno"))
                raise CameraAccessError(metadata["error"])
            if metadata.get("id") != identity:
                self._error = "USB helper reply does not match the command"
                raise CameraError(self._error)
            return metadata, response

    def open(self) -> NegotiatedMode:
        """
        Start one authorized capture process; the calling application keeps its current UID.
        """
        if self.device is not None and self.mode is not None:
            return self.mode
        self._error = None
        self._frames.clear()
        self._replies = queue.Queue(maxsize=1)
        self._closed = False
        try:
            self._process = subprocess.Popen(
                helper_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, bufsize=0, cwd="/", env={"PATH": "/usr/bin:/bin"},
            )
            reader = threading.Thread(
                target=self._read_messages, name="usb-helper-reader", daemon=True,
            )
            self._reader = reader
            reader.start()
            metadata, _ = self.request("open", timeout=self.timeout_ms, depth=self.usb_queue_depth)
            mode_data = metadata.get("mode")
            if not isinstance(mode_data, dict):
                raise CameraError("USB helper returned invalid negotiation metadata")
            mode = NegotiatedMode(**mode_data)
            self.mode = mode
            self.device = ControlDevice(self)
            self._running.set()
            return mode
        except (CameraError, OSError, TypeError, KeyError):
            self.close()
            raise

    def frames(self) -> Iterator[bytes]:
        """
        Yield unchanged complete frames from the original queued USB capture path.
        """
        self.open()
        self.request("start")
        while self._running.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frames or self._error or not self._running.is_set(),
                    timeout=self.timeout_ms / 1000,
                )
                if self._error:
                    raise CameraError(self._error)
                frame = self._frames.pop() if self._frames else None
            if frame is not None and self._running.is_set():
                yield frame

    def stop_stream(self) -> None:
        """
        Wake the user acquisition thread without discarding the control connection.
        """
        self._running.clear()
        with self._condition:
            self._condition.notify_all()
        if self._process is not None and not self._closed and not self._error:
            self.request("stop")

    def close(self) -> None:
        """
        End the pipe session so the helper drains transfers and reattaches camera drivers.

        HardwareControls restoration is performed by the existing session before close.
        """
        self._running.clear()
        self._closed = True
        with self._condition:
            self._condition.notify_all()
        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=max(15, self.timeout_ms / 1000 + 3))
            except subprocess.TimeoutExpired as exc:
                raise CameraError("USB helper did not release the camera; reconnect it") from exc
            finally:
                if self._reader is not None:
                    self._reader.join(timeout=2)
                if process.stdout is not None:
                    process.stdout.close()
                self._process = None
        self.device = None
        self.mode = None
        self._frames.clear()


def use_helper(factory: Any) -> Any:
    """
    Select the helper only for the real Duo on macOS; preserve injected and other backends.
    """
    if sys.platform == "darwin" and os.geteuid() != 0 and factory is TC002CDuoCamera:
        return MacOSCamera
    return factory
