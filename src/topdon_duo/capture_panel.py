"""Nonblocking communication with the separate Capture controls window."""

from __future__ import annotations

import json
import os
import subprocess
import sys


class CapturePanel:
    window_module = "topdon_duo.capture_window"
    window_label = "Capture"

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._incoming = b""
        self._outgoing = b""
        self._inflight = b""
        self._stderr = b""
        self._last_state: dict | None = None

    @property
    def is_open(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def open(self, state: dict) -> None:
        if self.is_open:
            self._outgoing = json.dumps({**state, "raise_window": True}).encode() + b"\n"
            self._flush()
            return
        self.close()
        env = os.environ.copy()
        # OpenCV's Qt 5 plugin paths cannot be used by the popup's Qt 6 process.
        for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            env.pop(name, None)
        self._process = subprocess.Popen(
            [sys.executable, "-m", self.window_module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        os.set_blocking(self._process.stderr.fileno(), False)
        self._incoming = b""
        self._last_state = None
        self.update(state)

    def _flush(self) -> None:
        if not self.is_open:
            return
        if not self._inflight:
            self._inflight, self._outgoing = self._outgoing, b""
        if not self._inflight:
            return
        try:
            written = os.write(self._process.stdin.fileno(), self._inflight)
            self._inflight = self._inflight[written:]
        except BlockingIOError:
            pass
        except BrokenPipeError:
            self._outgoing = b""
            self._inflight = b""

    def update(self, state: dict) -> None:
        if not self.is_open:
            return
        state = {**state, "status": str(state.get("status", ""))[:512]}
        if state != self._last_state:
            self._last_state = state.copy()
            self._outgoing = json.dumps(state).encode() + b"\n"
        self._flush()

    def poll(self) -> list[dict]:
        if self._process is None:
            return []
        self._flush()
        while True:
            try:
                chunk = os.read(self._process.stderr.fileno(), 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            self._stderr = (self._stderr + chunk)[-16384:]
        while True:
            try:
                chunk = os.read(self._process.stdout.fileno(), 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            self._incoming += chunk
        messages = []
        while b"\n" in self._incoming:
            line, self._incoming = self._incoming.split(b"\n", 1)
            try:
                message = json.loads(line)
                if isinstance(message, dict):
                    messages.append(message)
            except (ValueError, UnicodeDecodeError):
                continue
        if self._process.poll() is not None:
            stderr = self._stderr.decode(errors="replace").strip()
            if self._process.returncode:
                messages.append(
                    {
                        "action": "error",
                        "message": stderr or f"{self.window_label} window closed unexpectedly",
                    }
                )
            self.close()
        return messages

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                # EOF lets Qt quit normally and flush persistent window settings.
                try:
                    process.stdin.close()
                except OSError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        self._outgoing = b""
        self._inflight = b""
        self._stderr = b""
        self._incoming = b""
        self._last_state = None
