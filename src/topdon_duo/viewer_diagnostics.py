"""Opt-in viewer stage timings and thread stacks for intermittent stalls."""

from __future__ import annotations

import faulthandler
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from .camera import FRAME_BYTES, FRAME_MAGIC


class ViewerDiagnostics:
    def __init__(self, path: Path | None):
        self._output = None
        self._stacks = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stage = "startup"
        self._since = time.monotonic()
        self._sequence = 0
        self._frames = 0
        self._status = None
        self._path = path
        self._rejected_captures = 0
        self._last_rejected_capture = float("-inf")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._output = path.open("a", buffering=1)
            self._stacks = path.with_suffix(".stacks.log").open("a", buffering=1)
            self._write("start")
            self._thread = threading.Thread(target=self._watch, name="ui-diagnostics", daemon=True)
            self._thread.start()

    def _write(self, event, **details):
        self._output.write(
            json.dumps(
                {
                    "time": datetime.now(UTC).isoformat(),
                    "event": event,
                    "stage": self._stage,
                    "frames": self._frames,
                    **details,
                }
            )
            + "\n"
        )

    def stage(self, name):
        if self._output is None:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._since
            if elapsed >= 0.25:
                self._write("slow_stage", seconds=round(elapsed, 4))
            self._stage, self._since = name, now
            self._sequence += 1

    def frames(self, frames):
        self.stage("camera_wait")
        for frame in frames:
            if self._output is not None and frame is not None:
                with self._lock:
                    self._frames += 1
            self.stage("controls")
            yield frame
            self.stage("camera_wait")

    def measurements(self, status):
        if self._output is None:
            return
        with self._lock:
            if status != self._status:
                self._status = status
                self._write("measurement_status", status=status or "valid")

    def stream(self, counters):
        if self._output is not None:
            with self._lock:
                self._write("usb_stream", **counters)

    def rejected_frame(self, frame):
        """Save a bounded sample of long rejected frames for offline inspection."""
        if (
            self._output is None
            or self._rejected_captures >= 4
            or len(frame) < FRAME_BYTES
            or frame[:4] != FRAME_MAGIC.to_bytes(4, "little")
        ):
            return
        now = time.monotonic()
        if now - self._last_rejected_capture < 30:
            return
        with self._lock:
            self._rejected_captures += 1
            self._last_rejected_capture = now
            path = self._path.with_suffix(f".rejected-{self._rejected_captures:02d}.bin")
            try:
                path.write_bytes(frame)
            except OSError as exc:
                self._write("rejected_frame_capture_error", error=str(exc))
            else:
                self._write("rejected_frame_capture", path=str(path), bytes=len(frame))

    def _watch(self):
        reported = None
        while not self._stop.wait(1):
            with self._lock:
                elapsed = time.monotonic() - self._since
                self._write("heartbeat", stage_seconds=round(elapsed, 4), status=self._status)
                if elapsed >= 2 and reported != self._sequence:
                    reported = self._sequence
                    self._write("stall", seconds=round(elapsed, 4))
                    self._stacks.write(
                        f"\n{datetime.now(UTC).isoformat()} stage={self._stage} "
                        f"seconds={elapsed:.4f}\n"
                    )
                    self._stacks.flush()
                    faulthandler.dump_traceback(file=self._stacks, all_threads=True)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._output is not None:
            with self._lock:
                self._write("stop")
                self._output.close()
                self._stacks.close()
                self._output = None
