"""Timed MP4 capture of the thermal view, including timelapse sampling."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import FRAME_RATE


class VideoRecorder:
    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.mode: str | None = None
        self.fps = float(FRAME_RATE)
        self.frames_per_minute = 60.0
        self.frames_written = 0
        self._size = (0, 0)
        self._started_at = 0.0
        self._next_sample_at = 0.0
        self._last_frame: np.ndarray | None = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(
        self,
        path: Path,
        image_shape: tuple[int, ...],
        mode: str,
        *,
        frames_per_minute: float = 60.0,
        now: float | None = None,
    ) -> None:
        if self.is_recording:
            raise RuntimeError("A recording is already running")
        if mode not in ("video", "timelapse"):
            raise ValueError("Recording mode must be video or timelapse")
        if not math.isfinite(frames_per_minute) or not 1 <= frames_per_minute <= FRAME_RATE * 60:
            raise ValueError(
                f"Timelapse rate must be between 1 and {FRAME_RATE * 60} frames/minute"
            )
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".mp4")
            # The save panel did not confirm replacing this suffixed filename.
            if path.exists():
                raise FileExistsError(f"Choose the existing .mp4 filename to replace {path}")
        elif path.suffix.lower() != ".mp4":
            raise ValueError("Choose a filename ending in .mp4")
        size = (image_shape[1], image_shape[0])
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Could not open MP4 recording: {path}")
        self._writer = writer
        self.path = path
        self.mode = mode
        self.frames_per_minute = frames_per_minute
        self.frames_written = 0
        self._size = size
        self._started_at = time.monotonic() if now is None else now
        self._next_sample_at = self._started_at
        self._last_frame = None

    def _fit_frame(self, image: np.ndarray) -> np.ndarray:
        """Keep the file's dimensions fixed when the viewer is rotated."""
        width, height = self._size
        if image.shape[:2] == (height, width):
            return image
        ratio = min(width / image.shape[1], height / image.shape[0])
        resized_width = max(1, round(image.shape[1] * ratio))
        resized_height = max(1, round(image.shape[0] * ratio))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x, y = (width - resized_width) // 2, (height - resized_height) // 2
        frame[y : y + resized_height, x : x + resized_width] = resized
        return frame

    def write(self, image: np.ndarray, *, now: float | None = None) -> int:
        """Write due frames; video follows elapsed time, timelapse samples at intervals."""
        if self._writer is None:
            return 0
        now = time.monotonic() if now is None else now
        if self.mode == "timelapse":
            if now + 1e-7 < self._next_sample_at:
                return 0
            # Skip missed samples rather than filling a timelapse with duplicates.
            interval_seconds = 60.0 / self.frames_per_minute
            slot = math.floor(max(0.0, now - self._started_at) / interval_seconds + 1e-7)
            self._next_sample_at = self._started_at + (slot + 1) * interval_seconds
            count = 1
        else:
            target_count = math.floor(max(0.0, now - self._started_at) * self.fps + 1e-7) + 1
            count = max(0, target_count - self.frames_written)
            frame = self._fit_frame(image).copy()
            for _ in range(max(0, count - 1)):
                self._writer.write(self._last_frame if self._last_frame is not None else frame)
                self.frames_written += 1
            self._last_frame = frame
        if count:
            frame = self._fit_frame(image) if self.mode == "timelapse" else self._last_frame
            self._writer.write(frame)
            self.frames_written += 1
        return count

    def elapsed_seconds(self, *, now: float | None = None) -> float:
        if not self.is_recording:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, now - self._started_at)

    def stop(self) -> Path | None:
        writer, self._writer = self._writer, None
        if writer is None:
            return None
        try:
            writer.release()
        finally:
            self.mode = None
            self._last_frame = None
        return self.path
