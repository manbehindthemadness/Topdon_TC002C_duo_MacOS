"""
Deliver latest-frame results asynchronously with revision guards.
"""

from __future__ import annotations

from copy import deepcopy
from threading import Condition, Thread
from time import perf_counter
from typing import Any

import cv2

from ..pipeline import (
    thermal_source,
)
from .scheduler import PipelineProcessor


class PipelineWorker:
    """Single image worker with an overwritten pending slot and revision guard."""

    def __init__(self, apple_available: Any = None) -> None:
        """
        Init.
        """
        self.condition = Condition()
        self.pending = None
        self.result = None
        self.preview_result = None
        self.preview_errors = {}
        self.preview_timings = {}
        self.preview_enabled = False
        self.last_preview_at = 0.0
        self.closed = False
        self.processor = PipelineProcessor(apple_available=apple_available)
        self.thread = Thread(target=self._run, name="image-pipeline", daemon=True)
        self.thread.start()

    def submit(
        self,
        frame: Any,
        averaged: Any,
        document: Any,
        revision: Any,
        scale: Any,
        rotation: Any,
        camera_palette: Any,
    ) -> None:
        """
        Submit.
        """
        with self.condition:
            self.pending = (
                frame,
                None if averaged is None else averaged.copy(),
                deepcopy(document),
                revision,
                scale,
                rotation,
                camera_palette,
            )
            self.condition.notify()

    def latest(self, revision: Any) -> Any:
        """
        Latest.
        """
        with self.condition:
            return self.result if self.result is not None and self.result[0] == revision else None

    def enable_previews(self, enabled: Any) -> None:
        """
        Enable previews.
        """
        with self.condition:
            self.preview_enabled = bool(enabled)
            if not enabled:
                self.preview_result = None
                self.preview_errors = {}
                self.preview_timings = {}

    def latest_previews(self, revision: Any) -> Any:
        """
        Latest previews.
        """
        with self.condition:
            if self.preview_result is not None and self.preview_result[0] == revision:
                return self.preview_result[1]
            return {}

    def latest_preview_timings(self, revision: Any) -> Any:
        """
        Latest preview timings.
        """
        with self.condition:
            return (
                self.preview_timings
                if self.preview_result is not None and self.preview_result[0] == revision
                else {}
            )

    def latest_preview_errors(self, revision: Any) -> Any:
        """
        Latest preview errors.
        """
        with self.condition:
            return (
                self.preview_errors
                if self.preview_result is not None and self.preview_result[0] == revision
                else {}
            )

    def _run(self) -> None:
        """
        Run.
        """
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    self.processor.close()
                    return
                frame, averaged, document, revision, scale, rotation, palette = self.pending
                self.pending = None
                make_preview = self.preview_enabled and (
                    perf_counter() - self.last_preview_at >= 0.5
                    or self.preview_result is None
                    or self.preview_result[0] != revision
                )
                self.processor.collect_previews = make_preview
                self.processor.preview_active = self.preview_enabled
            started = perf_counter()
            try:
                image, source = self.processor.process(
                    frame, averaged, document, scale, rotation, palette
                )
                result = revision, image, source, (perf_counter() - started) * 1000, ""
                previews = self.processor.last_previews if make_preview else None
                self.processor.last_previews = {}
            except (ValueError, TypeError, KeyError, cv2.error, MemoryError) as exc:
                previews = {}
                result = (
                    revision,
                    None,
                    "raw" if thermal_source(document) else "preview",
                    (perf_counter() - started) * 1000,
                    str(exc),
                )
            with self.condition:
                self.result = result
                if previews is not None and self.preview_enabled:
                    self.preview_result = revision, previews
                    self.preview_errors = self.processor.last_preview_errors.copy()
                    self.preview_timings = (
                        self.processor.last_preview_timings.copy() if previews else {}
                    )
                    self.last_preview_at = perf_counter()

    def close(self) -> None:
        """
        Close.
        """
        with self.condition:
            self.closed = True
            self.preview_enabled = False
            self.pending = None
            self.preview_result = None
            self.preview_errors = {}
            self.preview_timings = {}
            self.condition.notify()
        self.thread.join(timeout=1)
