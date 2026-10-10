"""
Isolate the desktop measurement renderer and image worker during live capture.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from topdon_duo.camera import decode_duo_frame
from topdon_duo.pipeline import node, validate_pipeline


class CaptureComponents:
    """
    Exercise real desktop components with a minimal source/output pipeline.
    """

    def __init__(self, mode: str) -> None:
        """
        Build only the requested components without camera-settings access.
        """
        self.renderer: Any = None
        self.worker: Any = None
        self.last_frame: bytes | None = None
        self.document = validate_pipeline({
            "version": 4, "hardware": [],
            "software": [node("software", "source"), node("software", "output")],
            "branches": {tab: [node("software", "source")] for tab in "BCD"},
        })
        if mode in ("renderer", "both"):
            from topdon_duo.render import ThermalRenderer

            self.renderer = ThermalRenderer(scale=4, rotation=270, ambient_celsius=None)
            self.renderer.native_temperatures = True
            self.renderer.set_pipeline(self.document)
        if mode in ("worker", "both"):
            from topdon_duo.processing.worker import PipelineWorker

            self.worker = PipelineWorker(apple_available=False)

    def process(self, frame: bytes | None) -> np.ndarray | None:
        """
        Match desktop fresh/held-frame processing and return a display preview.
        """
        fresh = frame is not None
        if fresh:
            self.last_frame = frame
        held = self.last_frame
        if held is None:
            return None
        averaged = None
        image_frame = held
        if self.renderer is not None:
            rendered = self.renderer.render_detailed(
                held, update_measurements=fresh, image_processing=False,
            )
            averaged = self.renderer.averaged_raw_counts
            image_frame = held if rendered.measurements_valid else self.renderer.last_valid_frame
        if self.worker is not None:
            if image_frame is not None:
                self.worker.submit(image_frame, averaged, self.document, 0, 4, 270, 1)
            result = self.worker.latest(0)
            if result is not None:
                _, image, _, _, error = result
                if error:
                    raise RuntimeError(error)
                if image is not None:
                    return image
        _, _, preview = decode_duo_frame(held)
        image = cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)
        return image

    def close(self) -> None:
        """
        Close component executors and model caches after capture stops.
        """
        if self.worker is not None:
            self.worker.close()
        if self.renderer is not None:
            self.renderer.pipeline_processor.close()
