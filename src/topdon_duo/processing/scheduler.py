"""
Schedule branch dependencies and assemble viewer/preview results.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

import cv2
import numpy as np

from ..apple_acceleration import apple_acceleration
from ..camera_backends import decode_frame
from ..custom_nodes.labels import draw_display_labels
from ..nvidia_acceleration import nvidia_acceleration
from ..pipeline import (
    active_nodes,
    execution_dependencies,
    preview_roots,
    software_tabs,
    validate_pipeline,
)
from .branch import BranchProcessor
from .images import (
    bytes_image,
    resize,
)


class PipelineProcessor:
    def __init__(
        self,
        apple_available: bool | None = None,
        nvidia_available: bool | None = None,
    ) -> None:
        """
        Init.
        """
        self.apple_available = (
            apple_acceleration()["available"] if apple_available is None else apple_available
        )
        self.nvidia_available = (
            nvidia_acceleration()["available"] if nvidia_available is None else nvidia_available
        )
        self.models = {}
        self.coreml_models = {}
        self.onnx_models = {}
        self.executors = {}
        self.branches = {}
        self.last_previews = {}
        self.last_preview_timings = {}
        self.collect_previews = False
        self.preview_active = False
        self.last_preview_errors = {}

    def close(self) -> None:
        """
        Close.
        """
        for branch in self.branches.values():
            branch.close()
        for executor in self.executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
        self.executors.clear()
        self.branches.clear()
        self.models = {}
        self.coreml_models = {}
        self.onnx_models = {}
        self.last_previews = {}
        self.last_preview_timings = {}

    def process(
        self,
        frame: Any,
        averaged: Any,
        document: Any,
        scale: int = 3,
        rotation: int = 0,
        camera_palette: Any = 1,
    ) -> Any:
        """
        Process.
        """
        self.last_previews = {}
        self.last_preview_timings = {}
        self.last_preview_errors = {}
        frame = decode_frame(frame)
        document = validate_pipeline(document, hardware_profile=frame.profile.id)
        viewer_dependencies = execution_dependencies(document)
        roots = preview_roots(document) if self.preview_active or self.collect_previews else ()
        retained_dependencies = execution_dependencies(document, roots=("A", *roots))
        dependencies = retained_dependencies if self.collect_previews else viewer_dependencies
        tabs = software_tabs(document)
        # Stop idle branch threads and release their model caches after disconnection.
        for tab in set(self.executors) - set(retained_dependencies):
            self.executors.pop(tab).shutdown(wait=True)
            self.branches.pop(tab).close()
        for tab in dependencies:
            if tab not in self.executors:
                self.executors[tab] = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"pipeline-{tab}"
                )
                self.branches[tab] = BranchProcessor(
                    apple_available=self.apple_available,
                    nvidia_available=self.nvidia_available,
                )
        outputs, pending, submitted, errors = {}, {}, set(), {}
        try:
            while len(outputs) < len(dependencies):
                for tab, required in dependencies.items():
                    if tab not in submitted and required <= outputs.keys():
                        failed_inputs = [t for t in required if outputs[t][0] is None]
                        if failed_inputs:
                            errors[tab] = (
                                f"Input tab {failed_inputs[0]} failed: {errors[failed_inputs[0]]}"
                            )
                            outputs[tab] = None, ""
                            submitted.add(tab)
                            continue
                        branch = {"hardware": document["hardware"], "software": tabs[tab]}
                        inputs = {t: outputs[t][0] for t in required}
                        self.branches[tab].collect_previews = self.collect_previews
                        self.branches[tab].input_labels = {
                            t: list(self.branches[t].custom.labels) for t in required
                        }
                        future = self.executors[tab].submit(
                            self.branches[tab].process,
                            frame,
                            averaged,
                            branch,
                            camera_palette,
                            inputs,
                        )
                        pending[future] = tab
                        submitted.add(tab)
                if not pending and len(outputs) == len(dependencies):
                    break
                if not pending:
                    raise ValueError("Unresolved combine connections")
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    tab = pending.pop(future)
                    try:
                        outputs[tab] = future.result()
                    except (ValueError, TypeError, KeyError, cv2.error, MemoryError) as exc:
                        if tab in viewer_dependencies:
                            raise
                        outputs[tab] = None, ""
                        errors[tab] = str(exc)
        finally:
            # Drain this frame before processing another; never mutate a branch model concurrently.
            if pending:
                wait(pending)
        self.last_previews = {
            identity: payload
            for tab in dependencies
            if outputs[tab][0] is not None
            for identity, payload in self.branches[tab].last_previews.items()
        }
        self.last_preview_timings = {
            identity: elapsed
            for tab in dependencies
            if outputs[tab][0] is not None
            for identity, elapsed in self.branches[tab].last_preview_timings.items()
        }
        self.last_preview_errors = {
            n["id"]: errors[tab]
            for tab in errors
            for n in tabs[tab]
            if n["type"] == "preview" and n["expanded"] and not n["bypass"]
        }
        self.models = self.branches["A"].models
        self.coreml_models = self.branches["A"].coreml_models
        self.onnx_models = self.branches["A"].onnx_models
        image, source = outputs["A"]
        if rotation:
            image = np.rot90(image, -(rotation // 90)).copy()
        width, height = frame.profile.native_size
        target: tuple[int, int] = (width * scale, height * scale)
        if rotation in (90, 270):
            target = target[1], target[0]
        interpolation = (
            cv2.INTER_CUBIC
            if any(n["type"] == "antialiasing" for n in active_nodes(document, "software"))
            else cv2.INTER_NEAREST
        )
        display = bytes_image(resize(image, target, interpolation))
        display = draw_display_labels(display, self.branches["A"].custom.labels, rotation)
        return display, source
