"""
Run the full viewer without camera-settings or calibration control transfers.

Close existing viewers first. Pass normal desktop arguments to this launcher.
Normal UVC negotiation and USB capture remain enabled. Subsequent control
transfers are rejected locally, and camera-settings initialization is skipped.
This isolates desktop rendering from settings traffic; keep the minimal pipeline
from the previous trial. USB permissions and the project environment are required.
No network, extra dependencies, or production application changes are involved.
Use --no-graphs to isolate the graph display path and --seconds 600 for a bounded
trial that also stops after sustained frame loss. Diagnostic runs do not save
desktop preferences or main-window geometry.
Use --python-switch-interval 0.001 to test shorter interpreter thread handoffs
with live charts; the process-wide interval is restored when the trial exits.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

import cv2
import numpy as np

from topdon_duo import desktop
from topdon_duo import graphs as graph_module
from topdon_duo.camera import CameraError, NegotiatedMode, TC002CDuoCamera
from topdon_duo.desktop_app.session import DesktopSession


class ChartsWithoutMarkers:
    """
    Forward chart drawing calls while omitting individual sample circles.
    """

    def __getattr__(self, name: str) -> Any:
        """
        Preserve every other OpenCV operation within the graph module.
        """
        return getattr(cv2, name)

    def circle(self, *args: Any, **kwargs: Any) -> None:
        """
        Skip only the historical point markers, retaining curve rendering.
        """


def cached_chart_renderer(
    render: Callable[..., np.ndarray],
) -> Callable[..., np.ndarray]:
    """
    Render once per pane size while retaining normal history and pane composition.
    """
    images: dict[tuple[int, int], np.ndarray] = {}

    def cached_render(
        snapshot: graph_module.GraphSnapshot, master: Any, spots: Any, now: float,
    ) -> np.ndarray:
        """
        Reuse the first chart image until the pane dimensions change.
        """
        size = snapshot.size
        if size not in images:
            images[size] = render(snapshot, master, spots, now)
        return images[size]

    return cached_render


def reject_control(*args: Any, **kwargs: Any) -> NoReturn:
    """
    Prevent settings reads, writes and calibration commands during this trial.
    """
    raise CameraError("Camera control transfers disabled for this diagnostic run")


class CaptureOnlyDesktop(DesktopSession):
    """
    Preserve the normal viewer while omitting its hardware-settings handshake.
    """

    def initialize_hardware(self) -> bool:
        """
        Leave camera settings untouched and prevent startup calibration.
        """
        self.calibration_available = False
        self.startup_calibration_pending = False
        self.hardware.error = "Camera controls disabled for this diagnostic run"
        return True


def main() -> int:
    """
    Negotiate normally, block later controls, and run the diagnostic desktop.
    """
    parser = argparse.ArgumentParser(add_help=False)
    graphs = parser.add_mutually_exclusive_group()
    graphs.add_argument("--no-graphs", action="store_true")
    graphs.add_argument("--cached-graph", action="store_true")
    graphs.add_argument("--no-chart-markers", action="store_true")
    parser.add_argument("--python-switch-interval", type=float)
    parser.add_argument("--seconds", type=int)
    options, desktop_args = parser.parse_known_args()
    if options.seconds is not None and options.seconds <= 0:
        parser.error("--seconds must be positive")
    if options.python_switch_interval is not None and (
        not math.isfinite(options.python_switch_interval)
        or not 0.0001 <= options.python_switch_interval <= 0.1
    ):
        parser.error("--python-switch-interval must be between 0.0001 and 0.1 seconds")
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = process.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(arg.decode(errors="replace")).name == "topdon-duo-desktop"
               for arg in command if arg):
            raise SystemExit("Close the desktop viewer before running this diagnostic")
    original_open = TC002CDuoCamera.open
    original_switch_interval = sys.getswitchinterval()

    def open_without_controls(camera: TC002CDuoCamera) -> NegotiatedMode:
        """
        Block control traffic only after the normal UVC setup has completed.
        """
        mode = original_open(camera)
        camera.device.ctrl_transfer = reject_control
        return mode

    with ExitStack() as patches:
        if options.python_switch_interval is not None:
            sys.setswitchinterval(options.python_switch_interval)
            patches.callback(sys.setswitchinterval, original_switch_interval)
            print(f"Python switch interval: {sys.getswitchinterval():g} seconds", flush=True)
        patches.enter_context(patch.object(TC002CDuoCamera, "open", open_without_controls))
        patches.enter_context(patch.object(desktop, "save_settings", lambda _settings: None))
        patches.enter_context(patch.object(desktop, "save_main_window_size", lambda _size: None))
        if options.cached_graph:
            patches.enter_context(patch.object(
                graph_module, "render_graphs", cached_chart_renderer(graph_module.render_graphs),
            ))
        if options.no_chart_markers:
            patches.enter_context(patch.object(graph_module, "cv2", ChartsWithoutMarkers()))
        session = CaptureOnlyDesktop(desktop, desktop_args)
        if options.no_graphs:
            session.show_graph = False
        elif options.cached_graph or options.no_chart_markers:
            session.show_graph = True
        if options.seconds is not None:
            started = time.monotonic()
            original_tick = session.tick

            def bounded_tick() -> bool:
                """
                Stop the trial at its deadline or after five seconds of frame loss.
                """
                now = time.monotonic()
                if now - started >= options.seconds:
                    return False
                if session.last_frame_at is not None and now - session.last_frame_at >= 5:
                    return False
                result = original_tick()
                return result

            session.tick = bounded_tick
        result = session.run()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
