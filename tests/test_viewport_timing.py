"""Frame-linked viewport timing and compact status-bar display."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.capture_panel import CapturePanel
from based_thermals.viewer_panel import ViewerPanel


def test_render_timing_excludes_saving_and_input(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Accumulate render/overlay time without capture operations or interaction delays.
    """
    del viewer
    session = desktop.DesktopSession(desktop, [])
    clock = [0.0]

    def advance(seconds: float) -> None:
        """
        Advance a deterministic clock only while the simulated stage is running.
        """
        clock[0] += seconds

    monkeypatch.setattr(desktop.time, "perf_counter", lambda: clock[0])
    # Controller methods are inherited and are intentionally replaced at this test boundary.
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "poll_viewer_commands", lambda: True)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "receive_frame", lambda: None)
    for method in ("poll_spot_and_graph_commands", "poll_camera_commands", "poll_capture_commands"):
        monkeypatch.setattr(session, method, lambda: None)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "render_frame", lambda: advance(0.002))
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "finish_capture", lambda: advance(10))
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "draw_status", lambda: advance(0.003))

    def clicks() -> bool:
        """
        Model slow input handling outside the viewport drawing stages.
        """
        advance(20)
        return True

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "handle_clicks", clicks)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(session, "present_and_handle_keys", lambda: True)
    assert session.tick()
    assert session.viewport_draw_ms == pytest.approx(5.0)


def test_draw_time_is_attached_to_the_published_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A pending frame retains its own timing even when newer state arrives.
    """
    messages: list[dict] = []
    monkeypatch.setattr(CapturePanel, "open", lambda _self, _state: None)
    monkeypatch.setattr(CapturePanel, "is_open", property(lambda _self: True))
    monkeypatch.setattr(CapturePanel, "update", lambda _self, state: messages.append(state))
    panel = ViewerPanel()
    panel.open({})
    try:
        image = np.zeros((10, 20, 3), np.uint8)
        panel.present(image, {"viewport_draw_ms": 12.5})
        descriptor = messages[-1]["frame"]
        assert descriptor["viewport_draw_ms"] == 12.5
        assert descriptor["viewport_copy_ms"] >= 0
        panel.present(image, {"viewport_draw_ms": 90})
        assert "frame" not in messages[-1]
        assert descriptor["viewport_draw_ms"] == 12.5
    finally:
        panel.close()


def test_status_bar_combines_frame_copy_paint_and_preparation(tmp_path: Path) -> None:
    """
    Show a numeric total beside status, preserving frame matching and unavailable values.
    """
    script = r"""
import tempfile
from PySide6.QtWidgets import QApplication
from based_thermals import viewer_window
from based_thermals.viewer_ui import canvas
from based_thermals.viewer_panel import FRAME_CAPACITY

app = QApplication([])
window = viewer_window.ViewerWindow(lambda _message: None)
window.show()
app.processEvents()
window.update_state({"unit": "C", "status": "Ready"})
assert window.statusBar().currentMessage() == "Ready | -- ms"
for value in (None, True, "5", -1, float("nan"), float("inf"), 10**1000):
    assert viewer_window.validated_draw_time(value) is None
assert viewer_window.validated_draw_time(0) == 0
with tempfile.NamedTemporaryFile() as stream:
    stream.truncate(FRAME_CAPACITY)
    stream.write(bytes([10, 20, 30]) * 20 * 10)
    stream.flush()
    stamps = iter((1.0, 1.002))
    viewer_window.perf_counter = lambda: next(stamps)
    window.update_state({"frame": {"path": stream.name, "serial": 1,
                         "width": 20, "height": 10, "viewport_draw_ms": 5.0,
                         "viewport_copy_ms": 1.5}})
    stamps = iter((2.0, 2.003))
    canvas.perf_counter = lambda: next(stamps)
    window.canvas.repaint()
    assert window.statusBar().currentMessage() == "Ready | 11.5 ms"
    window.update_state({"unit": "C", "status": "Recording", "viewport_draw_ms": 999})
    assert window.statusBar().currentMessage() == "Recording | 11.5 ms"
    stamps = iter((3.0, 3.002))
    viewer_window.perf_counter = lambda: next(stamps)
    window.update_state({"frame": {"path": stream.name, "serial": 2,
                         "width": 20, "height": 10, "viewport_draw_ms": 20.0}})
    stamps = iter((4.0, 4.003))
    canvas.perf_counter = lambda: next(stamps)
    window.canvas.repaint()
    assert window.statusBar().currentMessage() == "Recording | 25.0 ms"
import time
viewer_window.perf_counter = time.perf_counter
canvas.perf_counter = time.perf_counter
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
