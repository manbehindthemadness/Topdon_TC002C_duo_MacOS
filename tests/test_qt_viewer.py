"""
Qt viewer controls, transport ownership and integration without camera access.
"""

import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from support.qt_process import run_popup

from topdon_duo import desktop
from topdon_duo.capture_panel import CapturePanel
from topdon_duo.viewer_panel import ViewerPanel


def test_frame_acknowledgement_prevents_overwrite_and_cleans_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Hold frame bytes until Qt acknowledges them, preserving resize and cleanup.
    """
    monkeypatch.setattr(CapturePanel, "open", lambda _self, _state: None)
    monkeypatch.setattr(CapturePanel, "is_open", property(lambda _self: True))
    messages = []
    monkeypatch.setattr(CapturePanel, "update", lambda _self, state: messages.append(state))
    incoming = []
    monkeypatch.setattr(CapturePanel, "poll", lambda _self: incoming)
    panel = ViewerPanel()
    panel.open({})
    first = np.full((10, 20, 3), 17, np.uint8)
    second = np.full_like(first, 90)
    try:
        panel.present(first, {"status": "one"})
        descriptor = messages[-1]["frame"]
        path = Path(descriptor["path"])
        with path.open("rb") as stream:
            assert stream.read(first.nbytes) == first.tobytes()
        panel.resize((930, 710))
        panel.present(second, {"status": "two"})
        assert "frame" not in messages[-1]
        assert messages[-1]["resize"] == [930, 710]
        with path.open("rb") as stream:
            assert stream.read(first.nbytes) == first.tobytes()
        incoming.append({"action": "frame_read", "serial": descriptor["serial"] + 1})
        panel.poll()
        panel.present(second, {})
        assert "frame" not in messages[-1]
        incoming[:] = [{"action": "frame_read", "serial": descriptor["serial"]}]
        panel.poll()
        panel.present(second, {})
        assert messages[-1]["frame"]["serial"] == descriptor["serial"] + 1
        with path.open("rb") as stream:
            assert stream.read(second.nbytes) == second.tobytes()
    finally:
        panel.close()
    assert not path.exists()


def test_real_qt_bridge_acknowledges_frames_and_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Launch the actual offscreen Qt process and recycle its shared buffer repeatedly.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    panel = ViewerPanel()
    panel.open({"unit": "C"})
    try:
        for value in (20, 90, 160):
            image = np.full((100, 200, 3), value, np.uint8)
            panel.present(image, {"unit": "C"})
            deadline = time.monotonic() + 5
            while panel._pending_frame is not None and time.monotonic() < deadline:
                assert not panel.poll()
                time.sleep(0.01)
            assert panel._pending_frame is None
            assert panel.is_open
        assert panel.size is not None
        assert panel.windowed_size is not None
    finally:
        panel.close()
    assert not panel.is_open


def test_qt_session_has_no_bitmap_controls_and_samples_top_row(
    viewer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Run the actual desktop controllers with Qt input, preserving sensor coordinates.
    """
    panel = Mock(size=None, windowed_size=(768, 576), over_image=True, closed=False)
    panel.poll.side_effect = [
        [{"action": "spots"}],
        [{"action": "pointer", "kind": "release", "button": "left", "x": 0, "y": 0}],
        [{"action": "unit"}],
        [{"action": "quit"}],
    ]
    monkeypatch.setattr("topdon_duo.desktop.ViewerPanel", lambda: panel)
    named_window = Mock()
    resize_window = Mock()
    monkeypatch.setattr(desktop.cv2, "namedWindow", named_window)
    monkeypatch.setattr(desktop.cv2, "resizeWindow", resize_window)
    monkeypatch.setattr(desktop.cv2, "waitKey", Mock(side_effect=AssertionError("No HighGUI")))
    assert desktop.main([]) == 0
    assert panel.present.call_count == 3
    assert all(call.args[0].shape == (576, 768, 3) for call in panel.present.call_args_list)
    assert not viewer.draw_toolbar.called
    named_window.assert_not_called()
    assert not viewer.displayed
    resize_window.assert_not_called()
    panel.close.assert_called_once()
    saved = desktop.load_settings()
    assert saved["spots"]["items"][0]["x"] == 0
    assert saved["spots"]["items"][0]["y"] == 0
    assert saved["display"]["temperature_unit"] == "F"


def test_qt_graph_commands_keep_logging_locks(
    viewer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate stale/invalid interval changes and retain locks outside the Qt process.
    """
    session = desktop.DesktopSession(desktop, [])
    panel = Mock(closed=False)
    session.viewer = panel
    for value in (None, True, "2", float("nan"), 0, 61):
        panel.poll.return_value = [{"action": "graph_interval", "value": value}]
        assert session.poll_viewer_commands()
    viewer.graphs.set_interval.reset_mock()
    panel.poll.return_value = [{"action": "graph_interval", "value": 2.5}]
    assert session.poll_viewer_commands()
    assert session.graph_interval == 2.5
    viewer.graphs.set_interval.assert_called_once_with(2.5)
    viewer.graphs.logging = True
    viewer.graphs.clear_history.reset_mock()
    panel.poll.return_value = [
        {"action": "graph_interval", "value": 3}, {"action": "graph_reset"},
    ]
    assert session.poll_viewer_commands()
    assert session.graph_interval == 2.5
    viewer.graphs.clear_history.assert_not_called()


def test_qt_graph_display_has_no_header_hotspots(
    viewer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep graph controls out of pixels and preserve cached capture graphs with their heading.
    """
    desktop.save_settings({"show_graph": True})
    panel = Mock(size=(1536, 576), windowed_size=(1536, 576), over_image=False, closed=False)
    panel.poll.side_effect = [
        [],
        [{"action": "pointer", "kind": "release", "button": "left", "x": 1400, "y": 10}],
        [{"action": "quit"}],
    ]
    monkeypatch.setattr("topdon_duo.desktop.ViewerPanel", lambda: panel)
    cached = np.full((576, 768, 3), (10, 20, 30), np.uint8)
    cached[:34] = (80, 90, 100)
    viewer.graphs.image.side_effect = lambda _size, **_kwargs: cached
    assert desktop.main([]) == 0
    assert panel.present.call_count == 2
    for call in panel.present.call_args_list:
        image = call.args[0]
        assert image.shape == (576, 1536, 3)
        assert np.all(image[:, 768:] == (10, 20, 30))
    assert np.all(cached[:34] == (80, 90, 100))
    viewer.graphs.clear_history.assert_not_called()


def test_qt_toolbar_canvas_fullscreen_and_shared_image(tmp_path: Path) -> None:
    """
    Exercise Qt widgets, mapped pixels, input and fullscreen in an isolated process.
    """
    script = '''
import sys
import tempfile
from pathlib import Path
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from topdon_duo.viewer_panel import FRAME_CAPACITY
from topdon_duo.viewer_window import ViewerWindow
app = QApplication([])
messages = []
window = ViewerWindow(messages.append)
window.show()
app.processEvents()
state = {"unit": "F", "show_graph": True, "graph_interval": 0.5}
window.update_state(state)
window.update_state({"resize": [900, 600]})
app.processEvents()
assert [window.canvas.width(), window.canvas.height()] == [900, 600]
with tempfile.NamedTemporaryFile(dir=sys.argv[1]) as stream:
    stream.truncate(FRAME_CAPACITY)
    stream.write(bytes([10, 20, 30]) * 200 * 100)
    stream.flush()
    window.update_state({"frame": {"path": stream.name, "serial": 1, "width": 200, "height": 100}})
    assert window.canvas.image.pixelColor(0, 0).getRgb()[:3] == (30, 20, 10)
    stream.seek(0)
    stream.write(bytes([0]) * 200 * 100 * 3)
    stream.flush()
    assert window.canvas.image.pixelColor(0, 0).getRgb()[:3] == (30, 20, 10)
assert {"action": "frame_read", "serial": 1} in messages
rectangle = window.canvas.image_rect()
assert window.canvas.image_position(QPointF(rectangle.left(), rectangle.top())) == (0, 0)
assert window.canvas.image_position(QPointF(0, 0)) is None
assert window.canvas.image_position(rectangle.center()) == (100, 50)
messages.clear()
QTest.mouseMove(window.canvas, rectangle.center().toPoint())
QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=rectangle.topLeft().toPoint())
assert any(m.get("action") == "pointer" and m.get("x") == 0 and m.get("y") == 0 for m in messages)
window.toolbar.buttons["rotate"].click()
assert {"action": "rotate"} in messages
window.toolbar.interval.setValue(2)
assert {"action": "graph_interval", "value": 2.0} in messages
window.update_state({**state, "settings_locked": True, "spots_locked": True, "graph_locked": True})
for action in ("rotate", "unit", "spots", "graph", "graph_reset"):
    assert not window.toolbar.buttons[action].isEnabled()
assert not window.toolbar.interval.isEnabled()
assert window.toolbar.buttons["graph_log"].isEnabled()
messages.clear()
window.update_state(state)
assert not messages
window.canvas.setFocus()
QTest.keyClick(window.canvas, Qt.Key.Key_O)
assert {"action": "key", "key": ord("o")} in messages
before = window.size()
window.toolbar.buttons["fullscreen"].click()
app.processEvents()
assert window.isFullScreen()
messages.clear()
QTest.keyClick(window.canvas, Qt.Key.Key_Escape)
app.processEvents()
assert not window.isFullScreen()
assert window.size() == before
assert not any(m.get("action") == "quit" for m in messages)
QTest.keyClick(window.canvas, Qt.Key.Key_Escape)
assert {"action": "quit"} in messages
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


__all__ = ["viewer_fixture"]
