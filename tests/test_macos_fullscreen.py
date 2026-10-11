"""
Escape exits native macOS fullscreen while preserving ordinary quit behavior.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture

from topdon_duo import desktop, macos_window


@pytest.mark.parametrize("fullscreen", [True, False])
def test_native_exit_targets_only_named_viewer(
    monkeypatch: pytest.MonkeyPatch, fullscreen: bool
) -> None:
    """
    Detect the native style mask even when another fullscreen window comes first.
    """
    monkeypatch.setattr(macos_window.sys, "platform", "darwin")
    windows = [SimpleNamespace(title=b"other", mask=macos_window.FULLSCREEN_MASK),
               SimpleNamespace(title=b"viewer", mask=macos_window.FULLSCREEN_MASK if fullscreen else 0)]

    def object_message(receiver: Any, selector: str) -> Any:
        """
        Simulate application, window-list and title object messages.
        """
        if selector == "sharedApplication":
            return "application"
        if selector == "windows":
            return windows
        return receiver.title

    def integer_message(receiver: Any, selector: str) -> int:
        """
        Return native collection length or the current window style.
        """
        return len(receiver) if selector == "count" else receiver.mask

    api = SimpleNamespace(
        library=Mock(), selector=lambda name: name, object=object_message,
        integer=integer_message, item=lambda items, _selector, index: items[index],
        string=lambda title, _selector: title, action=Mock(),
    )
    monkeypatch.setattr(macos_window, "_cocoa_api", lambda: api)
    assert macos_window.exit_fullscreen("viewer") is fullscreen
    if fullscreen:
        api.action.assert_called_once_with(windows[1], "toggleFullScreen:", None)
    else:
        api.action.assert_not_called()
    assert not macos_window.exit_fullscreen("missing")


def test_linux_does_not_load_cocoa(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Leave Linux's window backend untouched.
    """
    monkeypatch.setattr(macos_window.sys, "platform", "linux")
    loader = Mock(side_effect=AssertionError("Cocoa must not load"))
    monkeypatch.setattr(macos_window, "_cocoa_api", loader)
    assert not macos_window.exit_fullscreen("viewer")
    loader.assert_not_called()


@pytest.mark.parametrize("platform,handled,frames", [
    ("darwin", True, 2), ("darwin", False, 1), ("linux", True, 1),
])
def test_escape_exits_fullscreen_before_quitting(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, platform: str, handled: bool, frames: int
) -> None:
    """
    Continue capture after fullscreen exit, preserving windowed and Linux Escape quit.
    """
    monkeypatch.setattr(desktop.sys, "platform", platform)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_: 1)
    exit_native = Mock(return_value=handled)
    monkeypatch.setattr(macos_window, "exit_fullscreen", exit_native)
    monkeypatch.setattr(desktop.cv2, "waitKey", Mock(side_effect=[27, ord("q")]))
    assert desktop.main([]) == 0
    assert len(viewer.displayed) == frames
    if platform == "darwin":
        exit_native.assert_called_once_with(desktop.WINDOW_NAME)
    else:
        exit_native.assert_not_called()


__all__ = ["viewer_fixture"]
