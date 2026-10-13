"""
Escape exits native macOS fullscreen while preserving ordinary quit behavior.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture

from based_thermals import desktop, macos_window
from based_thermals.desktop_app.session import DesktopSession
from based_thermals.desktop_app.settings import SettingsController
from based_thermals.pointer import PointerMonitor


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
    assert macos_window.fullscreen_state("viewer") is fullscreen
    assert macos_window.fullscreen_state("missing") is None
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
    assert macos_window.fullscreen_state("viewer") is None
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


@pytest.mark.parametrize("temporarily_unavailable", [True, False])
def test_fullscreen_exit_restores_last_manually_resized_window(
    monkeypatch: pytest.MonkeyPatch, temporarily_unavailable: bool
) -> None:
    """
    Restore the pre-fullscreen size after either native or Escape exit completes.
    """
    state = Mock(return_value=False)
    monkeypatch.setattr(macos_window, "fullscreen_state", state)
    resize = Mock()
    bounds = Mock(return_value=(930, 710))
    api = SimpleNamespace(
        sys=SimpleNamespace(platform="darwin"), WINDOW_NAME="viewer",
        window_resize_size=lambda _name: None, cv2=SimpleNamespace(resizeWindow=resize),
    )
    controller = cast(DesktopSession, SettingsController(api=api, requested_window_size=(930, 710)))
    monitor = object.__new__(PointerMonitor)
    monkeypatch.setattr(monitor, "window_size", bounds)
    controller.pointer_monitor = monitor
    assert controller.current_window_size() == (930, 710)
    bounds.return_value = (1234, 780)
    assert controller.current_window_size() == (1234, 780)
    state.return_value = True
    bounds.return_value = (2560, 1412)
    assert controller.current_window_size() == (2560, 1412)
    bounds.assert_called_with((930, 710), fullscreen=True)
    assert controller.cocoa_windowed_size == (1234, 780)
    resize.assert_not_called()
    if temporarily_unavailable:
        state.return_value = None
        assert controller.current_window_size() == (2560, 1412)
        bounds.assert_called_with((930, 710), fullscreen=True)
        assert controller.cocoa_fullscreen
    state.return_value = False
    assert controller.current_window_size() == (1234, 780)
    resize.assert_called_once_with("viewer", 1234, 780)
    assert controller.requested_window_size == (1234, 780)
    bounds.return_value = (1234, 780)
    assert controller.current_window_size() == (1234, 780)
    assert resize.call_count == 1


__all__ = ["viewer_fixture"]
