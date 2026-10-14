from typing import Any
from unittest.mock import Mock

import cv2
import pytest

from based_thermals.pointer import Point, PointerMonitor


@pytest.fixture
def monitor(monkeypatch):
    # Unit tests need neither a display connection nor macOS native libraries.
    monkeypatch.setattr("based_thermals.pointer.C.CDLL", Mock(side_effect=OSError))
    return PointerMonitor("test viewer")


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
def test_native_pointer_detects_image_exit_toolbar_and_reentry(
    monitor: Any, monkeypatch: pytest.MonkeyPatch, scale: Any
) -> None:
    """
    Verify native pointer detects image exit toolbar and reentry.
    """
    left, top = 100, 200
    width, height = int(768 * scale), int(630 * scale)
    monkeypatch.setattr(
        "based_thermals.pointer.cv2.getWindowImageRect",
        lambda _name: (left, top, width, height),
    )
    # Native screen position changes even without any OpenCV mouse callback.
    for x, y, expected in (
        (left + width // 2, top + height // 2, True),
        (left - 1, top + height // 2, False),
        (left + width, top + height // 2, False),
        (left + 100, top + height, False),
        (left + 100, top + 10, False),
        (left + width // 2, top + height // 2, True),
    ):
        # noinspection PyUnresolvedReferences
        monkeypatch.setattr(
            monitor, "_screen_position", lambda screen_x=x, screen_y=y: (screen_x, screen_y)
        )
        assert monitor.over_image(576, 54) is expected


def test_native_pointer_unavailable_preserves_event_bounds_checks(monitor, monkeypatch):
    assert monitor.over_image(576, 54) is None
    monkeypatch.setattr(monitor, "_screen_position", lambda: (200, 300))
    get_rect = Mock(side_effect=cv2.error("window not displayed yet"))
    monkeypatch.setattr("based_thermals.pointer.cv2.getWindowImageRect", get_rect)
    assert monitor.over_image(576, 54) is None
    get_rect.side_effect = None
    get_rect.return_value = (0, 0, -1, -1)
    assert monitor.over_image(576, 54) is None
    monitor.close()


def test_macos_coregraphics_pointer_uses_safe_c_api(monitor, monkeypatch):
    coregraphics = Mock()
    coregraphics.CGEventCreate.return_value = 123
    coregraphics.CGEventGetLocation.return_value = Point(200, 350)
    monitor._coregraphics = coregraphics
    monkeypatch.setattr(monitor, "_mac_window_rect", lambda: (100, 200, 384, 315))
    monkeypatch.setattr(
        "based_thermals.pointer.cv2.getWindowImageRect",
        lambda _name: (100, 200, 384, 315),
    )
    assert monitor.over_image(576, 54) is True
    coregraphics.CGEventGetLocation.return_value = Point(90, 350)
    assert monitor.over_image(576, 54) is False
    coregraphics.CFRelease.assert_called_with(123)


def test_macos_pointer_containment_ignores_cocoa_image_rectangle(
    monitor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify macos pointer containment ignores cocoa image rectangle.
    """
    monitor._coregraphics = Mock()
    # This rectangle is deliberately in an incompatible coordinate space.
    get_rect = Mock(return_value=(200, 900, 768, 576))
    monkeypatch.setattr("based_thermals.pointer.cv2.getWindowImageRect", get_rect)
    bounds = Mock(return_value=(100, 200, 384, 315))
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(monitor, "_mac_window_rect", bounds)
    for position, expected in (((200, 350), True), ((90, 350), False), ((200, 350), True)):
        # noinspection PyUnresolvedReferences
        monkeypatch.setattr(
            monitor, "_screen_position", lambda screen_position=position: screen_position
        )
        assert monitor.over_image(576, 54) is expected
    get_rect.assert_not_called()
    bounds.return_value = None
    assert monitor.over_image(576, 54) is None
