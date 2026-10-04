from unittest.mock import Mock

import cv2
import pytest

from topdon_duo.pointer import Point, PointerMonitor, Rect, Size


@pytest.fixture
def monitor(monkeypatch):
    # Unit tests need neither a display connection nor macOS native libraries.
    monkeypatch.setattr("topdon_duo.pointer.C.CDLL", Mock(side_effect=OSError))
    return PointerMonitor("test viewer")


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
def test_native_pointer_detects_image_exit_toolbar_and_reentry(monitor, monkeypatch, scale):
    left, top = 100, 200
    width, height = int(768 * scale), int(630 * scale)
    monkeypatch.setattr(
        "topdon_duo.pointer.cv2.getWindowImageRect",
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
        monkeypatch.setattr(monitor, "_screen_position", lambda x=x, y=y: (x, y))
        assert monitor.over_image(576, 54) is expected


def test_native_pointer_unavailable_preserves_event_bounds_checks(monitor, monkeypatch):
    assert monitor.over_image(576, 54) is None
    monkeypatch.setattr(monitor, "_screen_position", lambda: (200, 300))
    get_rect = Mock(side_effect=cv2.error("window not displayed yet"))
    monkeypatch.setattr("topdon_duo.pointer.cv2.getWindowImageRect", get_rect)
    assert monitor.over_image(576, 54) is None
    get_rect.side_effect = None
    get_rect.return_value = (0, 0, -1, -1)
    assert monitor.over_image(576, 54) is None
    monitor.close()


@pytest.mark.parametrize("flipped", [False, True])
def test_cocoa_pointer_checks_actual_image_view_bounds(monitor, monkeypatch, flipped):
    monitor._objc = Mock()
    point = Point(100, 100)

    def message(_receiver, selector, _result=None, arguments=(), values=()):
        if selector == "count":
            return 1
        if selector == "UTF8String":
            return b"test viewer"
        if selector == "convertPoint:fromView:":
            return point
        if selector == "bounds":
            return Rect(Point(0, 0), Size(384, 315))
        if selector == "isFlipped":
            return flipped
        return 1

    monkeypatch.setattr(monitor, "_message", message)
    assert monitor.over_image(576, 54) is True
    point.x = -1
    assert monitor.over_image(576, 54) is False
    point.x = 100
    point.y = 0 if flipped else 315
    assert monitor.over_image(576, 54) is False  # Toolbar.
    point.y = 315 if flipped else -1
    assert monitor.over_image(576, 54) is False  # Outside image view.
