"""
Native macOS window sizing across chart toggles and manual window resizes.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture

from topdon_duo import desktop, window_style
from topdon_duo.pointer import PointerMonitor


def test_cocoa_bitmap_dimensions_are_not_used_as_window_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ignore Cocoa's stale image dimensions instead of overwriting the native window size.
    """
    monkeypatch.setattr(window_style.sys, "platform", "darwin")
    get_rectangle = Mock(return_value=(0, 0, 768, 1056))
    monkeypatch.setattr(window_style.cv2, "getWindowImageRect", get_rectangle)
    assert window_style.window_resize_size("viewer") is None
    get_rectangle.assert_not_called()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_retain_existing_window_size_detection(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    """
    Leave existing non-Cocoa geometry behavior intact.
    """
    monkeypatch.setattr(window_style.sys, "platform", platform)
    monkeypatch.setattr(window_style, "_qt_window_size_functions", lambda: None)
    monkeypatch.setattr(window_style.cv2, "getWindowImageRect", lambda _: (0, 0, 1860, 710))
    assert window_style.window_resize_size("viewer") == (1860, 710)


def test_native_content_size_waits_for_resize_then_preserves_chrome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Calibrate measured chrome once, then track manual resizing without height drift.
    """
    monkeypatch.setattr("topdon_duo.pointer.C.CDLL", Mock(side_effect=OSError))
    monitor = PointerMonitor("viewer")
    monitor._coregraphics = Mock()
    bounds = Mock(return_value=(50, 50, 768, 660))
    monkeypatch.setattr(monitor, "_mac_window_rect", bounds)
    assert monitor.window_size((930, 710)) is None
    bounds.return_value = (50, 50, 930, 738)
    assert monitor.window_size((930, 710)) == (930, 710)
    bounds.return_value = (50, 50, 2020, 788)
    assert monitor.window_size((930, 710)) == (2020, 760)
    bounds.return_value = None
    assert monitor.window_size((930, 710)) is None
    monitor.close()


def test_macos_chart_toggle_uses_native_size_after_manual_resize(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Show, manually resize, hide and show charts without stale canvas sizes or height drift.
    """
    from topdon_duo.window_preferences import load_main_window_size, save_main_window_size

    save_main_window_size((930, 710))
    monkeypatch.setattr(desktop.sys, "platform", "darwin")
    # The desktop facade re-exports these dependencies for injection.
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(desktop, "window_resize_size", window_style.window_resize_size)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_: 1)
    window = [930, 710]
    monkeypatch.setattr("topdon_duo.pointer.C.CDLL", Mock(side_effect=OSError))

    class NativeMonitor(PointerMonitor):
        """
        Simulate the window server while exercising the actual content-size calculation.
        """

        def __init__(self, name: str) -> None:
            """
            Initialize only the state required for fake native geometry.
            """
            super().__init__(name)
            self._coregraphics = Mock()
            self._mac_chrome_height = None

        def _mac_window_rect(self) -> tuple[float, float, float, float]:
            """
            Return window bounds with a measured 28-point title bar.
            """
            return 50, 50, window[0], window[1] + 28

        def over_image(self, image_height: int, toolbar_height: int) -> bool:
            """
            Keep hover handling independent of the window geometry regression.
            """
            return True

        def close(self) -> None:
            """
            No native resources are allocated by this fake.
            """

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(desktop, "PointerMonitor", NativeMonitor)
    resize_window = Mock(
        side_effect=lambda _name, width, height: window.__setitem__(slice(None), [width, height])
    )
    monkeypatch.setattr(desktop.cv2, "resizeWindow", resize_window)
    initial_image = True

    def show_image(_name: str, image: Any) -> None:
        """
        Reproduce Cocoa resetting content dimensions when its first bitmap arrives.
        """
        nonlocal initial_image
        if initial_image:
            window[:] = [image.shape[1], image.shape[0]]
            initial_image = False
        viewer.displayed.append(image.copy())

    monkeypatch.setattr(desktop.cv2, "imshow", show_image)
    step = 0

    def wait_key(_delay: int) -> int:
        """
        Simulate native user resizing between chart hide/show operations.
        """
        nonlocal step
        step += 1
        if step == 1:
            return ord("g")
        if step == 2:
            assert viewer.displayed[-1].shape[:2] == (710, 1860)
            window[:] = [2020, 760]
            return -1
        if step == 3:
            assert viewer.displayed[-1].shape[:2] == (760, 2020)
            return ord("g")
        if step == 4:
            assert window == [1010, 760]
            return ord("g")
        assert viewer.displayed[-1].shape[:2] == (760, 2020)
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert [call.args[1:] for call in resize_window.call_args_list] == [
        (930, 710),
        (1860, 710),
        (1010, 760),
        (2020, 760),
    ]
    assert load_main_window_size() == (2020, 760)


__all__ = ["viewer_fixture"]
