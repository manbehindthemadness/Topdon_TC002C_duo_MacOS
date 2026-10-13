"""
Keep Cocoa fullscreen toolbar windows out of viewer geometry calculations.
"""

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from based_thermals import pointer


@pytest.mark.parametrize("toolbar_first", [True, False])
def test_fullscreen_selects_image_window_instead_of_toolbar(
    monkeypatch: pytest.MonkeyPatch, toolbar_first: bool
) -> None:
    """
    Select the full viewer bounds regardless of native window ordering.
    """
    monkeypatch.setattr(pointer.C, "CDLL", Mock(side_effect=OSError))
    monitor = pointer.PointerMonitor("viewer")
    toolbar = {"pid": os.getpid(), "layer": 0, "bounds": (0, 0, 2560, 44)}
    image = {"pid": os.getpid(), "layer": 0, "bounds": (0, 0, 2560, 1440)}
    other_process = {"pid": os.getpid() + 1, "layer": 0, "bounds": (0, 0, 9999, 9999)}
    overlay = {"pid": os.getpid(), "layer": 1, "bounds": (0, 0, 9999, 9999)}
    windows = [toolbar, image] if toolbar_first else [image, toolbar]
    windows.extend([other_process, overlay])
    keys = {"kCGWindowOwnerPID": "pid", "kCGWindowBounds": "bounds", "kCGWindowLayer": "layer"}

    def library_constant(_library: Any, name: str) -> SimpleNamespace:
        """
        Replace native exported key addresses with inspectable dictionary keys.
        """
        return SimpleNamespace(value=keys[name])

    def read_number(value: SimpleNamespace, _kind: int, destination: Any) -> bool:
        """
        Populate a ctypes scalar as the CoreFoundation API would.
        """
        destination._obj.value = value.value
        return True

    def read_rectangle(bounds: tuple[int, int, int, int], destination: Any) -> bool:
        """
        Populate a ctypes rectangle from fake native bounds.
        """
        rectangle = destination._obj
        rectangle.origin.x, rectangle.origin.y, rectangle.size.width, rectangle.size.height = bounds
        return True

    coregraphics = Mock()
    coregraphics.CGWindowListCopyWindowInfo.return_value = windows
    coregraphics.CFArrayGetCount.side_effect = len
    coregraphics.CFArrayGetValueAtIndex.side_effect = lambda items, index: items[index]
    coregraphics.CFDictionaryGetValue.side_effect = lambda window, key: (
        window[key] if key == "bounds" else SimpleNamespace(value=window[key])
    )
    coregraphics.CFNumberGetValue.side_effect = read_number
    coregraphics.CGRectMakeWithDictionaryRepresentation.side_effect = read_rectangle
    monitor._coregraphics = coregraphics
    monkeypatch.setattr(pointer.C, "c_void_p", SimpleNamespace(in_dll=library_constant))
    # The normal window was calibrated with a 28-point title bar.
    monitor._mac_chrome_height = 28
    assert monitor._mac_window_rect() == (0, 0, 2560, 1440)
    assert monitor.window_size((1536, 610)) == (2560, 1412)
    assert monitor.window_size((1536, 610), fullscreen=True) == (2560, 1440)
    assert monitor._mac_chrome_height == 28
    coregraphics.CFRelease.assert_called_with(windows)
