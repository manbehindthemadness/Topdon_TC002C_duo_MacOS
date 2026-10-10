"""Black backgrounds for OpenCV's native window and ambient trackbar."""

from __future__ import annotations

# noinspection PyPep8Naming
import ctypes as C
import sys
from functools import lru_cache
from pathlib import Path

import cv2

QT_STYLESHEET = b"""
QWidget { background-color: #000000; color: #eeeeee; }
QSlider { background-color: #000000; }
QSlider::groove:horizontal { background: #444444; height: 4px; border: none; }
QSlider::handle:horizontal {
    background: #dddddd; width: 12px; margin: -5px 0; border-radius: 2px;
}
QSlider::sub-page:horizontal { background: #777777; }
"""


class _QtPoint(C.Structure):
    _fields_ = [("x", C.c_int), ("y", C.c_int)]


class _QtRect(C.Structure):
    # QRect stores inclusive corner coordinates, rather than width and height.
    _fields_ = [("left", C.c_int), ("top", C.c_int), ("right", C.c_int), ("bottom", C.c_int)]


@lru_cache(maxsize=1)
def _qt_window_size_functions():
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split(maxsplit=5)[-1]
        if "libQt5Widgets" not in Path(path).name:
            continue
        library = C.CDLL(path)
        for namespace in ("14QtOpenCVPython", ""):
            prefix = f"_ZN{namespace}"
            point_ref = "ERKNS_6QPointE" if namespace else "ERK6QPoint"
            try:
                at = getattr(library, prefix + "12QApplication10topLevelAt" + point_ref)
                bounds = getattr(library, f"_ZNK{namespace}7QWidget12contentsRectEv")
            except AttributeError:
                continue
            at.argtypes, at.restype = [C.POINTER(_QtPoint)], C.c_void_p
            bounds.argtypes, bounds.restype = [C.c_void_p], _QtRect
            return at, bounds
    return None


def window_resize_size(window_name: str) -> tuple[int, int] | None:
    """
    Return native resize dimensions where OpenCV exposes trustworthy window geometry.
    """
    if sys.platform == "darwin":
        # Cocoa reports the imshow bitmap size, not the resized content view.
        # The session uses CoreGraphics bounds and its known resize request instead.
        return None
    try:
        left, top, width, height = cv2.getWindowImageRect(window_name)
        if width <= 0 or height <= 0:
            return None
        if sys.platform.startswith("linux"):
            functions = _qt_window_size_functions()
            if functions:
                at, bounds = functions
                widget = at(C.byref(_QtPoint(left + width // 2, top + height // 2)))
                if widget:
                    rect = bounds(widget)
                    return rect.right - rect.left + 1, rect.bottom - rect.top + 1
                return None  # Retain the last full size while another window covers it.
        return width, height
    except (cv2.error, OSError, AttributeError, ValueError):
        return None


def _qt_black_backgrounds() -> bool:
    # Use the Qt 5 libraries already loaded by OpenCV. Importing another Qt
    # binding here would create a second QApplication with incompatible widgets.
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split(maxsplit=5)[-1]
        for name in ("Core", "Widgets"):
            if f"libQt5{name}" in Path(path).name:
                libraries[name] = path
    if len(libraries) != 2:
        return False
    core = C.CDLL(libraries["Core"])
    widgets = C.CDLL(libraries["Widgets"])
    for namespace in ("14QtOpenCVPython", ""):
        prefix = f"_ZN{namespace}"
        reference = "ERKNS_7QStringE" if namespace else "ERK7QString"
        try:
            app = C.c_void_p.in_dll(core, prefix + "16QCoreApplication4selfE").value
            latin1 = getattr(core, prefix + "7QString17fromLatin1_helperEPKci")
            assign = getattr(
                core, prefix + "7QStringaSERKS0_" if namespace else "_ZN7QStringaSERKS_"
            )
            set_style = getattr(widgets, prefix + "12QApplication13setStyleSheet" + reference)
        except (AttributeError, ValueError):
            continue
        if not app:
            return False
        # Qt 5 QString consists of one shared-data pointer. Its exported Latin-1
        # factory and assignment operator manage ownership without a C++ shim.
        latin1.argtypes, latin1.restype = [C.c_char_p, C.c_int], C.c_void_p
        assign.argtypes = [C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        assign.restype = C.c_void_p
        set_style.argtypes, set_style.restype = [C.c_void_p, C.POINTER(C.c_void_p)], None
        style = C.c_void_p(latin1(QT_STYLESHEET, len(QT_STYLESHEET)))
        empty = C.c_void_p(latin1(None, 0))
        try:
            set_style(app, C.byref(style))
        finally:
            assign(C.byref(style), C.byref(empty))
        return True
    return False


def set_black_window_backgrounds(_window_name: str) -> bool:
    """
    Apply black Qt widget backgrounds where native window styling is supported.
    """
    try:
        if sys.platform.startswith("linux"):
            return _qt_black_backgrounds()
        # OpenCV's Cocoa widgets are intentionally left to their native style.
        # Reaching through them with raw Objective-C messages is ABI-fragile and
        # can abort the process before the first camera frame is displayed.
    except (OSError, AttributeError, ValueError):
        return False
    return False
