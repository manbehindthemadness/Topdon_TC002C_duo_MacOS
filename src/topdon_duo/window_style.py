"""Black backgrounds for OpenCV's native window and ambient trackbar."""

from __future__ import annotations

import ctypes as C
import sys
from functools import lru_cache
from pathlib import Path

import cv2

from .pointer import PointerMonitor, Rect

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
    """Return dimensions suitable for resizeWindow, including unused image margins."""
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
        elif sys.platform == "darwin":
            monitor = PointerMonitor(window_name)
            pool = None
            try:
                if monitor._objc is not None:
                    message = monitor._message
                    pool = message(monitor._objc.objc_getClass(b"NSAutoreleasePool"), "new")
                    app = message(
                        monitor._objc.objc_getClass(b"NSApplication"), "sharedApplication"
                    )
                    windows = message(app, "windows")
                    for index in range(message(windows, "count", C.c_ulong)):
                        window = message(
                            windows, "objectAtIndex:", arguments=(C.c_ulong,), values=(index,)
                        )
                        if (
                            message(message(window, "title"), "UTF8String", C.c_char_p)
                            != window_name.encode()
                        ):
                            continue
                        content = message(window, "contentView")
                        rect = message(content, "bounds", Rect)
                        slider_height = message(content, "sliderHeight", C.c_int)
                        return round(rect.size.width), round(rect.size.height - slider_height)
            finally:
                if pool is not None:
                    monitor._message(pool, "drain", None)
                monitor.close()
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


def _cocoa_black_backgrounds(window_name: str) -> bool:
    monitor = PointerMonitor(window_name)
    if monitor._objc is None:
        return False
    message = monitor._message
    get_class = monitor._objc.objc_getClass
    pool = message(get_class(b"NSAutoreleasePool"), "new")
    try:
        black = message(get_class(b"NSColor"), "blackColor")
        white = message(get_class(b"NSColor"), "whiteColor")
        cg_black = message(black, "CGColor")
        windows = message(message(get_class(b"NSApplication"), "sharedApplication"), "windows")
        for index in range(message(windows, "count", C.c_ulong)):
            window = message(windows, "objectAtIndex:", arguments=(C.c_ulong,), values=(index,))
            if message(message(window, "title"), "UTF8String", C.c_char_p) != window_name.encode():
                continue
            message(window, "setBackgroundColor:", arguments=(C.c_void_p,), values=(black,))
            views = [message(window, "contentView")]
            while views:
                view = views.pop()
                message(view, "setWantsLayer:", arguments=(C.c_bool,), values=(True,))
                message(
                    message(view, "layer"),
                    "setBackgroundColor:",
                    arguments=(C.c_void_p,),
                    values=(cg_black,),
                )
                for selector, color in (("setBackgroundColor:", black), ("setTextColor:", white)):
                    native_selector = monitor._objc.sel_registerName(selector.encode())
                    if message(
                        view,
                        "respondsToSelector:",
                        C.c_bool,
                        arguments=(C.c_void_p,),
                        values=(native_selector,),
                    ):
                        message(view, selector, arguments=(C.c_void_p,), values=(color,))
                children = message(view, "subviews")
                views.extend(
                    message(children, "objectAtIndex:", arguments=(C.c_ulong,), values=(i,))
                    for i in range(message(children, "count", C.c_ulong))
                )
            return True
        return False
    finally:
        message(pool, "drain", None)
        monitor.close()


def set_black_window_backgrounds(window_name: str) -> bool:
    try:
        if sys.platform.startswith("linux"):
            return _qt_black_backgrounds()
        if sys.platform == "darwin":
            return _cocoa_black_backgrounds(window_name)
    except (OSError, AttributeError, ValueError):
        return False
    return False
