"""Native pointer checks for exits that OpenCV's mouse callback does not report."""

from __future__ import annotations

import ctypes as C
import platform
import sys

import cv2


class Point(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double)]


class Size(C.Structure):
    _fields_ = [("width", C.c_double), ("height", C.c_double)]


class Rect(C.Structure):
    _fields_ = [("origin", Point), ("size", Size)]


class PointerMonitor:
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self._display = None
        self._x11 = None
        self._objc = None
        try:
            if sys.platform.startswith("linux"):
                self._x11 = C.CDLL("libX11.so.6")
                self._x11.XOpenDisplay.argtypes = [C.c_char_p]
                self._x11.XOpenDisplay.restype = C.c_void_p
                self._x11.XDefaultRootWindow.argtypes = [C.c_void_p]
                self._x11.XDefaultRootWindow.restype = C.c_ulong
                self._x11.XQueryPointer.argtypes = [
                    C.c_void_p,
                    C.c_ulong,
                    C.POINTER(C.c_ulong),
                    C.POINTER(C.c_ulong),
                    C.POINTER(C.c_int),
                    C.POINTER(C.c_int),
                    C.POINTER(C.c_int),
                    C.POINTER(C.c_int),
                    C.POINTER(C.c_uint),
                ]
                self._x11.XQueryPointer.restype = C.c_int
                self._x11.XCloseDisplay.argtypes = [C.c_void_p]
                self._x11.XCloseDisplay.restype = C.c_int
                self._display = self._x11.XOpenDisplay(None)
            elif sys.platform == "darwin":
                self._objc = C.CDLL("/usr/lib/libobjc.A.dylib")
                self._objc.objc_getClass.argtypes = [C.c_char_p]
                self._objc.objc_getClass.restype = C.c_void_p
                self._objc.sel_registerName.argtypes = [C.c_char_p]
                self._objc.sel_registerName.restype = C.c_void_p
        except (OSError, AttributeError):
            # Keep event-based bounds checks if the native API is unavailable.
            self.close()
            self._objc = None

    def close(self) -> None:
        if self._display is not None:
            self._x11.XCloseDisplay(self._display)
            self._display = None

    def _screen_position(self) -> tuple[int, int] | None:
        if self._display is None:
            return None
        root = self._x11.XDefaultRootWindow(self._display)
        root_return, child = C.c_ulong(), C.c_ulong()
        x, y, window_x, window_y = (C.c_int() for _ in range(4))
        mask = C.c_uint()
        if not self._x11.XQueryPointer(
            self._display,
            root,
            C.byref(root_return),
            C.byref(child),
            C.byref(x),
            C.byref(y),
            C.byref(window_x),
            C.byref(window_y),
            C.byref(mask),
        ):
            return None
        return x.value, y.value

    def _message(self, receiver, selector, result=C.c_void_p, arguments=(), values=()):
        selector = self._objc.sel_registerName(selector.encode())
        # Intel macOS returns a four-double NSRect through an output pointer.
        if result is Rect and platform.machine() == "x86_64":
            rect = Rect()
            function = C.CFUNCTYPE(None, C.POINTER(Rect), C.c_void_p, C.c_void_p)(
                ("objc_msgSend_stret", self._objc)
            )
            function(C.byref(rect), receiver, selector)
            return rect
        function = C.CFUNCTYPE(result, C.c_void_p, C.c_void_p, *arguments)(
            ("objc_msgSend", self._objc)
        )
        return function(receiver, selector, *values)

    def _cocoa_over_image(self, toolbar_fraction: float) -> bool | None:
        pool = self._message(self._objc.objc_getClass(b"NSAutoreleasePool"), "new")
        try:
            return self._cocoa_image_contains_pointer(toolbar_fraction)
        finally:
            self._message(pool, "drain", None)

    def _cocoa_image_contains_pointer(self, toolbar_fraction: float) -> bool | None:
        app = self._message(self._objc.objc_getClass(b"NSApplication"), "sharedApplication")
        windows = self._message(app, "windows")
        for index in range(self._message(windows, "count", C.c_ulong)):
            window = self._message(
                windows, "objectAtIndex:", arguments=(C.c_ulong,), values=(index,)
            )
            title = self._message(self._message(window, "title"), "UTF8String", C.c_char_p)
            if title != self.window_name.encode():
                continue
            content = self._message(window, "contentView")
            view = self._message(content, "imageView")
            if not view:
                return None
            point = self._message(window, "mouseLocationOutsideOfEventStream", Point)
            point = self._message(
                view,
                "convertPoint:fromView:",
                Point,
                arguments=(Point, C.c_void_p),
                values=(point, None),
            )
            bounds = self._message(view, "bounds", Rect)
            x, y = point.x - bounds.origin.x, point.y - bounds.origin.y
            if not self._message(view, "isFlipped", C.c_bool):
                y = bounds.size.height - y
            return (
                0 <= x < bounds.size.width
                and toolbar_fraction * bounds.size.height <= y < bounds.size.height
            )
        return False

    def over_image(self, image_height: int, toolbar_height: int) -> bool | None:
        """Return None when native tracking is unavailable, otherwise image containment."""
        toolbar_fraction = toolbar_height / (image_height + toolbar_height)
        if self._objc is not None:
            return self._cocoa_over_image(toolbar_fraction)
        position = self._screen_position()
        if position is None:
            return None
        try:
            left, top, width, height = cv2.getWindowImageRect(self.window_name)
        except cv2.error:
            return None
        if width <= 0 or height <= 0:
            return None
        x, y = position
        return left <= x < left + width and top + height * toolbar_fraction <= y < top + height
