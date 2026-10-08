"""Native pointer checks for exits that OpenCV's mouse callback does not report."""

from __future__ import annotations

import ctypes as C
import sys

import cv2


class Point(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double)]


class PointerMonitor:
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self._display = None
        self._x11 = None
        self._coregraphics = None
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
                self._coregraphics = C.CDLL(
                    "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
                )
                self._coregraphics.CGEventCreate.argtypes = [C.c_void_p]
                self._coregraphics.CGEventCreate.restype = C.c_void_p
                self._coregraphics.CGEventGetLocation.argtypes = [C.c_void_p]
                self._coregraphics.CGEventGetLocation.restype = Point
                self._coregraphics.CFRelease.argtypes = [C.c_void_p]
        except (OSError, AttributeError):
            # Keep event-based bounds checks if the native API is unavailable.
            self.close()

    def close(self) -> None:
        if self._display is not None:
            self._x11.XCloseDisplay(self._display)
            self._display = None

    def _screen_position(self) -> tuple[int, int] | None:
        if self._coregraphics is not None:
            event = self._coregraphics.CGEventCreate(None)
            if not event:
                return None
            try:
                point = self._coregraphics.CGEventGetLocation(event)
                return round(point.x), round(point.y)
            finally:
                self._coregraphics.CFRelease(event)
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

    def over_image(self, image_height: int, toolbar_height: int) -> bool | None:
        """Return None when native tracking is unavailable, otherwise image containment."""
        toolbar_fraction = toolbar_height / (image_height + toolbar_height)
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
