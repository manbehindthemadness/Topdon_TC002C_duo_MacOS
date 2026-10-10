"""Native pointer checks for exits that OpenCV's mouse callback does not report."""

from __future__ import annotations

# noinspection PyPep8Naming
import ctypes as C
import os
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
        self._coregraphics = None
        self._mac_chrome_height: int | None = None
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
                cg = self._coregraphics
                cg.CGWindowListCopyWindowInfo.argtypes = [C.c_uint32, C.c_uint32]
                cg.CGWindowListCopyWindowInfo.restype = C.c_void_p
                cg.CFArrayGetCount.argtypes = [C.c_void_p]
                cg.CFArrayGetCount.restype = C.c_long
                cg.CFArrayGetValueAtIndex.argtypes = [C.c_void_p, C.c_long]
                cg.CFArrayGetValueAtIndex.restype = C.c_void_p
                cg.CFDictionaryGetValue.argtypes = [C.c_void_p, C.c_void_p]
                cg.CFDictionaryGetValue.restype = C.c_void_p
                cg.CFNumberGetValue.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
                cg.CFNumberGetValue.restype = C.c_bool
                cg.CGRectMakeWithDictionaryRepresentation.argtypes = [C.c_void_p, C.POINTER(Rect)]
                cg.CGRectMakeWithDictionaryRepresentation.restype = C.c_bool
        except (OSError, AttributeError):
            # Keep event-based bounds checks if the native API is unavailable.
            self.close()

    def close(self) -> None:
        if self._display is not None and self._x11 is not None:
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
        if self._display is None or self._x11 is None:
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
        if self._coregraphics is not None:
            # Cocoa's getWindowImageRect is not a top-left screen rectangle.
            # Compare native pointer and native window bounds in the same
            # coordinate system; toolbar/image bounds come from mouse events.
            rectangle = self._mac_window_rect()
            if rectangle is None:
                return None
            left, top, width, height = rectangle
            x, y = position
            return left <= x < left + width and top <= y < top + height
        try:
            left, top, width, height = cv2.getWindowImageRect(self.window_name)
        except cv2.error:
            return None
        if width <= 0 or height <= 0:
            return None
        x, y = position
        return left <= x < left + width and top + height * toolbar_fraction <= y < top + height

    def window_size(self, requested_size: tuple[int, int] | None) -> tuple[int, int] | None:
        """
        Measure Cocoa content size, calibrating window chrome from the initial resize.
        """
        if self._coregraphics is None or requested_size is None:
            return None
        rectangle = self._mac_window_rect()
        if rectangle is None:
            return None
        width, outer_height = round(rectangle[2]), round(rectangle[3])
        if self._mac_chrome_height is None:
            # Wait for the window server to reflect the requested initial size.
            difference = outer_height - requested_size[1]
            if width != requested_size[0] or not 0 <= difference <= 96:
                return None
            self._mac_chrome_height = difference
        height = outer_height - self._mac_chrome_height
        return (width, height) if width > 0 and height > 0 else None

    def _mac_window_rect(self) -> tuple[float, float, float, float] | None:
        """
        Read the current process's normal on-screen window bounds through CoreGraphics.
        """
        cg = self._coregraphics
        if cg is None:
            return None
        windows = None
        try:
            pid_key = C.c_void_p.in_dll(cg, "kCGWindowOwnerPID").value
            bounds_key = C.c_void_p.in_dll(cg, "kCGWindowBounds").value
            layer_key = C.c_void_p.in_dll(cg, "kCGWindowLayer").value
            windows = cg.CGWindowListCopyWindowInfo(1, 0)  # On-screen windows only.
            if not windows:
                return None
            for index in range(cg.CFArrayGetCount(windows)):
                window = cg.CFArrayGetValueAtIndex(windows, index)
                number = cg.CFDictionaryGetValue(window, pid_key)
                pid = C.c_int32()
                if not number or not cg.CFNumberGetValue(number, 3, C.byref(pid)):
                    continue
                if pid.value != os.getpid():
                    continue
                layer_number = cg.CFDictionaryGetValue(window, layer_key)
                layer = C.c_int32()
                if (
                    not layer_number
                    or not cg.CFNumberGetValue(layer_number, 3, C.byref(layer))
                    or layer.value != 0
                ):
                    continue
                bounds = cg.CFDictionaryGetValue(window, bounds_key)
                rect = Rect()
                if (
                    bounds
                    and cg.CGRectMakeWithDictionaryRepresentation(bounds, C.byref(rect))
                    and rect.size.width > 0
                    and rect.size.height > 0
                ):
                    return rect.origin.x, rect.origin.y, rect.size.width, rect.size.height
        except (OSError, AttributeError, ValueError):
            return None
        finally:
            if windows:
                cg.CFRelease(windows)
        return None
