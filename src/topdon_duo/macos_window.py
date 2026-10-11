"""
Native fullscreen exit for the Cocoa viewer, independent of OpenCV's cached flag.
"""

from __future__ import annotations

# noinspection PyPep8Naming
import ctypes as C
import sys
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

FULLSCREEN_MASK = 1 << 14


def _message(address: int, return_type: Any, *arguments: Any) -> Any:
    """
    Instantiate a dynamically typed native message signature at the runtime address.
    """
    factory = cast(
        Callable[[int], Any], C.CFUNCTYPE(return_type, C.c_void_p, C.c_void_p, *arguments)
    )
    message = factory(address)
    return message


class _CocoaAPI:
    """
    Bind Objective-C messages with signatures appropriate to their return values.
    """

    def __init__(self) -> None:
        """
        Use the Objective-C runtime already loaded by OpenCV's Cocoa backend.
        """
        self.library = C.CDLL("/usr/lib/libobjc.A.dylib")
        self.library.objc_getClass.argtypes = [C.c_char_p]
        self.library.objc_getClass.restype = C.c_void_p
        self.library.sel_registerName.argtypes = [C.c_char_p]
        self.library.sel_registerName.restype = C.c_void_p
        address = C.cast(self.library.objc_msgSend, C.c_void_p).value
        if address is None:
            raise ValueError("Objective-C message runtime is unavailable")
        self.object = _message(address, C.c_void_p)
        self.integer = _message(address, C.c_ulong)
        self.string = _message(address, C.c_char_p)
        self.item = _message(address, C.c_void_p, C.c_ulong)
        self.action = _message(address, None, C.c_void_p)

    def selector(self, name: str) -> Any:
        """
        Resolve a native selector without modifying Objective-C classes.
        """
        selector = self.library.sel_registerName(name.encode("ascii"))
        return selector


@lru_cache(maxsize=1)
def _cocoa_api() -> _CocoaAPI:
    """
    Lazily load native bindings only when the macOS viewer requests fullscreen exit.
    """
    api = _CocoaAPI()
    return api


def exit_fullscreen(window_name: str) -> bool:
    """
    Leave native fullscreen for the named viewer and report whether it was handled.
    """
    if sys.platform != "darwin":
        return False
    try:
        api = _cocoa_api()
        application_class = api.library.objc_getClass(b"NSApplication")
        application = api.object(application_class, api.selector("sharedApplication"))
        windows = api.object(application, api.selector("windows"))
        for index in range(api.integer(windows, api.selector("count"))):
            window = api.item(windows, api.selector("objectAtIndex:"), index)
            title = api.object(window, api.selector("title"))
            title_bytes = api.string(title, api.selector("UTF8String"))
            if title_bytes != window_name.encode("utf-8"):
                continue
            if not api.integer(window, api.selector("styleMask")) & FULLSCREEN_MASK:
                return False
            api.action(window, api.selector("toggleFullScreen:"), None)
            return True
    except (OSError, AttributeError, ValueError):
        # Preserve normal Escape handling if Cocoa bindings are unavailable.
        return False
    return False
