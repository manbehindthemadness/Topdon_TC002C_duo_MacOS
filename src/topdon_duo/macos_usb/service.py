"""
Fixed USB-only helper. No UI, processing packages, downloads or user preferences.

The development command authorizes this trusted checkout and its interpreter with
sudo. A distributable helper must be installed with administrator-owned code and
dependencies; a user-writable Python environment is not a permanent root service.
"""

import logging
import os
import sys
import threading
from dataclasses import asdict
from typing import Any, BinaryIO

import usb.core

from ..camera import CameraError, TC002CDuoCamera
from .protocol import receive, send


def integer(message: dict, name: str, minimum: int, maximum: int) -> int:
    """
    Reject booleans, coercions and out-of-range request fields.
    """
    value = message.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid USB helper {name}")
    return value


def transfer(camera: Any, message: dict, payload: bytes) -> tuple[dict, bytes]:
    """
    Forward only the Duo's bounded control-interface requests.
    """
    request_type = integer(message, "request_type", 0, 255)
    request = integer(message, "request", 0, 255)
    value = integer(message, "value", 0, 65535)
    index = integer(message, "index", 0, 65535)
    timeout = integer(message, "timeout", 1, 2000)
    length = integer(message, "length", 0, 4096)
    permitted = {0x21: (1,), 0x41: (1,), 0xA1: (0x81, 0x85), 0xC1: (0x81, 0x85)}
    if index != 0x0A00 or request not in permitted.get(request_type, ()):
        raise ValueError("Unsupported Duo control request")
    reading = bool(request_type & 0x80)
    if (reading and payload) or (not reading and len(payload) != length):
        raise ValueError("Invalid Duo control payload")
    result = camera.device.ctrl_transfer(
        request_type, request, value, index, length if reading else payload, timeout=timeout,
    )
    if reading:
        return {}, bytes(result)
    return {"written": int(result)}, b""


def serve(source: BinaryIO, target: BinaryIO, camera: Any) -> None:
    """
    Retain one USB owner until EOF; use the existing queued capture implementation.
    """
    writing = threading.Lock()
    acquisition: threading.Thread | None = None

    def publish(metadata: dict, body: bytes = b"") -> None:
        """
        Serialize replies and frames on the private output pipe.
        """
        with writing:
            send(target, metadata, body)

    def capture() -> None:
        """
        Publish complete raw frames without decoding or running application callbacks.
        """
        try:
            for frame in camera.frames():
                publish({"kind": "frame"}, frame)
        except (CameraError, OSError, ValueError) as capture_error:
            try:
                publish({"kind": "failure", "error": str(capture_error)})
            except (OSError, ValueError):
                pass  # The peer may have exited; cleanup still runs in the command thread.

    try:
        while True:
            message, payload = receive(source)
            identity = integer(message, "id", 1, 2**31 - 1)
            operation = message.get("operation")
            try:
                extra, response = {}, b""
                if operation == "open":
                    if acquisition is not None or camera.device is not None or payload:
                        raise ValueError("USB helper already opened or invalid open payload")
                    camera.timeout_ms = integer(message, "timeout", 1, 2000)
                    depth = integer(message, "depth", 0, 128)
                    if depth == 1:
                        raise ValueError("Invalid USB queue depth")
                    camera.usb_queue_depth = depth
                    camera.stream_observer = lambda event: publish({"kind": "diagnostics",
                                                                    "event": event})
                    mode = camera.open()
                    extra = {"mode": asdict(mode)}
                elif operation == "start":
                    if camera.device is None or acquisition is not None or payload:
                        raise ValueError("USB helper cannot start capture")
                    worker = threading.Thread(target=capture, name="usb-helper", daemon=True)
                    acquisition = worker
                    worker.start()
                elif operation == "transfer":
                    if camera.device is None:
                        raise ValueError("USB helper is not open")
                    extra, response = transfer(camera, message, payload)
                elif operation == "stop":
                    camera.stop_stream()
                else:
                    raise ValueError("Unknown USB helper operation")
                publish({"kind": "reply", "id": identity, **extra}, response)
            except (CameraError, usb.core.USBError, ValueError, TypeError) as exc:
                publish({"kind": "reply", "id": identity, "error": str(exc),
                         "errno": getattr(exc, "errno", None)})
    except (EOFError, OSError, ValueError, TypeError):
        pass  # Invalid or disconnected clients end this single-session helper.
    finally:
        camera.stop_stream()
        if acquisition is not None:
            acquisition.join(timeout=camera.timeout_ms / 1000 + 1)
        if acquisition is None or not acquisition.is_alive():
            camera.close()


def main() -> int:
    """
    Run only as a macOS authorized helper, with diagnostics on stderr.
    """
    if sys.platform != "darwin" or os.geteuid() != 0:
        print("The USB helper requires macOS USB authorization.", file=sys.stderr)
        return 1
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    serve(sys.stdin.buffer, sys.stdout.buffer, TC002CDuoCamera())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
