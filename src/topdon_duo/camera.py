"""Direct UVC bulk capture for the TOPDON TC002C Duo on macOS."""

from __future__ import annotations

import logging
import struct
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import libusb_package
import numpy as np
import usb.core
import usb.util

LOG = logging.getLogger(__name__)

VENDOR_ID = 0x2BDF
PRODUCT_ID = 0x0102
VIDEO_STREAMING_INTERFACE = 1
BULK_ENDPOINT = 0x81

FRAME_WIDTH = 256
FRAME_HEIGHT = 392
PANE_HEIGHT = FRAME_HEIGHT // 2
FRAME_RATE = 25
FRAME_INTERVAL = 400_000
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 2

FORMAT_INDEX = 1
FRAME_INDEX = 1
PROBE_LENGTH = 34
SET_CUR = 0x01
GET_CUR = 0x81
REQUEST_TYPE_SET = 0x21
REQUEST_TYPE_GET = 0xA1
VS_PROBE_CONTROL = 0x01
VS_COMMIT_CONTROL = 0x02


class CameraError(RuntimeError):
    """Base error for camera discovery, setup, or streaming."""


class CameraAccessError(CameraError):
    """Raised when macOS will not release the camera interface."""


@dataclass(frozen=True)
class NegotiatedMode:
    format_index: int
    frame_index: int
    frame_interval: int
    max_frame_size: int
    max_payload_size: int

    @property
    def fps(self) -> float:
        return 10_000_000 / self.frame_interval if self.frame_interval else 0.0


class FrameAssembler:
    """Reassemble UVC payloads into complete 256x392 YUY2 frames."""

    def __init__(self, frame_size: int = FRAME_BYTES) -> None:
        self.frame_size = frame_size
        self._fid: int | None = None
        self._data = bytearray()

    def feed(self, packet: bytes) -> bytes | None:
        if len(packet) < 2:
            return None

        header_length = packet[0]
        if header_length < 2 or header_length > len(packet):
            return None

        flags = packet[1]
        if flags & 0x40:
            self._data.clear()
            self._fid = flags & 1
            return None

        fid = flags & 1
        completed = None
        if self._fid is None:
            self._fid = fid
        elif fid != self._fid:
            completed = self._finish()
            self._fid = fid

        self._data.extend(packet[header_length:])
        if flags & 0x02:
            completed = self._finish() or completed
            self._fid = None
        return completed

    def _finish(self) -> bytes | None:
        if len(self._data) < self.frame_size:
            self._data.clear()
            return None
        frame = bytes(self._data[: self.frame_size])
        self._data.clear()
        return frame


def decode_yuy2_frame(frame: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return the top YUY2 pane and bottom radiometric pane from a frame."""
    if len(frame) < FRAME_BYTES:
        raise ValueError(f"short frame: expected {FRAME_BYTES} bytes, got {len(frame)}")
    array = np.frombuffer(frame[:FRAME_BYTES], dtype=np.uint8).reshape(
        FRAME_HEIGHT, FRAME_WIDTH, 2
    )
    return array[:PANE_HEIGHT].copy(), array[PANE_HEIGHT:].copy()


def raw_temperatures(radiometric: np.ndarray) -> np.ndarray:
    """Decode little-endian 16-bit radiometric values to uncalibrated Celsius."""
    raw = radiometric[..., 0].astype(np.uint16)
    raw |= radiometric[..., 1].astype(np.uint16) << 8
    return raw.astype(np.float32) / 64.0 - 273.15


def build_probe() -> bytearray:
    probe = bytearray(PROBE_LENGTH)
    struct.pack_into("<H", probe, 0, 0x0001)
    probe[2] = FORMAT_INDEX
    probe[3] = FRAME_INDEX
    struct.pack_into("<I", probe, 4, FRAME_INTERVAL)
    return probe


def parse_probe(data: bytes) -> NegotiatedMode:
    if len(data) < 26:
        raise CameraError(f"camera returned a short UVC probe ({len(data)} bytes)")
    values = struct.unpack("<HBBIHHHHHII", data[:26])
    return NegotiatedMode(
        format_index=values[1],
        frame_index=values[2],
        frame_interval=values[3],
        max_frame_size=values[9],
        max_payload_size=values[10],
    )


class TC002CDuoCamera:
    """Single-owner direct USB connection to the Duo's UVC bulk stream."""

    def __init__(self, timeout_ms: int = 2_000) -> None:
        self.timeout_ms = timeout_ms
        self.device = None
        self.mode: NegotiatedMode | None = None
        self._claimed: list[int] = []
        self._detached: list[int] = []
        self._running = threading.Event()

    @staticmethod
    def find():
        backend = libusb_package.get_libusb1_backend()
        return usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID, backend=backend)

    @classmethod
    def diagnostics(cls) -> dict[str, object]:
        device = cls.find()
        if device is None:
            return {"connected": False, "usb_id": f"{VENDOR_ID:04x}:{PRODUCT_ID:04x}"}
        interfaces = []
        for config in device:
            for interface in config:
                interfaces.append(
                    {
                        "number": interface.bInterfaceNumber,
                        "alternate": interface.bAlternateSetting,
                        "class": interface.bInterfaceClass,
                        "subclass": interface.bInterfaceSubClass,
                        "endpoints": [f"0x{ep.bEndpointAddress:02x}" for ep in interface],
                    }
                )
        return {
            "connected": True,
            "usb_id": f"{device.idVendor:04x}:{device.idProduct:04x}",
            "manufacturer": str(device.manufacturer),
            "product": str(device.product),
            "serial": str(device.serial_number),
            "interfaces": interfaces,
        }

    def open(self) -> NegotiatedMode:
        if self.device is not None:
            return self.mode  # type: ignore[return-value]
        device = self.find()
        if device is None:
            raise CameraError(
                f"TC002C Duo {VENDOR_ID:04x}:{PRODUCT_ID:04x} not found; reconnect it and wait"
            )

        try:
            device.set_configuration()
            self.device = device
            for interface_number in (0, VIDEO_STREAMING_INTERFACE):
                self._detach_and_claim(device, interface_number)
            self.mode = self._negotiate()
            self._running.set()
            LOG.info(
                "Negotiated %dx%d at %.1f fps (payload %d bytes)",
                FRAME_WIDTH,
                FRAME_HEIGHT,
                self.mode.fps,
                self.mode.max_payload_size,
            )
            return self.mode
        except usb.core.USBError as exc:
            self.close()
            if getattr(exc, "errno", None) in (1, 13) or "Access denied" in str(exc):
                raise CameraAccessError(
                    "macOS owns the camera's UVC interface. Run: sudo .venv/bin/topdon-duo"
                ) from exc
            raise CameraError(f"USB setup failed: {exc}") from exc

    def _detach_and_claim(self, device, interface_number: int) -> None:
        try:
            if device.is_kernel_driver_active(interface_number):
                device.detach_kernel_driver(interface_number)
                self._detached.append(interface_number)
        except NotImplementedError:
            pass
        usb.util.claim_interface(device, interface_number)
        self._claimed.append(interface_number)

    def _negotiate(self) -> NegotiatedMode:
        probe = build_probe()
        written = self.device.ctrl_transfer(
            REQUEST_TYPE_SET,
            SET_CUR,
            VS_PROBE_CONTROL << 8,
            VIDEO_STREAMING_INTERFACE,
            probe,
            self.timeout_ms,
        )
        if written != PROBE_LENGTH:
            raise CameraError(f"SET_CUR(PROBE) wrote {written}/{PROBE_LENGTH} bytes")

        response = bytes(
            self.device.ctrl_transfer(
                REQUEST_TYPE_GET,
                GET_CUR,
                VS_PROBE_CONTROL << 8,
                VIDEO_STREAMING_INTERFACE,
                PROBE_LENGTH,
                self.timeout_ms,
            )
        )
        mode = parse_probe(response)
        if (mode.format_index, mode.frame_index) != (FORMAT_INDEX, FRAME_INDEX):
            raise CameraError(
                "camera rejected 256x392 mode "
                f"(returned format {mode.format_index}, frame {mode.frame_index})"
            )

        committed = self.device.ctrl_transfer(
            REQUEST_TYPE_SET,
            SET_CUR,
            VS_COMMIT_CONTROL << 8,
            VIDEO_STREAMING_INTERFACE,
            response,
            self.timeout_ms,
        )
        if committed != len(response):
            raise CameraError(f"SET_CUR(COMMIT) wrote {committed}/{len(response)} bytes")
        return mode

    def frames(self) -> Iterator[bytes]:
        if self.device is None or self.mode is None:
            self.open()
        assembler = FrameAssembler()
        read_size = max(16_384, self.mode.max_payload_size)
        while self._running.is_set():
            try:
                packet = bytes(
                    self.device.read(BULK_ENDPOINT, read_size, timeout=self.timeout_ms)
                )
            except usb.core.USBTimeoutError:
                continue
            except usb.core.USBError as exc:
                if not self._running.is_set():
                    break
                raise CameraError(f"USB stream read failed: {exc}") from exc
            frame = assembler.feed(packet)
            if frame is not None:
                yield frame

    def close(self) -> None:
        self._running.clear()
        device, self.device = self.device, None
        if device is None:
            return
        for interface_number in reversed(self._claimed):
            try:
                usb.util.release_interface(device, interface_number)
            except usb.core.USBError:
                pass
        for interface_number in reversed(self._detached):
            try:
                device.attach_kernel_driver(interface_number)
            except (NotImplementedError, usb.core.USBError):
                pass
        self._claimed.clear()
        self._detached.clear()
        usb.util.dispose_resources(device)

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def platform_warning() -> str | None:
    if sys.platform != "darwin":
        return "This package targets macOS; direct UVC capture on this platform is untested."
    return None
