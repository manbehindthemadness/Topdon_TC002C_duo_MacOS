"""Direct UVC bulk capture for the TOPDON TC002C Duo on macOS and Linux."""

from __future__ import annotations

import logging
import struct
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Self

import libusb_package
import numpy as np
import usb.core
import usb.util

from .queued_usb import QueuedBulkReader

LOG = logging.getLogger(__name__)

VENDOR_ID = 0x2BDF
PRODUCT_ID = 0x0102
VIDEO_STREAMING_INTERFACE = 1
BULK_ENDPOINT = 0x81

SENSOR_WIDTH = 256
SENSOR_HEIGHT = 192
FRAME_RATE = 25
FRAME_INTERVAL = 400_000
DEFAULT_USB_QUEUE_DEPTH = 32

# Frame index 10 is advertised as the intentionally odd 8x12578 YUY2 mode.
# It is actually a flat array of 100624 little-endian uint16 values:
# telemetry, a 256x192 temperature plane, then a 256x192 preview plane.
FRAME_U16 = 100_624
FRAME_BYTES = FRAME_U16 * 2
FRAME_MAGIC = 0x70827773
HEADER_U16 = 2_320
SENSOR_PIXELS = SENSOR_WIDTH * SENSOR_HEIGHT
TEMPERATURE_OFFSET = HEADER_U16
IMAGE_OFFSET = TEMPERATURE_OFFSET + SENSOR_PIXELS
YUY2_PREVIEW_PIXELS = SENSOR_PIXELS * 4
YUY2_FRAME_BYTES = (IMAGE_OFFSET + YUY2_PREVIEW_PIXELS) * 2

FORMAT_INDEX = 1
FRAME_INDEX = 10
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
    """Raised when USB permissions prevent access to the camera."""


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
        self.last_rejected_size = None
        self.last_rejected_prefix = None
        self.rejected_frame_observer = None
        self.rejected = {
            "invalid_header": 0,
            "uvc_error": 0,
            "partial": 0,
            "size_mismatch": 0,
            "magic_mismatch": 0,
        }

    def feed(self, packet: bytes) -> bytes | None:
        if len(packet) < 2:
            self.rejected["invalid_header"] += 1
            return None

        header_length = packet[0]
        if header_length < 2 or header_length > len(packet):
            self.rejected["invalid_header"] += 1
            return None

        flags = packet[1]
        if flags & 0x40:
            self.rejected["uvc_error"] += 1
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
            self.rejected["partial"] += 1
            self._data.clear()
            return None
        frame = bytes(self._data[: self.frame_size])
        self._data.clear()
        return frame


class LinuxFrameAssembler(FrameAssembler):
    """Keep complete negotiated frames and reject stale/partial USB data."""

    def _finish(self) -> bytes | None:
        if len(self._data) != self.frame_size:
            self.last_rejected_size = len(self._data)
            self.last_rejected_prefix = self._data[:16].hex()
            self.rejected["size_mismatch"] += 1
            if self.rejected_frame_observer is not None:
                # The observer must consume this buffer synchronously; it is cleared below.
                self.rejected_frame_observer(self._data)
            self._data.clear()
            return None
        if not self._data.startswith(struct.pack("<I", FRAME_MAGIC)):
            self.last_rejected_size = len(self._data)
            self.last_rejected_prefix = self._data[:16].hex()
            self.rejected["magic_mismatch"] += 1
            self._data.clear()
            return None
        return super()._finish()


def has_yuy2_preview(frame: bytes) -> bool:
    """Return whether *frame* contains the Ubuntu 512x384 YUY2 preview.

    The shorter macOS layout has the same 256x192 radiometric data followed by
    one 16-bit grayscale sample per pixel.  Treating those samples as pairs of
    YUY2 bytes produces the distinctive one-pixel vertical stripe corruption.
    """
    return len(frame) == YUY2_FRAME_BYTES


def decode_duo_frame(frame: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the telemetry, raw temperature, and preview planes."""
    if len(frame) < FRAME_BYTES:
        raise ValueError(f"short frame: expected {FRAME_BYTES} bytes, got {len(frame)}")
    values = np.frombuffer(frame[:FRAME_BYTES], dtype="<u2")
    magic = int(values[0]) | (int(values[1]) << 16)
    if magic != FRAME_MAGIC:
        raise ValueError(f"invalid Duo frame magic: 0x{magic:08x}")
    telemetry = values[:HEADER_U16].copy()
    temperatures = (
        values[TEMPERATURE_OFFSET : TEMPERATURE_OFFSET + SENSOR_PIXELS]
        .reshape(SENSOR_HEIGHT, SENSOR_WIDTH)
        .copy()
    )
    # Linux also exposes a complete 512x384 YUY2 grayscale preview. Each
    # little-endian word contains luminance in its low byte and chroma above.
    preview_pixels = YUY2_PREVIEW_PIXELS if has_yuy2_preview(frame) else SENSOR_PIXELS
    preview_scale = 2 if preview_pixels == YUY2_PREVIEW_PIXELS else 1
    preview_words = np.frombuffer(frame, dtype="<u2", count=preview_pixels, offset=IMAGE_OFFSET * 2)
    preview = (
        (preview_words & 0xFF)
        .astype(np.uint8)
        .reshape(SENSOR_HEIGHT * preview_scale, SENSOR_WIDTH * preview_scale)
    )
    return telemetry, temperatures, preview


def estimate_temperature_offset(raw: np.ndarray, ambient_celsius: float = 22.0) -> float:
    """Anchor the stable cold-background percentile to ambient temperature."""
    return float(np.percentile(raw, 2.0)) / 64.0 - ambient_celsius


def measurement_frame_status(telemetry: np.ndarray, raw: np.ndarray) -> str:
    """Recognize mode-8 frozen radiometry and impossible endpoint counts.

    The related HCUSBSDK calls this dwIsFreezedata (1 frozen, 0 live).
    Byte 32 was observed as 1 through a shutter cycle, with the temperature
    plane repeated exactly for 16 frames; normal frames carry 0. Avoid using
    scene changes or delayed telemetry min/max as a calibration detector.
    """
    frozen = int(telemetry[16]) | (int(telemetry[17]) << 16)
    if frozen:
        return "Auto calibrate; readings held"
    invalid = int(np.count_nonzero((raw == 0) | (raw == 65535)))
    if invalid:
        return f"Invalid temperature frame ({invalid} invalid pixels); readings held"
    return ""


def raw_temperatures(
    raw: np.ndarray, ambient_celsius: float = 22.0, offset: float | None = None
) -> np.ndarray:
    """Convert Duo raw counts to apparent Celsius using its 1/64 °C gain."""
    raw_float = np.asarray(raw, dtype=np.float32)
    if offset is None:
        offset = estimate_temperature_offset(raw_float, ambient_celsius)
    return raw_float / 64.0 - np.float32(offset)


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
        self.stream_observer = None
        self.rejected_frame_observer = None
        self.usb_queue_depth = (
            DEFAULT_USB_QUEUE_DEPTH
            if sys.platform == "darwin" or sys.platform.startswith("linux")
            else 0
        )

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
        strings = {}
        for name in ("manufacturer", "product", "serial_number"):
            try:
                strings[name] = getattr(device, name)
            except (usb.core.USBError, ValueError) as exc:
                strings[name] = f"unavailable: {exc}"
        return {
            "connected": True,
            "usb_id": f"{device.idVendor:04x}:{device.idProduct:04x}",
            "manufacturer": strings["manufacturer"],
            "product": strings["product"],
            "serial": strings["serial_number"],
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
            self.device = device
            if sys.platform.startswith("linux"):
                # Linux already configures UVC devices and binds uvcvideo.
                # Re-setting that configuration can fail with EBUSY before
                # we have a chance to detach the kernel driver.
                try:
                    device.get_active_configuration()
                except usb.core.USBError as exc:
                    if exc.strerror != "Configuration not set":
                        raise
                    device.set_configuration()
            else:
                device.set_configuration()
            for interface_number in (0, VIDEO_STREAMING_INTERFACE):
                self._detach_and_claim(device, interface_number)
            self.mode = self._negotiate()
            self._running.set()
            LOG.info(
                "Negotiated TC002C Duo radiometric mode at %.1f fps (payload %d bytes)",
                self.mode.fps,
                self.mode.max_payload_size,
            )
            return self.mode
        except usb.core.USBError as exc:
            self.close()
            if getattr(exc, "errno", None) in (1, 13) or "Access denied" in str(exc):
                if sys.platform.startswith("linux"):
                    raise CameraAccessError(
                        "USB access denied. Install the rule in "
                        "packaging/udev/70-topdon-duo.rules, reload udev rules, "
                        "and reconnect the camera (see README.md)."
                    ) from exc
                raise CameraAccessError(
                    "macOS owns the camera's UVC interface. Run: sudo .venv/bin/topdon-duo"
                ) from exc
            raise CameraError(f"USB setup failed: {exc}") from exc
        except Exception:
            self.close()
            raise

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
                "camera rejected TC002C Duo radiometric mode "
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
        if sys.platform.startswith("linux"):
            # Linux can negotiate a larger preview plane. The radiometric
            # plane stays at the same offset, but the entire USB frame must
            # arrive before it is safe to use it. Keep macOS assembly intact.
            if self.mode.max_frame_size < FRAME_BYTES:
                raise CameraError(f"invalid negotiated frame size: {self.mode.max_frame_size}")
            assembler = LinuxFrameAssembler(self.mode.max_frame_size)
        else:
            # Preserve the proven macOS framing behavior. The device can append
            # padding beyond the radiometric frame; queued requests prevent the
            # host gaps that previously caused it to become desynchronized.
            assembler = FrameAssembler()
        assembler.rejected_frame_observer = self.rejected_frame_observer
        read_size = max(16_384, self.mode.max_payload_size)
        observer = self.stream_observer
        totals = {"packets": 0, "bytes": 0, "timeouts": 0, "frames": 0}
        next_report = 0.0
        packet_lengths = {}
        packet_headers = {}
        longest_read = longest_gap = 0.0
        last_read_finished = None

        def report():
            nonlocal next_report, longest_read, longest_gap
            now = time.monotonic()
            if now >= next_report:
                observer(
                    {
                        **totals,
                        "rejected": assembler.rejected.copy(),
                        "expected_frame_bytes": assembler.frame_size,
                        "buffered_bytes": len(assembler._data),
                        "last_rejected_size": assembler.last_rejected_size,
                        "last_rejected_prefix": assembler.last_rejected_prefix,
                        "packet_lengths": packet_lengths.copy(),
                        "packet_headers": packet_headers.copy(),
                        "longest_read_seconds": round(longest_read, 6),
                        "longest_host_gap_seconds": round(longest_gap, 6),
                        "usb_queue_depth": self.usb_queue_depth,
                        "transfer_statuses": dict(reader.statuses) if reader else {},
                    }
                )
                next_report = now + 1
                packet_lengths.clear()
                packet_headers.clear()
                longest_read = longest_gap = 0.0

        transport = (
            QueuedBulkReader(
                self.device, BULK_ENDPOINT, read_size, self.timeout_ms, self.usb_queue_depth
            )
            if self.usb_queue_depth else nullcontext(None)
        )
        with transport as reader:
            while self._running.is_set():
                if observer is not None:
                    read_started = time.perf_counter()
                    if last_read_finished is not None:
                        longest_gap = max(longest_gap, read_started - last_read_finished)
                try:
                    packet = (
                        reader.read() if reader is not None
                        else bytes(self.device.read(BULK_ENDPOINT, read_size, timeout=self.timeout_ms))
                    )
                except usb.core.USBTimeoutError:
                    if observer is not None:
                        last_read_finished = time.perf_counter()
                        longest_read = max(longest_read, last_read_finished - read_started)
                        totals["timeouts"] += 1
                        report()
                    continue
                except usb.core.USBError as exc:
                    if not self._running.is_set():
                        break
                    raise CameraError(f"USB stream read failed: {exc}") from exc
                if observer is not None:
                    last_read_finished = time.perf_counter()
                    longest_read = max(longest_read, last_read_finished - read_started)
                    length = str(len(packet))
                    header = packet[:2].hex()
                    packet_lengths[length] = packet_lengths.get(length, 0) + 1
                    packet_headers[header] = packet_headers.get(header, 0) + 1
                frame = assembler.feed(packet)
                if observer is not None:
                    totals["packets"] += 1
                    totals["bytes"] += len(packet)
                    totals["frames"] += int(frame is not None)
                    report()
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
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        return "Direct UVC capture is supported on macOS and Linux; this platform is untested."
    return None
