"""Verified, reversible UVC controls, used by the stream's single USB owner."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import usb.core

from .camera import CameraError
from .tone_curves import composite_curve

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareControl:
    title: str
    selector: int
    command: int
    offset: int
    minimum: float
    maximum: float
    step: float = 1
    scale: float = 1
    bias: float = 0
    fmt: str = "B"
    unit: str = ""
    options: tuple[tuple[int, str], ...] = ()
    extra: tuple[tuple[int, int], ...] = ()

    def decode(self, payload: bytes) -> float:
        return struct.unpack_from("<" + self.fmt, payload, self.offset)[0] / self.scale - self.bias

    def apply(self, payload: bytearray, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{self.title} needs a number")
        if not math.isfinite(value) or not self.minimum <= value <= self.maximum:
            raise ValueError(f"{self.title} must be between {self.minimum} and {self.maximum}")
        if self.options and value not in dict(self.options):
            raise ValueError(f"Invalid {self.title} selection")
        encoded = (value + self.bias) * self.scale
        if not math.isclose(encoded, round(encoded), abs_tol=1e-6):
            raise ValueError(f"Invalid precision for {self.title}")
        struct.pack_into("<" + self.fmt, payload, self.offset, round(encoded))
        for offset, setting in self.extra:
            payload[offset] = setting


# Descriptive names from sampled TC002C Duo output; USB palette IDs stay unchanged.
PALETTES = (
    (1, "White hot"),
    (2, "Black hot"),
    (10, "Violet iron"),
    (11, "Classic rainbow"),
    (12, "Fire"),
    (13, "Sunset"),
    (14, "Soft iron"),
    (15, "Amber"),
    (16, "Vivid rainbow"),
    (17, "Neon sunset"),
    (18, "Silver"),
    (19, "Pastel rainbow"),
    (20, "Red hot"),
    (21, "Green hot"),
    (22, "Ocean heat"),
)
HARDWARE_CONTROLS = {
    "ambient": HardwareControl(
        "Ambient temperature", 3, 1, 76, -50, 100, 0.1, 100, 100, "I", "°C", extra=((75, 2),)
    ),
    "distance": HardwareControl(
        "Distance to spot", 3, 1, 21, 0.3, 99, 0.01, 100, fmt="I", unit="m"
    ),
    "emissivity": HardwareControl("Emissivity", 3, 1, 16, 0.01, 1, 0.01, 100, fmt="I"),
    "reflected": HardwareControl(
        "Reflected temperature", 3, 1, 26, -50, 100, 0.1, 10, 100, "I", "°C", extra=((25, 1),)
    ),
    "transmission": HardwareControl("Optical transmission", 3, 1, 40, 1, 100, fmt="I", unit="%"),
    "humidity": HardwareControl("Relative humidity", 3, 1, 69, 0, 100, 0.1, 10, fmt="I", unit="%"),
    "center_overlay": HardwareControl(
        "Camera center overlay", 3, 1, 44, 0, 1, options=((0, "Off"), (1, "On"))
    ),
    "brightness": HardwareControl("Brightness", 2, 1, 1, 0, 100),
    "contrast": HardwareControl("Contrast", 2, 2, 1, 0, 100),
    "noise_mode": HardwareControl(
        "Noise reduction mode", 2, 5, 1, 0, 2, options=((0, "Off"), (1, "General"), (2, "Expert"))
    ),
    "noise_general": HardwareControl("General noise reduction", 2, 5, 2, 0, 100),
    "noise_spatial": HardwareControl("Spatial noise reduction (expert)", 2, 5, 3, 0, 100),
    "noise_temporal": HardwareControl("Temporal noise reduction (expert)", 2, 5, 4, 0, 100),
    "detail": HardwareControl("Detail enhancement", 2, 5, 7, 0, 100, extra=((6, 1),)),
    "detail_enabled": HardwareControl(
        "Detail enhancement mode", 2, 5, 6, 0, 1, options=((0, "Off"), (1, "On"))
    ),
    "palette": HardwareControl("Camera color palette", 2, 5, 5, 1, 22, options=PALETTES),
}
BLOCK_LENGTHS = {(2, 1): 2, (2, 2): 2, (2, 5): 79, (3, 1): 80}
PROCESSING_PRESETS = {"balanced": 1, "shadow": 2, "soft": 0}


def validate_fixed_range_bounds(bounds: object) -> tuple[int, int]:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("Fixed range needs lower and upper bounds")
    lower, upper = bounds
    if type(lower) is not int or type(upper) is not int or not 0 <= lower < upper < 0x4000:
        raise ValueError("Fixed range needs integer bounds: 0 <= lower < upper <= 16383")
    return lower, upper


class HardwareControls:
    def __init__(self, camera) -> None:
        self.camera = camera
        self.original: dict[tuple[int, int], bytes] = {}
        self.enabled: set[str] = set()
        self.values: dict[str, float] = {}
        self.error = ""
        self.snapshot_path: Path | None = None
        self.auto_calibrate: bool | None = None
        self._auto_calibrate_owned = False
        self.fixed_range = False
        self._fixed_range_owned = False
        self._fixed_range_bounds: tuple[int, int] | None = None
        self.processing_preset = "balanced"
        self._processing_preset_owned = False
        self.gamma = 50
        self.boost = False
        self._tone_owned = False
        self._tone_queue: list[int] = []
        self._tone_sent = 0

    @property
    def tone_busy(self) -> bool:
        return bool(self._tone_queue)

    def _tone_command(self, body: bytes, replies: tuple[bytes, ...]) -> None:
        packet = bytes((0xF0, len(body))) + body + bytes((sum(body) & 255, 0xFF))
        try:
            device = self.camera.device
            if device.ctrl_transfer(0x41, 1, 0, 0x0A00, packet, timeout=1000) != len(packet):
                raise CameraError("Incomplete camera tone command")
            for expected in replies:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    length = bytes(device.ctrl_transfer(0xC1, 0x85, 0, 0x0A00, 4, timeout=1000))
                    size = int.from_bytes(length, "little")
                    if len(length) not in (2, 4) or size > 128:
                        raise CameraError("Invalid camera tone mailbox")
                    if size:
                        reply = bytes(device.ctrl_transfer(0xC1, 0x81, 0, 0x0A00, size, timeout=1000))
                        if reply != expected:
                            raise CameraError("Unexpected camera tone reply")
                        break
                    time.sleep(0.005)
                else:
                    raise CameraError("Camera tone command timed out")
        except (usb.core.USBError, AttributeError) as exc:
            raise CameraError(f"Camera tone transfer failed: {exc}") from exc

    def _apply_boost(self, enabled: bool) -> None:
        # This handler rebuilds before staging and emits success THEN a known
        # fall-through error. Consume both replies for EACH of the two applies.
        for _ in range(2):
            self._tone_command(
                bytes((0x36, 0x78, 0x31, 0, 3 if enabled else 0)),
                (bytes.fromhex("f0053678310301e3ff"), bytes.fromhex("f0053678310400e3ff")),
            )

    def set_tone(self, gamma: int, boost: bool) -> None:
        if type(gamma) is not int or not 0 <= gamma <= 100 or type(boost) is not bool:
            raise ValueError("Gamma needs an integer 0..100 and boost needs a boolean")
        if self._fixed_range_owned:
            raise CameraError("Turn off Fixed mode before changing gamma or boost")
        if gamma == 50 and not boost:
            self.restore_tone()
            return
        self.load()
        if not self._tone_owned:
            self._fixed_range_baseline()  # Only the exact audited factory ISP.
            if self.original[2, 5][23] != 1:
                raise CameraError("Camera tone controls require the tested ISP baseline")
        self._tone_owned = True  # Retain cleanup responsibility after failure.
        try:
            if gamma == 50 or boost != self.boost or not self._tone_queue:
                self._apply_boost(boost)
            self.gamma, self.boost = gamma, boost
            self._tone_sent = 0
            if gamma == 50:
                self._tone_queue = []  # Native boost refresh already composes this.
            else:
                contrast = round(self.state()["contrast"]["value"])
                curve = composite_curve(gamma, boost, contrast, self.processing_preset)
                self._tone_queue = [0x80000000 | (value << 16) | i for i, value in enumerate(curve)] + [0]
        except CameraError:
            self.restore_tone()
            raise

    def advance_tone(self) -> bool:
        """One LUT entry per viewer iteration; keep rendering and sampling alive."""
        if not self._tone_queue:
            return False
        try:
            value = self._tone_queue[0]
            self._tone_command(
                bytes.fromhex("36741300") + struct.pack(">II", 0x206110, value),
                (bytes.fromhex("f0053674130301c1ff"),),
            )
            self._tone_queue.pop(0)
            self._tone_sent += 1
            if not self._tone_queue:
                # Direct2090a4 is rejected; normal brightness refresh preserves
                # the uploaded composite and the current brightness setting.
                self.write((2, 1), self.read(2, 1))
                return True
        except CameraError:
            self.restore_tone()
            raise
        return False

    def restore_tone(self) -> None:
        self._tone_queue = []
        if self._tone_owned:
            self._apply_boost(False)  # Native builder, not a guessed LUT backup.
            self._tone_owned = False
        self.gamma, self.boost = 50, False

    def _processing_command(self, mode: int | None = None) -> int:
        body = bytes((0x36, 0x23, 1 if mode is None else 0))
        if mode is not None:
            if mode not in PROCESSING_PRESETS.values():
                raise ValueError("Unsupported camera processing mode")
            body += struct.pack(">H", mode)
        packet = bytes((0xF0, len(body))) + body + bytes((sum(body) & 255, 0xFF))
        try:
            device = self.camera.device
            if device.ctrl_transfer(0x41, 1, 0, 0x0A00, packet, timeout=2000) != len(packet):
                raise CameraError("Incomplete processing preset command")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                length = bytes(device.ctrl_transfer(0xC1, 0x85, 0, 0x0A00, 4, timeout=2000))
                size = int.from_bytes(length, "little")
                if len(length) not in (2, 4) or size > 128:
                    raise CameraError("Unsupported processing preset mailbox")
                if size:
                    reply = bytes(device.ctrl_transfer(0xC1, 0x81, 0, 0x0A00, size, timeout=2000))
                    expected_body = bytes((0x36, 0x23, 3))
                    data_size = 2 if mode is None else 1
                    if (len(reply) != data_size + 7 or reply[0] != 0xF0
                            or reply[1] != len(reply) - 4 or reply[-1] != 0xFF
                            or reply[2:5] != expected_body
                            or sum(reply[2:-2]) & 255 != reply[-2]):
                        raise CameraError("Invalid processing preset acknowledgement")
                    value = int.from_bytes(reply[5:-2], "big")
                    if mode is not None and value != 1:
                        raise CameraError("Camera rejected processing preset")
                    return value
                time.sleep(0.04)
            raise CameraError("Processing preset command timed out")
        except (usb.core.USBError, AttributeError) as exc:
            raise CameraError(f"Processing preset transfer failed: {exc}") from exc

    def _apply_processing_preset(self, preset: str) -> None:
        mode = PROCESSING_PRESETS[preset]
        self._processing_command(mode)
        # The getter returns a bank, not the setter's mode. Gain selects
        # a second set of banks; never use the returned bank as a restore mode.
        if self._processing_command() not in {1: (1, 4), 2: (2, 5), 0: (3, 6)}[mode]:
            raise CameraError("Camera processing bank did not match the selected preset")

    def set_processing_preset(self, preset: str) -> None:
        if not isinstance(preset, str) or preset not in PROCESSING_PRESETS:
            raise ValueError("Unknown camera processing preset")
        if self._fixed_range_owned:
            raise CameraError("Turn off fixed mode before changing processing presets")
        if preset == "balanced":
            self.restore_processing_preset()
            return
        self.load()
        if not self._processing_preset_owned:
            self._fixed_range_baseline()  # Same verified factory ISP table.
            if self.original[2, 5][23] != 1 or self._processing_command() != 1:
                raise CameraError("Processing presets require the tested Balanced camera baseline")
        self._processing_preset_owned = True
        try:
            self._apply_processing_preset(preset)
        except CameraError:
            self.restore_processing_preset()
            raise
        self.processing_preset = preset
        if self._tone_owned:
            self.set_tone(self.gamma, self.boost)

    def restore_processing_preset(self) -> None:
        if self._processing_preset_owned:
            self._apply_processing_preset("balanced")
            self._processing_preset_owned = False
        self.processing_preset = "balanced"

        if self._tone_owned:
            self.set_tone(self.gamma, self.boost)

    def _fixed_range_baseline(self) -> tuple[int, int]:
        """Only enable the experimental path with the exact tested ISP preset."""
        if self._fixed_range_bounds is not None:
            return self._fixed_range_bounds
        self.load()
        if self._transfer(0x21, 1, 5, b"\x01\x10") != 2:
            raise CameraError("Incomplete ISP export selection")
        length = bytes(self._transfer(0xA1, 0x85, 1, 4))
        if int.from_bytes(length, "little") != 5:
            raise CameraError("Unsupported ISP export metadata")
        metadata = bytes(self._transfer(0xA1, 0x81, 1, 5))
        if metadata != b"\x01\x74\x0f\x00\x00":
            raise CameraError("Fixed range requires the tested Duo ISP layout")
        export = bytearray()
        for sequence in range(1, 9):
            packet = bytes(self._transfer(0xA1, 0x81, 1, 512))
            if len(packet) <= 5 or packet[0] != 2 or int.from_bytes(packet[1:5], "little") != sequence:
                raise CameraError("Invalid ISP export sequence")
            export.extend(packet[5:])
        if hashlib.sha256(export).hexdigest() != "b1ae66b05878b4d44b68697b4ca185c482b9d8c69b1ad51d9863387908f76e90":
            raise CameraError("Fixed range requires the verified factory ISP preset")
        # Bounds are public parameters. Preserve them separately: disable restores
        # the four cached processing controls, but does not restore the bounds.
        parameters = dict(struct.iter_unpack("<II", export[20:20 + 197 * 8]))
        bounds = parameters[0x20603C], parameters[0x206040]
        if bounds != (1000, 2800):
            raise CameraError("Unexpected original fixed-range bounds")
        self._fixed_range_bounds = bounds
        return bounds

    def _fixed_range_command(self, lower: int, upper: int) -> None:
        if (lower, upper) != (0, 0):
            validate_fixed_range_bounds((lower, upper))
        body = b"\x36\xfe\x00" + struct.pack(">IHH", 0xF113, lower, upper)
        packet = bytes((0xF0, len(body))) + body + bytes((sum(body) & 255, 0xFF))
        try:
            device = self.camera.device
            if device.ctrl_transfer(0x41, 1, 0, 0x0A00, packet, timeout=2000) != len(packet):
                raise CameraError("Incomplete fixed-range command")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                length = bytes(device.ctrl_transfer(0xC1, 0x85, 0, 0x0A00, 4, timeout=2000))
                size = int.from_bytes(length, "little")
                if len(length) not in (2, 4) or size > 128:
                    raise CameraError("Unsupported fixed-range mailbox")
                if size:
                    reply = bytes(device.ctrl_transfer(0xC1, 0x81, 0, 0x0A00, size, timeout=2000))
                    if reply != bytes.fromhex("f00736fe030000000138ff"):
                        raise CameraError("Unexpected fixed-range acknowledgement")
                    LOG.info("Fixed-range command acknowledged: lower=%d upper=%d (raw units)", lower, upper)
                    return
                time.sleep(0.04)
            raise CameraError("Fixed-range command timed out")
        except (usb.core.USBError, AttributeError) as exc:
            raise CameraError(f"Fixed-range transfer failed: {exc}") from exc

    def set_fixed_range(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("Fixed range state must be a boolean")
        if not enabled:
            self.restore_fixed_range()
            return
        self.load()
        if self.processing_preset != "balanced":
            raise CameraError("Select the Balanced processing preset before enabling fixed mode")
        if self._tone_owned:
            raise CameraError("Set gamma to 50 and turn boost off before enabling Fixed mode")
        if self.state()["detail_enabled"]["value"] != 1:
            raise CameraError("Enable detail enhancement before enabling fixed mode")
        self._fixed_range_baseline()
        self._fixed_range_owned = True  # Retain cleanup responsibility after failure.
        try:
            self._fixed_range_command(0, 16383)
            self._fixed_range_command(0, 16383)  # Second enable latches the update.
        except CameraError:
            self.restore_fixed_range()
            raise
        self.fixed_range = True

    def restore_fixed_range(self) -> None:
        if self._fixed_range_owned:
            try:
                self._fixed_range_command(*self._fixed_range_bounds)
            finally:
                self._fixed_range_command(0, 0)
            self._fixed_range_owned = False
            self.fixed_range = False

    def _transfer(self, request_type, request, selector, data):
        try:
            return self.camera.device.ctrl_transfer(
                request_type, request, selector << 8, 0x0A00, data, timeout=2000
            )
        except (usb.core.USBError, AttributeError) as exc:
            raise CameraError(f"Camera control transfer failed: {exc}") from exc

    def _select(self, selector, command, delay=0) -> int:
        if self._transfer(0x21, 1, 5, bytes((selector, command))) != 2:
            raise CameraError("Incomplete camera command selection")
        if delay:
            time.sleep(delay)
        response = bytes(self._transfer(0xA1, 0x85, selector, 4))
        size = int.from_bytes(response, "little")
        expected = {**BLOCK_LENGTHS, (1, 24): 11, (2, 4): 1}[selector, command]
        if len(response) not in (2, 4) or size != expected:
            raise CameraError(
                "Camera returned an unsupported control layout "
                f"for {selector}:{command}: expected {expected} bytes, "
                f"received length {size} ({response.hex()})"
            )
        return size

    def read(self, selector, command) -> bytes:
        size = self._select(selector, command, delay=0.1)
        payload = bytes(self._transfer(0xA1, 0x81, selector, size))
        if len(payload) != size:
            raise CameraError("Incomplete camera configuration")
        return payload

    def write(self, key, payload) -> None:
        size = self._select(*key)
        if len(payload) != size or self._transfer(0x21, 1, key[0], payload) != size:
            raise CameraError("Incomplete camera configuration write")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.read(*key) == payload:
                return
        raise CameraError("Camera did not apply this setting")

    def _calibration_command(self, selector: int, command: int, payload: bytes) -> None:
        size = self._select(selector, command)
        if size != len(payload) or self._transfer(0x21, 1, selector, payload) != size:
            raise CameraError("Incomplete camera calibration command")
        # Selector 6 is the direct command status control. Reselecting a command
        # here can interfere with pending serial commands; the mailbox getter
        # cannot read back the auto-calibration setting on this firmware.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            length = bytes(self._transfer(0xA1, 0x85, 6, 4))
            if len(length) not in (2, 4) or int.from_bytes(length, "little") != 1:
                raise CameraError("Unsupported camera command status layout")
            status = bytes(self._transfer(0xA1, 0x81, 6, 1))
            if status == b"\x00":
                return
            if status != b"\x01":
                raise CameraError(f"Camera calibration command rejected (status {status.hex()})")
            time.sleep(0.05)
        raise CameraError("Camera calibration command timed out")

    def set_auto_calibrate(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("Auto calibrate state must be a boolean")
        self._auto_calibrate_owned = True
        self._calibration_command(1, 24, struct.pack("<BHII", 2, 0, 0x2001, int(enabled)))
        self.auto_calibrate = enabled

    def calibrate_now(self) -> None:
        self._calibration_command(2, 4, b"\x01")

    def restore_auto_calibrate(self) -> None:
        # The getter is unavailable; return to the observed original automatic
        # operation when relinquishing ownership, including after failed writes.
        if self._auto_calibrate_owned:
            self.set_auto_calibrate(True)
            self._auto_calibrate_owned = False

    def load(self) -> None:
        if self.original:
            return
        # Publish a baseline only after all required blocks are validated.
        original = {key: self.read(*key) for key in BLOCK_LENGTHS}
        values = {
            name: spec.decode(original[spec.selector, spec.command])
            for name, spec in HARDWARE_CONTROLS.items()
        }
        state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        directory = state_root / "topdon-duo" / "camera-baselines"
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            snapshot_path = directory / f"{uuid4().hex}.json"
            with snapshot_path.open("x") as snapshot:
                json.dump(
                    {
                        "usb_id": "2bdf:0102",
                        "created_unix": time.time(),
                        "blocks": {f"{s}:{c}": data.hex() for (s, c), data in original.items()},
                        "values": values,
                    },
                    snapshot,
                    indent=2,
                )
        except OSError as exc:
            raise CameraError(f"Could not preserve original camera settings: {exc}") from exc
        self.snapshot_path = snapshot_path
        self.original = original
        self.values = values
        self.error = ""

    def set(self, name: str, value: object, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("Control enabled state must be a boolean")
        spec = HARDWARE_CONTROLS[name]
        resume_fixed = self.fixed_range and name == "detail"
        if self._fixed_range_owned and name in ("detail_enabled", "detail"):
            spec.apply(bytearray(self.original[2, 5]), value)
            self.restore_fixed_range()
        if self._fixed_range_owned and spec.selector == 2:
            raise CameraError("Turn off fixed range before changing camera display controls")
        self.load()
        proposed = self.enabled | {name} if enabled else self.enabled - {name}
        values = {**self.values, name: value}
        key = spec.selector, spec.command
        previous = self.read(*key)
        target = bytearray(previous)
        # Restore only owned fields; retain all unrelated configuration bytes.
        for control in HARDWARE_CONTROLS.values():
            if (control.selector, control.command) == key:
                size = struct.calcsize("<" + control.fmt)
                start = control.offset
                target[start : start + size] = self.original[key][start : start + size]
                for offset, _ in control.extra:
                    target[offset] = self.original[key][offset]
        for other_name, control in HARDWARE_CONTROLS.items():
            if other_name in proposed and (control.selector, control.command) == key:
                control.apply(target, values[other_name])
        # Also validate disabled row values before retaining them for re-enabling.
        spec.apply(bytearray(self.original[key]), value)
        try:
            self.write(key, bytes(target))
            if spec.selector == 2 and self._processing_preset_owned:
                self._apply_processing_preset(self.processing_preset)
        except CameraError:
            self.write(key, previous)
            if spec.selector == 2 and self._processing_preset_owned:
                self.restore_processing_preset()
            raise
        self.enabled = proposed
        self.values = values
        if resume_fixed:
            self.set_fixed_range(True)
        if spec.selector == 2 and self._tone_owned:
            self.set_tone(self.gamma, self.boost)

    def restore(self) -> None:
        self.restore_tone()
        self.restore_fixed_range()
        self.restore_processing_preset()
        for name in tuple(self.enabled):
            self.set(name, self.values[name], False)

    def state(self) -> dict:
        return {
            name: {
                "value": (
                    self.values[name]
                    if name in self.enabled
                    else spec.decode(self.original[spec.selector, spec.command])
                    if self.original
                    else spec.minimum
                ),
                "enabled": name in self.enabled,
                "available": bool(self.original),
            }
            for name, spec in HARDWARE_CONTROLS.items()
        }

    @property
    def measurement_active(self) -> bool:
        return any(
            HARDWARE_CONTROLS[name].selector == 3
            for name in self.enabled
            if name != "center_overlay"
        )

    @property
    def preview_active(self) -> bool:
        return self._tone_owned or self._processing_preset_owned or any(
            HARDWARE_CONTROLS[name].selector == 2 for name in self.enabled
        )
