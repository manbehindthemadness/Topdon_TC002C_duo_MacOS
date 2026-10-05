"""Verified, reversible UVC controls, used by the stream's single USB owner."""

from __future__ import annotations

import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import usb.core

from .camera import CameraError


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


class HardwareControls:
    def __init__(self, camera) -> None:
        self.camera = camera
        self.original: dict[tuple[int, int], bytes] = {}
        self.enabled: set[str] = set()
        self.values: dict[str, float] = {}
        self.error = ""
        self.snapshot_path: Path | None = None

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
        if len(response) not in (2, 4) or size != BLOCK_LENGTHS[selector, command]:
            raise CameraError("Camera returned an unsupported control layout")
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
        except CameraError:
            self.write(key, previous)
            raise
        self.enabled = proposed
        self.values = values

    def restore(self) -> None:
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
        return any(HARDWARE_CONTROLS[name].selector == 2 for name in self.enabled)
