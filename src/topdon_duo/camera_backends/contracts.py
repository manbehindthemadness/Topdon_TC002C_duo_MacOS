"""
Camera-independent descriptions; protocol packets belong to backend implementations.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

ControlValue = bool | float | int | str


@dataclass(frozen=True)
class ControlSpec:
    """
    Describe a verified setting independently of its transport and wire encoding.
    """

    title: str
    kind: Literal["number", "boolean", "choice"] = "number"
    minimum: float = 0
    maximum: float = 100
    step: float = 1
    unit: str = ""
    options: tuple[tuple[ControlValue, str], ...] = ()
    effect: Literal["preview", "measurement"] = "preview"
    scope: Literal["device", "spot"] = "device"
    default: ControlValue = 0
    supported: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        """
        Require bounded, finite presentation metadata before it reaches the UI.
        """
        if (not self.title or self.kind not in ("number", "boolean", "choice")
                or self.scope not in ("device", "spot") or self.effect not in ("preview", "measurement")
                or not all(math.isfinite(number) for number in (self.minimum, self.maximum, self.step))
                or self.minimum > self.maximum or self.step <= 0
                or self.kind == "choice" and not self.options):
            raise ValueError("Invalid camera control description")

    def validate(self, value: ControlValue) -> None:
        """
        Check a host value before a backend serializes any command.
        """
        if self.kind == "boolean":
            valid = type(value) is bool
        elif self.kind == "choice":
            valid = any(type(value) is type(option) and value == option
                        for option, _ in self.options)
        else:
            valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(value) and self.minimum <= value <= self.maximum
                     and math.isclose((value - self.minimum) / self.step,
                                      round((value - self.minimum) / self.step), abs_tol=1e-6))
        if not valid:
            raise ValueError(f"Invalid {self.title}")

    def as_dict(self) -> dict[str, Any]:
        """
        Serialize metadata for the separate Qt process, without backend code.
        """
        return asdict(self)


@dataclass(frozen=True)
class CameraProfile:
    """
    Identify a compatible geometry/protocol/optics profile, independently of USB handles.
    """

    id: str
    title: str
    native_size: tuple[int, int]
    frame_rate: float
    radiometry: bool = True
    preview: bool = True
    calibration_key: str = ""
    firmware: str = ""
    device_id: str = ""

    def __post_init__(self) -> None:
        """
        Reject unusable geometry and rates before constructing a viewer.
        """
        if (not isinstance(self.id, str) or not self.id or len(self.native_size) != 2
                or any(type(size) is not int or size <= 0 for size in self.native_size)
                or math.prod(self.native_size) > 4_000_000
                or not math.isfinite(self.frame_rate) or self.frame_rate <= 0):
            raise ValueError("Invalid camera profile")

    @property
    def settings_key(self) -> str:
        """
        Scope settings to an explicitly compatible geometry and calibration identity.
        """
        width, height = self.native_size
        return f"{self.id}:{width}x{height}:{self.calibration_key}:{self.device_id}"

    def as_dict(self) -> dict[str, Any]:
        """
        Describe the device to UI and capture consumers.
        """
        return asdict(self)


class CameraBackend(Protocol):
    """
    Own transport and return decoded frames; only the legacy Duo accepts raw byte frames.
    """

    profile: CameraProfile
    timeout_ms: int

    def open(self) -> object:
        """
        Open the selected device using its negotiated transport.
        """
        ...

    def frames(self) -> Iterator[Any]:
        """
        Yield complete frames, preserving source bytes as frame evidence.
        """
        ...

    def stop_stream(self) -> None:
        """
        Wake acquisition without closing buffers still owned by the capture thread.
        """
        ...

    def close(self) -> None:
        """
        Drain and close device resources.
        """
        ...

    def create_controls(self) -> Any:
        """
        Create the control owner for this device and protocol.
        """
        ...
