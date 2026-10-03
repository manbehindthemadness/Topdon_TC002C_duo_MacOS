"""Thermal frame rendering."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import decode_duo_frame, raw_temperatures


@dataclass(frozen=True)
class TemperatureStats:
    minimum: float
    average: float
    maximum: float
    center: float

    def as_dict(self) -> dict[str, float]:
        return {
            "minimum": round(self.minimum, 2),
            "average": round(self.average, 2),
            "maximum": round(self.maximum, 2),
            "center": round(self.center, 2),
        }


@dataclass(frozen=True)
class RenderedThermalFrame:
    image: np.ndarray
    stats: TemperatureStats
    temperatures_celsius: np.ndarray
    raw_counts: np.ndarray


class ThermalRenderer:
    def __init__(
        self,
        scale: int = 3,
        smoothing: float = 0.25,
        ambient_celsius: float = 22.0,
        rotation: int = 0,
        temperature_unit: str = "C",
    ) -> None:
        if rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180, or 270")
        if temperature_unit not in ("C", "F"):
            raise ValueError("temperature_unit must be C or F")
        self.scale = scale
        self.smoothing = smoothing
        self.ambient_celsius = ambient_celsius
        self.rotation = rotation
        self.temperature_unit = temperature_unit
        self._average_raw: np.ndarray | None = None

    def display_temperature(self, celsius: float) -> float:
        if self.temperature_unit == "F":
            return celsius * 9.0 / 5.0 + 32.0
        return celsius

    def toggle_temperature_unit(self) -> str:
        self.temperature_unit = "F" if self.temperature_unit == "C" else "C"
        return self.temperature_unit

    def rotate_clockwise(self) -> int:
        self.rotation = (self.rotation + 90) % 360
        return self.rotation

    def render(self, frame: bytes) -> tuple[np.ndarray, TemperatureStats]:
        rendered = self.render_detailed(frame)
        return rendered.image, rendered.stats

    def _orient(self, array: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            return cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(array, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return array.copy()

    def render_detailed(self, frame: bytes) -> RenderedThermalFrame:
        _telemetry, raw, _preview = decode_duo_frame(frame)

        if self._average_raw is None:
            self._average_raw = raw.astype(np.float32)
        else:
            cv2.accumulateWeighted(raw, self._average_raw, self.smoothing)

        averaged = self._average_raw
        celsius = self._orient(
            raw_temperatures(averaged, ambient_celsius=self.ambient_celsius)
        )
        oriented_raw = self._orient(raw)
        oriented_average = self._orient(averaged)
        center_y, center_x = celsius.shape[0] // 2, celsius.shape[1] // 2
        stats = TemperatureStats(
            minimum=float(celsius.min()),
            average=float(celsius.mean()),
            maximum=float(celsius.max()),
            center=float(celsius[center_y, center_x]),
        )

        low, high = np.percentile(oriented_average, (1.0, 99.0))
        if high <= low:
            normalized = np.zeros_like(oriented_average, dtype=np.uint8)
        else:
            normalized = np.clip(
                (oriented_average - low) * (255.0 / (high - low)), 0, 255
            ).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        heatmap = cv2.resize(
            heatmap,
            (heatmap.shape[1] * self.scale, heatmap.shape[0] * self.scale),
            interpolation=cv2.INTER_CUBIC,
        )

        center = (heatmap.shape[1] // 2, heatmap.shape[0] // 2)
        cv2.drawMarker(
            heatmap,
            center,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        unit = self.temperature_unit
        label = (
            f"Min {self.display_temperature(stats.minimum):.1f} {unit}   "
            f"Avg {self.display_temperature(stats.average):.1f} {unit}   "
            f"Max {self.display_temperature(stats.maximum):.1f} {unit}   "
            f"Center {self.display_temperature(stats.center):.1f} {unit}   "
            f"Ambient {self.display_temperature(self.ambient_celsius):.1f} {unit}"
        )
        cv2.rectangle(heatmap, (0, 0), (heatmap.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(
            heatmap,
            label,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return RenderedThermalFrame(
            image=heatmap,
            stats=stats,
            temperatures_celsius=celsius,
            raw_counts=oriented_raw,
        )


def decode_temperatures(frame: bytes) -> np.ndarray:
    """Public helper for consumers that need the radiometric values."""
    _, raw, _ = decode_duo_frame(frame)
    return raw_temperatures(raw)
