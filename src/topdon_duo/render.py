"""Thermal frame rendering."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import decode_yuy2_frame, raw_temperatures


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


class ThermalRenderer:
    def __init__(self, scale: int = 3, smoothing: float = 0.25) -> None:
        self.scale = scale
        self.smoothing = smoothing
        self._average_raw: np.ndarray | None = None

    def render(self, frame: bytes) -> tuple[np.ndarray, TemperatureStats]:
        _image, radiometric = decode_yuy2_frame(frame)
        raw = radiometric[..., 0].astype(np.uint16)
        raw |= radiometric[..., 1].astype(np.uint16) << 8

        if self._average_raw is None:
            self._average_raw = raw.astype(np.float32)
        else:
            cv2.accumulateWeighted(raw, self._average_raw, self.smoothing)

        averaged = self._average_raw
        celsius = averaged / 64.0 - 273.15
        center_y, center_x = celsius.shape[0] // 2, celsius.shape[1] // 2
        stats = TemperatureStats(
            minimum=float(celsius.min()),
            average=float(celsius.mean()),
            maximum=float(celsius.max()),
            center=float(celsius[center_y, center_x]),
        )

        low, high = np.percentile(averaged, (1.0, 99.0))
        if high <= low:
            normalized = np.zeros_like(averaged, dtype=np.uint8)
        else:
            normalized = np.clip((averaged - low) * (255.0 / (high - low)), 0, 255).astype(
                np.uint8
            )
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
        label = (
            f"Min {stats.minimum:.1f} C   Avg {stats.average:.1f} C   "
            f"Max {stats.maximum:.1f} C   Center {stats.center:.1f} C"
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
        return heatmap, stats


def decode_temperatures(frame: bytes) -> np.ndarray:
    """Public helper for consumers that need the radiometric values."""
    _, radiometric = decode_yuy2_frame(frame)
    return raw_temperatures(radiometric)
