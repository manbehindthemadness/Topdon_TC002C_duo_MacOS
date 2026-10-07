"""Thermal frame rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import IMAGE_OFFSET, decode_duo_frame, measurement_frame_status, raw_temperatures
from .pipeline import geometry, thermal_source, validate_pipeline
from .pipeline_processing import PipelineProcessor
from .upsampling import VisionUpsampler
from .view_settings import IMAGE_SOURCES, VIEW_DEFAULTS, validate_view_setting

READOUT_HEIGHT = 32


@dataclass(frozen=True)
class TemperatureStats:
    minimum: float
    average: float
    maximum: float
    center: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            name: round(value, 2) if np.isfinite(value) else None
            for name, value in (
                ("minimum", self.minimum),
                ("average", self.average),
                ("maximum", self.maximum),
                ("center", self.center),
            )
        }


@dataclass(frozen=True)
class RenderedThermalFrame:
    image: np.ndarray
    stats: TemperatureStats
    temperatures_celsius: np.ndarray
    raw_counts: np.ndarray
    image_source: str = "raw"
    display_settings: dict = field(default_factory=dict)
    measurements_valid: bool = True
    measurement_status: str = ""


def draw_temperature_readout(
    image: np.ndarray,
    stats: TemperatureStats,
    ambient_celsius: float | None,
    temperature_unit: str = "C",
) -> np.ndarray:
    """Prepend the temperature readout without covering any thermal pixels."""

    def display(celsius: float) -> float:
        return celsius * 9.0 / 5.0 + 32.0 if temperature_unit == "F" else celsius

    def reading(celsius: float) -> str:
        return f"{display(celsius):.1f}" if np.isfinite(celsius) else "--"

    ambient_label = (
        f"Ambient {display(ambient_celsius):.1f} {temperature_unit}"
        if ambient_celsius is not None
        else "Ambient unavailable"
    )
    label = (
        f"Min {reading(stats.minimum)} {temperature_unit}   "
        f"Avg {reading(stats.average)} {temperature_unit}   "
        f"Max {reading(stats.maximum)} {temperature_unit}   "
        f"Center {reading(stats.center)} {temperature_unit}   "
        f"{ambient_label}"
    )
    canvas = np.zeros((image.shape[0] + READOUT_HEIGHT, image.shape[1], 3), np.uint8)
    canvas[READOUT_HEIGHT:] = image
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
    font_scale = 0.52 * min(1.0, max(1, image.shape[1] - 20) / text_width)
    cv2.putText(
        canvas,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


class ThermalRenderer:
    def __init__(
        self,
        scale: int = 3,
        smoothing: float = 0.25,
        ambient_celsius: float | None = 22.0,
        rotation: int = 0,
        temperature_unit: str = "C",
        image_source: str = "preview",
    ) -> None:
        if rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180, or 270")
        if temperature_unit not in ("C", "F"):
            raise ValueError("temperature_unit must be C or F")
        if image_source not in IMAGE_SOURCES:
            raise ValueError("Unknown image source")
        for name, value in VIEW_DEFAULTS.items():
            setattr(self, name, value)
        self.image_source = "raw" if image_source == "analyze" else image_source
        self.analyze_mode = image_source == "analyze"
        self.scale = scale
        self.smoothing = smoothing
        self.ambient_celsius = ambient_celsius
        self.rotation = rotation
        self.temperature_unit = temperature_unit
        self._average_raw: np.ndarray | None = None
        self._last_valid_frame: bytes | None = None
        self._recover_measurements = False
        self.measurement_status = ""
        self.native_temperatures = False
        self.camera_preview = False
        self.camera_color = False
        self.hardware_settings = {}
        self.upsampler = VisionUpsampler()
        self.pipeline = None
        self.pipeline_processor = PipelineProcessor()

    def display_temperature(self, celsius: float) -> float:
        if self.temperature_unit == "F":
            return celsius * 9.0 / 5.0 + 32.0
        return celsius

    def toggle_temperature_unit(self) -> str:
        self.temperature_unit = "F" if self.temperature_unit == "C" else "C"
        return self.temperature_unit

    def toggle_image_source(self) -> str:
        self.image_source = "raw" if self.image_source == "preview" else "preview"
        return self.image_source

    def rotate_clockwise(self) -> int:
        self.rotation = (self.rotation + 90) % 360
        return self.rotation

    def render(self, frame: bytes) -> tuple[np.ndarray, TemperatureStats]:
        rendered = self.render_detailed(frame)
        image = draw_temperature_readout(
            rendered.image, rendered.stats, self.ambient_celsius, self.temperature_unit
        )
        return image, rendered.stats

    def view_settings(self) -> dict:
        settings = {name: getattr(self, name) for name in VIEW_DEFAULTS}
        if self.hardware_settings:
            settings["hardware"] = self.hardware_settings
            settings["temperature_conversion"] = (
                "camera" if self.native_temperatures else "software_ambient"
            )
        if self.pipeline is not None:
            settings["pipeline"] = self.pipeline
        return settings

    def set_pipeline(self, document):
        self.pipeline = validate_pipeline(document)
        self.image_source = self.pipeline["software"][0]["params"]["source"]
        self.mirror_horizontal, self.mirror_vertical = geometry(self.pipeline, self.rotation)
        self.analyze_mode = False

    def set_view_setting(self, name: str, value: object) -> None:
        validate_view_setting(name, value)
        if name == "image_source" and value == "analyze":
            # Keep the earlier command-line spelling working with the toggle.
            self.image_source = "raw"
            self.analyze_mode = True
            return
        if name == "raw_anime4k":
            name, value = "raw_upsampling", "anime4k09" if value else "off"
        low = value if name == "raw_temperature_low" else self.raw_temperature_low
        high = value if name == "raw_temperature_high" else self.raw_temperature_high
        if low >= high:
            raise ValueError("Raw thermal From temperature must be below To temperature")
        if name in ("upsampling", "raw_upsampling") and value != getattr(self, name):
            self.upsampler.reset()
        setattr(self, name, value)

    def _enhance_image(self, image: np.ndarray, native_size: tuple[int, int]) -> np.ndarray:
        if self.upsampling in ("off", "anime4k09") or not self.enhancement_amount:
            return image
        enhanced = self.upsampler.apply(
            self._enhancement_input(image, native_size),
            self.upsampling,
            amount=self.enhancement_amount,
        )
        return image if self.upsampler.error else enhanced

    def _enhancement_input(self, image: np.ndarray, native_size: tuple[int, int]) -> np.ndarray:
        if self.enhancement_input == "native" and image.shape[1::-1] != native_size:
            return cv2.resize(image, native_size, interpolation=cv2.INTER_AREA)
        return image

    def _orient(self, array: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            result = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            result = cv2.rotate(array, cv2.ROTATE_180)
        elif self.rotation == 270:
            result = cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            result = array.copy()
        if self.mirror_horizontal or self.mirror_vertical:
            code = (
                -1
                if self.mirror_horizontal and self.mirror_vertical
                else (1 if self.mirror_horizontal else 0)
            )
            result = cv2.flip(result, code)
        return result

    def _filter_image(self, gray: np.ndarray) -> np.ndarray:
        if self.image_filter == "bilateral":
            return cv2.bilateralFilter(gray, 5, 30, 3)
        if self.image_filter == "median":
            return cv2.medianBlur(gray, 3)
        if self.image_filter == "gaussian":
            return cv2.GaussianBlur(gray, (3, 3), 0.8)
        if self.image_filter == "sharpen":
            blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
            return cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
        return gray

    def _colorize(self, gray: np.ndarray, palette: str | None = None) -> np.ndarray:
        palette = self.color_palette if palette is None else palette
        if palette in ("white_hot", "black_hot"):
            if palette == "black_hot":
                gray = 255 - gray
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{palette.upper()}"))

    def _render_raw(self, counts: np.ndarray) -> np.ndarray:
        # Fixed hardware conversion; never anchor display colors to scene statistics.
        temperatures = raw_temperatures(counts, offset=50)
        intensity = np.clip(
            (temperatures - self.raw_temperature_low)
            * (255.0 / (self.raw_temperature_high - self.raw_temperature_low)),
            0, 255,
        )
        # Interpolate the fixed-range float plane before reducing to display bytes.
        # This path preserves fine gradients without a learned/edge-push model.
        if self.raw_upsampling == "bicubic":
            intensity = cv2.resize(intensity, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = np.clip(intensity, 0, 255).round().astype(np.uint8)
        if self.raw_sharpen_amount:
            blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
            gray = cv2.addWeighted(
                gray, 1 + self.raw_sharpen_amount, blur, -self.raw_sharpen_amount, 0
            )
        # Enhance grayscale RGB first; palettes remain a display-only final mapping.
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.raw_upsampling not in ("off", "bicubic"):
            rgb = self.upsampler.apply(rgb, self.raw_upsampling, 1.0, self.raw_anime4k_passes)
        return self._colorize(rgb[..., 0], self.raw_palette)

    def render_detailed(
        self, frame: bytes, *, update_measurements: bool = True, image_processing: bool = True
    ) -> RenderedThermalFrame:
        telemetry, raw, preview = decode_duo_frame(frame)
        self.measurement_status = measurement_frame_status(telemetry, raw)
        measurements_valid = not self.measurement_status
        if measurements_valid and update_measurements:
            if self._average_raw is None or self._recover_measurements:
                self._average_raw = raw.astype(np.float32)
            else:
                cv2.accumulateWeighted(raw, self._average_raw, self.smoothing)
            self._last_valid_frame = frame
            self._recover_measurements = False
        elif not measurements_valid:
            self._recover_measurements = True
            if self._last_valid_frame is not None:
                frame = self._last_valid_frame
                _, raw, preview = decode_duo_frame(frame)

        averaged = self._average_raw
        if averaged is None:
            # Remain responsive if the first frames arrive during calibration.
            averaged = np.zeros(raw.shape, np.float32)
            celsius = self._orient(np.full(raw.shape, np.nan, np.float32))
        else:
            celsius = self._orient(
                raw_temperatures(
                    averaged,
                    ambient_celsius=self.ambient_celsius,
                    offset=50 if self.native_temperatures else None,
                )
            )
        oriented_raw = self._orient(raw)
        oriented_average = self._orient(averaged)
        native_size = (oriented_raw.shape[1], oriented_raw.shape[0])
        center_y, center_x = celsius.shape[0] // 2, celsius.shape[1] // 2
        stats = TemperatureStats(
            minimum=float(celsius.min()),
            average=float(celsius.mean()),
            maximum=float(celsius.max()),
            center=float(celsius[center_y, center_x]),
        )

        if not image_processing or self.pipeline is not None:
            if image_processing:
                palette = self.hardware_settings.get("palette", {}).get("value", 1)
                heatmap, source = self.pipeline_processor.process(
                    frame, averaged, self.pipeline, self.scale, self.rotation, palette
                )
            else:
                heatmap = np.zeros((*celsius.shape, 3), np.uint8)
                heatmap = cv2.resize(heatmap, (celsius.shape[1] * self.scale, celsius.shape[0] * self.scale))
                source = "raw" if self.pipeline is not None and thermal_source(self.pipeline) else self.image_source
            return RenderedThermalFrame(heatmap, stats, celsius, oriented_raw, source,
                                        self.view_settings(), measurements_valid, self.measurement_status)

        # Image processing and measurement data remain independent. Some modes
        # leave the preview empty; those retain the radiometric visualization.
        use_preview = (
            not (self.camera_color and self.palette_source == "app")
            and self.image_source == "preview"
            and not self.analyze_mode
            and (self.camera_preview or bool(np.any(preview)))
        )
        image_plane = self._orient(preview) if use_preview else oriented_average
        if not self.analyze_mode:
            low, high = np.percentile(image_plane, (1.0, 99.0))
            if high <= low:
                normalized = np.zeros_like(image_plane, dtype=np.uint8)
            else:
                normalized = (
                    np.clip((image_plane.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
                    .round()
                    .astype(np.uint8)
                )
        if use_preview and self.camera_preview:
            # Keep actual camera intensities so brightness/contrast remain visible.
            if self.camera_color:
                yuyv = np.frombuffer(frame, dtype=np.uint8, offset=IMAGE_OFFSET * 2).reshape(
                    *preview.shape, 2
                )
                heatmap = self._enhance_image(
                    self._filter_image(self._orient(cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2))),
                    native_size,
                )
            else:
                heatmap = self._colorize(
                    self._enhance_image(self._filter_image(image_plane), native_size)
                )
        elif use_preview:
            heatmap = self._colorize(
                self._enhance_image(self._filter_image(normalized), native_size)
            )
        elif self.analyze_mode:
            heatmap = self._render_raw(oriented_average)
        else:
            heatmap = self._colorize(
                self._enhance_image(self._filter_image(normalized), native_size)
            )
        if not self.analyze_mode and self.upsampling == "anime4k09" and self.enhancement_amount:
            # The phone processes palette-converted display pixels, not raw temperatures.
            heatmap = self.upsampler.apply(
                self._enhancement_input(heatmap, native_size),
                self.upsampling,
                self.enhancement_amount,
                int(self.anime4k_passes),
            )
        interpolation = cv2.INTER_NEAREST
        if self.antialiasing:
            interpolation = cv2.INTER_AREA if self.scale == 1 and use_preview else cv2.INTER_CUBIC
            if heatmap.shape[1] > oriented_raw.shape[1] * self.scale:
                interpolation = cv2.INTER_AREA
        heatmap = cv2.resize(
            heatmap,
            (oriented_raw.shape[1] * self.scale, oriented_raw.shape[0] * self.scale),
            interpolation=interpolation,
        )

        return RenderedThermalFrame(
            image=heatmap,
            stats=stats,
            temperatures_celsius=celsius,
            raw_counts=oriented_raw,
            image_source="preview" if use_preview else "raw",
            display_settings=self.view_settings(),
            measurements_valid=measurements_valid,
            measurement_status=self.measurement_status,
        )


def decode_temperatures(frame: bytes) -> np.ndarray:
    """Public helper for consumers that need the radiometric values."""
    _, raw, _ = decode_duo_frame(frame)
    return raw_temperatures(raw)
