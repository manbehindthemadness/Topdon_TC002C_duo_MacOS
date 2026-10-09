"""
Execute one software branch and own its model caches.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np

from ..apple_acceleration import apple_acceleration
from ..camera import IMAGE_OFFSET, decode_duo_frame, has_yuy2_preview, raw_temperatures
from ..coreml_upsampling import CoreMLUpsampler
from ..enhancement_limits import MAX_PIXELS, acnet_pass_limit
from ..feature_processing import apply_features
from ..image_filters import apply_filter
from ..onnx_models import MODELS as ONNX_MODELS
from ..onnx_upsampling import ONNXUpsampler
from ..pipeline import (
    active_nodes,
    thermal_source,
)
from ..upsampling import VisionUpsampler
from .images import (
    INTERPOLATIONS,
    bytes_image,
    check_size,
    colorize,
    combine_images,
    encode_thumbnail,
    luminance,
    map_luminance,
    resize,
)


class BranchProcessor:
    def __init__(self, apple_available: bool | None = None) -> None:
        """
        Init.
        """
        self.apple_available = (
            apple_acceleration()["available"] if apple_available is None else apple_available
        )
        self.models: dict[tuple[str, str], VisionUpsampler] = {}
        self.coreml_models: dict[tuple[str, int, str], CoreMLUpsampler] = {}
        self.onnx_models: dict[tuple[str, str, str, str], ONNXUpsampler] = {}
        self.last_previews: dict[str, str] = {}
        self.last_preview_timings: dict[str, float] = {}
        self.collect_previews = False
        self.preview_active = False
        self.last_preview_errors = {}

    def close(self) -> None:
        """
        Close.
        """
        for engine in self.coreml_models.values():
            engine.close()
        for engine in self.onnx_models.values():
            engine.close()
        self.models = {}
        self.coreml_models = {}
        self.onnx_models = {}
        self.last_previews = {}
        self.last_preview_timings = {}

    def process(
        self,
        frame: bytes,
        averaged: np.ndarray | None,
        document: dict[str, Any],
        camera_palette: int = 1,
        inputs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, str]:
        """
        Process.
        """
        # Measurement averages remain a compatibility input, never a display source.
        del averaged
        self.last_previews = {}
        self.last_preview_timings = {}
        started = perf_counter()
        preview_overhead = 0.0
        _, raw, preview = decode_duo_frame(frame)
        software = active_nodes(document, "software")
        image, mapping, thermal = self.source_image(
            frame, raw, preview, document, software, camera_palette
        )
        direct_preview: np.ndarray | None = None
        available_inputs = inputs if inputs is not None else {}

        def resolve_input(source: str, params: dict[str, Any]) -> np.ndarray:
            """
            Resolve input.
            """
            nonlocal direct_preview
            if source == "preview":
                if direct_preview is None:
                    if not has_yuy2_preview(frame) or not np.any(preview):
                        raise ValueError("Camera preview is unavailable for the combine input/mask")
                    direct_preview = preview_image(frame, preview)
                return cast(np.ndarray, direct_preview)
            if source == "raw":
                plane = raw_temperatures(raw, offset=50)
                gray = np.clip(
                    (plane - params["raw_low"]) * 255 / (params["raw_high"] - params["raw_low"]),
                    0,
                    255,
                )
                return np.repeat(gray[..., None], 3, axis=2)
            return available_inputs[source]

        first_range = True
        for item in software:
            kind, p = item["type"], item["params"]
            if kind == "preview" and item["expanded"] and self.collect_previews:
                sampled_at = perf_counter()
                self.last_preview_timings[item["id"]] = max(
                    0, (sampled_at - started - preview_overhead) * 1000
                )
                self.last_previews[item["id"]] = encode_thumbnail(image)
                preview_overhead += perf_counter() - sampled_at
            elif kind == "combine":
                incoming = resolve_input(p["tab"], p)
                mask_source = p["mask_source"]
                mask = (
                    None
                    if mask_source == "none"
                    else incoming
                    if mask_source == "input"
                    else resolve_input(mask_source, p)
                )
                image = combine_images(image, incoming, p, mask)
            elif kind == "brightness":
                image = map_luminance(image, lambda y, params=p: y + params["amount"] * 2.55)
            elif kind == "contrast":
                image = map_luminance(
                    image, lambda y, params=p: (y - 127.5) * params["amount"] + 127.5
                )
            elif kind == "gamma":
                image = map_luminance(
                    image,
                    lambda y, params=p: 255 * (np.clip(y, 0, 255) / 255) ** (1 / params["amount"]),
                )
            elif kind == "colors":
                image = colorize(luminance(image), p["palette"]).astype(np.float32)
            elif kind in ("edges", "contours"):
                image = apply_features(image, kind, p)
            elif kind == "filter":
                image = apply_filter(image, p)
            elif kind == "antialiasing" and p["amount"]:
                image = cv2.GaussianBlur(image, (0, 0), p["amount"])
            elif kind == "mirror":
                if p["horizontal"]:
                    image = cv2.flip(image, 1)
                if p["vertical"]:
                    image = cv2.flip(image, 0)
            elif kind == "range":
                if first_range:
                    # Defines the source normalization before earlier operations run.
                    first_range = False
                else:
                    low, high = cast(tuple[float, float], mapping)
                    image = map_luminance(
                        image,
                        lambda y, lower=low, upper=high, params=p: (
                            ((y / 255 * (upper - lower) + lower) - params["low"])
                            * 255
                            / (params["high"] - params["low"])
                        ),
                    )
                    mapping = p["low"], p["high"]
            elif kind == "interpolation":
                image = resize(
                    image,
                    (image.shape[1] * p["scale"], image.shape[0] * p["scale"]),
                    INTERPOLATIONS[p["method"]],
                )
            elif (
                kind in ("onnx_superresolution", "onnx_denoise", "onnx_style")
                or kind == "enhance"
                and p["model"] in ONNX_MODELS
            ) and p["amount"]:
                image = self.apply_onnx(image, item, kind)
            elif kind == "coreml_acnet" and p["amount"]:
                image = self.apply_coreml(image, item)
            elif kind == "enhance" and p["model"] != "off" and p["amount"]:
                image = self.apply_legacy(image, item)
        self.prune_models(software)
        return image, "raw" if thermal else "preview"

    def apply_onnx(self, image: np.ndarray, item: dict[str, Any], kind: str) -> np.ndarray:
        """
        Apply an ONNX visual model with the requested backend.
        """
        p = item["params"]
        factor = ONNX_MODELS[p["model"]]["factor"]
        if kind != "onnx_denoise" and factor > 1 and p["input"] != "current":
            image = resize(image, (256, 192) if p["input"] == "native" else (512, 384))
        check_size(image.shape[1] * factor, image.shape[0] * factor)
        backend = "coreml" if p["backend"] == "coreml" and self.apple_available else "cpu"
        key = item["id"], p["model"], backend, p["apple_compute"]
        upsampler = self.onnx_models.setdefault(key, ONNXUpsampler())
        image = upsampler.apply(
            bytes_image(image),
            p["model"],
            backend,
            p["apple_compute"],
            p["amount"],
            **({"noise": p["noise"]} if p["model"] in ("dncnn-25", "ffdnet-gray") else {}),
        ).astype(np.float32)
        return image

    def apply_coreml(self, image: np.ndarray, item: dict[str, Any]) -> np.ndarray:
        """
        Apply the legacy Core ML ACNet node.
        """
        p = item["params"]
        if p["input"] != "current":
            image = resize(image, (256, 192) if p["input"] == "native" else (512, 384))
        check_size(image.shape[1] * 2, image.shape[0] * 2)
        key = item["id"], p["denoise"], p["compute"]
        upsampler = self.coreml_models.setdefault(key, CoreMLUpsampler())
        image = upsampler.apply(bytes_image(image), p["denoise"], p["compute"], p["amount"]).astype(
            np.float32
        )
        return image

    def apply_legacy(self, image: np.ndarray, item: dict[str, Any]) -> np.ndarray:
        """
        Apply Anime4K or legacy ACNet within the output size limit.
        """
        p = item["params"]
        if p["input"] != "current":
            image = resize(image, (256, 192) if p["input"] == "native" else (512, 384))
        model = "anime4k09" if p["model"] == "anime4k09" else f"acnet-legacy-hdn{p['denoise']}"
        use_apple = model != "anime4k09" and p["backend"] == "coreml" and self.apple_available
        if use_apple:
            key = item["id"], p["denoise"], p["apple_compute"]
            upsampler = self.coreml_models.setdefault(key, CoreMLUpsampler())
        else:
            upsampler = self.models.setdefault((item["id"], model), VisionUpsampler())
        # Anime passes refine one 2× output; ACNet passes repeatedly upscale.
        repetitions = 1 if model == "anime4k09" else int(p["passes"])
        if model != "anime4k09":
            maximum = acnet_pass_limit(image.shape[1], image.shape[0])
            if not maximum:
                raise ValueError(
                    "No ACNet upscale fits the 4 megapixel limit; "
                    "select Native sensor input or reduce preceding scales."
                )
            repetitions = min(repetitions, maximum)
        factor = 2**repetitions
        width, height = image.shape[1] * factor, image.shape[0] * factor
        if width * height > MAX_PIXELS:
            raise ValueError(
                f"{p['model']} {int(p['passes'])} passes would produce "
                f"{width}x{height} ({float(width * height / 1_000_000):.1f} megapixels). "
                "Limit is 4 megapixels; reduce passes or select Native sensor input."
            )
        for _ in range(repetitions):
            check_size(image.shape[1] * 2, image.shape[0] * 2)
            if use_apple:
                image = upsampler.apply(
                    bytes_image(image), p["denoise"], p["apple_compute"], p["amount"]
                )
            else:
                image = upsampler.apply(bytes_image(image), model, p["amount"], int(p["passes"]))
            if upsampler.error:
                raise ValueError(upsampler.error)
        image = image.astype(np.float32)
        return image

    def prune_models(self, software: list[dict[str, Any]]) -> None:
        """
        Close disconnected models and retain only active cache entries.
        """
        onnx_live = {
            (
                item["id"],
                item["params"]["model"],
                "coreml"
                if item["params"]["backend"] == "coreml" and self.apple_available
                else "cpu",
                item["params"]["apple_compute"],
            )
            for item in software
            if (
                item["type"] in ("onnx_superresolution", "onnx_denoise", "onnx_style")
                or item["type"] == "enhance"
                and item["params"]["model"] in ONNX_MODELS
            )
            and not item["bypass"]
            and item["params"]["amount"]
        }
        for key, engine in self.onnx_models.items():
            if key not in onnx_live:
                engine.close()
        self.onnx_models = {
            key: value for key, value in self.onnx_models.items() if key in onnx_live
        }
        live = {
            (
                item["id"],
                "anime4k09"
                if item["params"]["model"] == "anime4k09"
                else f"acnet-legacy-hdn{item['params']['denoise']}",
            )
            for item in software
            if item["type"] == "enhance" and item["params"]["model"] in ("anime4k09", "acnet")
        }
        self.models = {key: value for key, value in self.models.items() if key in live}
        coreml_live = {
            (item["id"], item["params"]["denoise"], item["params"]["compute"])
            for item in software
            if item["type"] == "coreml_acnet"
        }
        coreml_live.update(
            (item["id"], item["params"]["denoise"], item["params"]["apple_compute"])
            for item in software
            if item["type"] == "enhance"
            and item["params"]["model"] == "acnet"
            and item["params"]["backend"] == "coreml"
            and self.apple_available
        )
        for key, engine in self.coreml_models.items():
            if key not in coreml_live:
                engine.close()
        self.coreml_models = {
            key: value for key, value in self.coreml_models.items() if key in coreml_live
        }

    @staticmethod
    def source_image(
        frame: bytes,
        raw: np.ndarray,
        preview: np.ndarray,
        document: dict[str, Any],
        software: list[dict[str, Any]],
        camera_palette: int,
    ) -> tuple[np.ndarray, tuple[float, float] | None, bool]:
        """
        Select the current display plane and its thermal normalization bounds.
        """
        ranges = [item for item in software if item["type"] == "range"]
        # A complete 512x384 plane has a verified preview layout on either host.
        # Short frames retain the radiometric fallback.
        thermal = thermal_source(document) or not has_yuy2_preview(frame) or not np.any(preview)
        mapping = None
        if thermal:
            # Temporal averaging is reserved for measurements, not display pixels.
            plane = raw_temperatures(raw, offset=50)
            if ranges:
                mapping = ranges[0]["params"]["low"], ranges[0]["params"]["high"]
            else:
                percentiles = np.percentile(plane, (1, 99))
                mapping = float(percentiles[0]), float(percentiles[1])
            low, high = mapping
            gray = np.clip((plane - low) * (255 / max(high - low, 1e-6)), 0, 255)
            selected_palette = next(
                (
                    item["params"]["palette"]
                    for item in active_nodes(document, "hardware")
                    if item["type"] == "camera_colors"
                ),
                camera_palette,
            )
            palette = (
                f"camera_{int(selected_palette)}"
                if document["software"][0]["params"]["source"] == "preview"
                else "white_hot"
            )
            image = colorize(gray, palette).astype(np.float32)
            # Avoid quantizing grayscale thermal data until a color/model boundary.
            if palette in ("white_hot", "camera_1"):
                image = np.repeat(gray[..., None], 3, axis=2)
        else:
            image = preview_image(frame, preview)
            hardware = active_nodes(document, "hardware")
            # Preserve the legacy unconfigured, app-colored preview normalization.
            if any(n["type"] == "colors" for n in software) and not any(
                n["type"] not in ("source", "humidity") for n in hardware
            ):
                low, high = np.percentile(preview, (1, 99))
                gray = np.clip(
                    (preview.astype(np.float32) - low) * 255 / max(high - low, 1e-6), 0, 255
                )
                image = np.repeat(gray[..., None], 3, axis=2)
        return image, mapping, bool(thermal)


def preview_image(frame: bytes, preview: np.ndarray) -> np.ndarray:
    """
    Decode a camera display plane without affecting sensor measurements.
    """
    if has_yuy2_preview(frame):
        yuyv = np.frombuffer(frame, np.uint8, offset=IMAGE_OFFSET * 2).reshape(*preview.shape, 2)
        image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2).astype(np.float32)
    else:
        image = np.repeat(preview[..., None], 3, axis=2).astype(np.float32)
    return image
