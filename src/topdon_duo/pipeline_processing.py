"""Ordered display processing; radiometry never enters a display operation."""

from __future__ import annotations

from copy import deepcopy
from threading import Condition, Thread
from time import perf_counter

import cv2
import numpy as np

from .camera import IMAGE_OFFSET, decode_duo_frame, raw_temperatures
from .pipeline import active_nodes, thermal_source
from .upsampling import VisionUpsampler

MAX_PIXELS = 4_000_000
INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def check_size(width, height):
    if width * height > MAX_PIXELS:
        raise ValueError("Pipeline exceeds the 4 megapixel image limit; reduce scales/passes")


def resize(image, size, interpolation=cv2.INTER_CUBIC):
    check_size(*size)
    return cv2.resize(image, size, interpolation=interpolation)


def bytes_image(image):
    return np.clip(image, 0, 255).round().astype(np.uint8)


def luminance(image):
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2YCrCb)[..., 0]


def map_luminance(image, operation):
    color = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2YCrCb)
    color[..., 0] = operation(color[..., 0])
    return np.clip(cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR), 0, 255)


def colorize(gray, palette):
    gray = bytes_image(gray)
    if palette.startswith("camera_"):
        value = int(palette.split("_")[1])
        aliases = {
            1: "white_hot",
            2: "black_hot",
            10: "plasma",
            11: "jet",
            12: "hot",
            13: "magma",
            14: "inferno",
            15: "hot",
            16: "turbo",
            17: "plasma",
            18: "white_hot",
            19: "jet",
            20: "hot",
            21: "summer",
            22: "ocean",
        }
        if value not in aliases:
            raise ValueError(
                "Unknown baseline camera palette; select a supported Camera colors palette"
            )
        palette = aliases[value]
    if palette in ("white_hot", "black_hot"):
        return cv2.cvtColor(255 - gray if palette == "black_hot" else gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{palette.upper()}"))


class PipelineProcessor:
    def __init__(self):
        self.models = {}

    def process(self, frame, averaged, document, scale=3, rotation=0, camera_palette=1):
        _, raw, preview = decode_duo_frame(frame)
        software = active_nodes(document, "software")
        ranges = [item for item in software if item["type"] == "range"]
        thermal = thermal_source(document) or not np.any(preview)
        mapping = None
        if thermal:
            plane = raw_temperatures(averaged if averaged is not None else raw, offset=50)
            if ranges:
                mapping = ranges[0]["params"]["low"], ranges[0]["params"]["high"]
            else:
                mapping = tuple(np.percentile(plane, (1, 99)))
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
                if document["hardware"][0]["params"]["source"] == "preview"
                else "white_hot"
            )
            image = colorize(gray, palette).astype(np.float32)
            # Avoid quantizing grayscale thermal data until a color/model boundary.
            if palette in ("white_hot", "camera_1"):
                image = np.repeat(gray[..., None], 3, axis=2)
        else:
            yuyv = np.frombuffer(frame, np.uint8, offset=IMAGE_OFFSET * 2).reshape(
                *preview.shape, 2
            )
            image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2).astype(np.float32)
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
        first_range = True
        for item in software:
            kind, p = item["type"], item["params"]
            if kind == "brightness":
                image = map_luminance(image, lambda y, p=p: y + p["amount"] * 2.55)
            elif kind == "contrast":
                image = map_luminance(image, lambda y, p=p: (y - 127.5) * p["amount"] + 127.5)
            elif kind == "gamma":
                image = map_luminance(
                    image, lambda y, p=p: 255 * (np.clip(y, 0, 255) / 255) ** (1 / p["amount"])
                )
            elif kind == "colors":
                image = colorize(luminance(image), p["palette"]).astype(np.float32)
            elif kind == "filter":
                mode = p["filter"]
                if mode == "bilateral":
                    image = cv2.bilateralFilter(image.astype(np.float32), 5, 30 * p["amount"], 3)
                elif mode == "median":
                    image = cv2.medianBlur(bytes_image(image), 3).astype(np.float32)
                elif mode in ("gaussian", "sharpen") and p["amount"]:
                    blur = cv2.GaussianBlur(
                        image, (0, 0), 0.8 if mode == "sharpen" else max(0.1, p["amount"])
                    )
                    image = (
                        np.clip(image * (1 + p["amount"]) - blur * p["amount"], 0, 255)
                        if mode == "sharpen"
                        else blur
                    )
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
                    low, high = mapping
                    image = map_luminance(
                        image,
                        lambda y, low=low, high=high, p=p: (
                            ((y / 255 * (high - low) + low) - p["low"])
                            * 255
                            / (p["high"] - p["low"])
                        ),
                    )
                    mapping = p["low"], p["high"]
            elif kind == "interpolation":
                image = resize(
                    image,
                    (image.shape[1] * p["scale"], image.shape[0] * p["scale"]),
                    INTERPOLATIONS[p["method"]],
                )
            elif kind == "enhance" and p["model"] != "off" and p["amount"]:
                if p["input"] != "current":
                    image = resize(image, (256, 192) if p["input"] == "native" else (512, 384))
                model = (
                    "anime4k09" if p["model"] == "anime4k09" else f"acnet-legacy-hdn{p['denoise']}"
                )
                upsampler = self.models.setdefault((item["id"], model), VisionUpsampler())
                # Anime passes refine one 2× output; ACNet passes repeatedly upscale.
                repetitions = 1 if model == "anime4k09" else int(p["passes"])
                for _ in range(repetitions):
                    check_size(image.shape[1] * 2, image.shape[0] * 2)
                    image = upsampler.apply(
                        bytes_image(image), model, p["amount"], int(p["passes"])
                    )
                    if upsampler.error:
                        raise ValueError(upsampler.error)
                image = image.astype(np.float32)
        live = {
            (
                item["id"],
                "anime4k09"
                if item["params"]["model"] == "anime4k09"
                else f"acnet-legacy-hdn{item['params']['denoise']}",
            )
            for item in software
            if item["type"] == "enhance"
        }
        self.models = {key: value for key, value in self.models.items() if key in live}
        if rotation:
            image = np.rot90(image, -(rotation // 90)).copy()
        target = (raw.shape[1] * scale, raw.shape[0] * scale)
        if rotation in (90, 270):
            target = target[::-1]
        final_interpolation = (
            cv2.INTER_CUBIC
            if any(item["type"] == "antialiasing" for item in software)
            else cv2.INTER_NEAREST
        )
        return bytes_image(
            resize(image, target, final_interpolation)
        ), "raw" if thermal else "preview"


class PipelineWorker:
    """Single image worker with an overwritten pending slot and revision guard."""

    def __init__(self):
        self.condition = Condition()
        self.pending = None
        self.result = None
        self.closed = False
        self.processor = PipelineProcessor()
        self.thread = Thread(target=self._run, name="image-pipeline", daemon=True)
        self.thread.start()

    def submit(self, frame, averaged, document, revision, scale, rotation, camera_palette):
        with self.condition:
            self.pending = (
                frame,
                None if averaged is None else averaged.copy(),
                deepcopy(document),
                revision,
                scale,
                rotation,
                camera_palette,
            )
            self.condition.notify()

    def latest(self, revision):
        with self.condition:
            return self.result if self.result is not None and self.result[0] == revision else None

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    return
                frame, averaged, document, revision, scale, rotation, palette = self.pending
                self.pending = None
            started = perf_counter()
            try:
                image, source = self.processor.process(
                    frame, averaged, document, scale, rotation, palette
                )
                result = revision, image, source, (perf_counter() - started) * 1000, ""
            except (ValueError, TypeError, KeyError, cv2.error, MemoryError) as exc:
                result = (
                    revision,
                    None,
                    "raw" if thermal_source(document) else "preview",
                    (perf_counter() - started) * 1000,
                    str(exc),
                )
            with self.condition:
                self.result = result

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify()
        self.thread.join(timeout=1)
