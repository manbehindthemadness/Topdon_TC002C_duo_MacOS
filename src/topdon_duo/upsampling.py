"""Portable Anime4K09 and ACNet display enhancement."""

from importlib.resources import files
from time import perf_counter

import cv2
import numpy as np

from .anime4k09 import upscale_anime4k09


class VisionUpsampler:
    """Keep one model loaded; never modify the caller's measurement/image arrays."""

    def __init__(self) -> None:
        self._model = None
        self._net = None
        self.error = ""
        self.elapsed_ms = 0.0

    def _load(self, model: str) -> None:
        if model == self._model:
            if self.error:
                raise ValueError(self.error)
            return
        self._model = model
        self._net = None
        self.error = ""
        try:
            data = files("topdon_duo").joinpath("models", f"{model}.onnx").read_bytes()
            net = cv2.dnn.readNetFromONNX(np.frombuffer(data, dtype=np.uint8))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            # CPU is the default. OpenCV 5's graph engine warns on target setters.
            self._net = net
        except (OSError, cv2.error) as exc:
            self.error = f"ACNet unavailable: {exc}"
            raise ValueError(self.error) from exc

    def reset(self) -> None:
        self._model = self._net = None
        self.error = ""
        self.elapsed_ms = 0.0

    def apply(
        self, image: np.ndarray, model: str, amount: float = 1.0, passes: int = 3
    ) -> np.ndarray:
        if amount == 0:
            self.elapsed_ms = 0.0
            return image
        if model == "anime4k09":
            started = perf_counter()
            result = upscale_anime4k09(image, passes=passes, strength=0.5 * amount)
            self.elapsed_ms = (perf_counter() - started) * 1000
            return result
        if self.error and self._model == model:
            return image
        try:
            self._load(model)
            started = perf_counter()
            # Thermal intensity is enhanced before the selected display palette.
            # A hardware color preview uses enhanced luminance with resized chroma.
            color = image.ndim == 3
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) if color else None
            gray = ycrcb[..., 0] if color else image
            blob = np.ascontiguousarray(gray, dtype=np.float32)[None, None] / 255.0
            self._net.setInput(blob)
            output = self._net.forward()[0, 0]
            if output.shape != (gray.shape[0] * 2, gray.shape[1] * 2):
                raise ValueError("Unexpected ACNet output dimensions")
            enhanced = np.clip(output * 255, 0, 255).round().astype(np.uint8)
            if amount != 1:
                baseline = cv2.resize(
                    gray, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_CUBIC
                )
                enhanced = cv2.addWeighted(enhanced, amount, baseline, 1 - amount, 0)
            if color:
                ycrcb = cv2.resize(
                    ycrcb, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_CUBIC
                )
                ycrcb[..., 0] = enhanced
                enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            self.elapsed_ms = (perf_counter() - started) * 1000
            return enhanced
        except (ValueError, cv2.error) as exc:
            # Keep the live viewer usable and surface the failure in Camera status.
            self.error = self.error or f"ACNet unavailable: {exc}"
            self.elapsed_ms = 0.0
            return image
