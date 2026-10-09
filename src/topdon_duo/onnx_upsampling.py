"""Isolated experimental ONNX display enhancement, never radiometric inference."""

import logging
import multiprocessing
import sys
from contextlib import suppress
from importlib.resources import files
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .coreml_model import coreml_input_shape_overrides, explicit_coreml_padding
from .enhancement_limits import MAX_PIXELS
from .model_downloads import MODEL_DOWNLOADS
from .nvidia_acceleration import cuda_providers
from .onnx_models import MODELS, verified_model

LOG = logging.getLogger(__name__)

BUNDLED_ACNET = tuple(f"acnet-legacy-hdn{level}" for level in range(4))


def model_spec(model: str) -> dict[str, Any]:
    """
    Describe bundled ACNet without adding legacy weights to the download catalog.
    """
    return {"factor": 2, "rgb": False} if model in BUNDLED_ACNET else MODELS[model]


def model_data(model: str) -> bytes:
    """
    Load original bundled weights or verify a downloaded model's checksum.
    """
    if model in BUNDLED_ACNET:
        return files("topdon_duo").joinpath("models", f"{model}.onnx").read_bytes()
    return verified_model(model)


class ONNXRuntime:
    def __init__(self):
        self.session = None
        self.key = None
        self.elapsed_ms = 0.0

    def _infer(self, blob: np.ndarray, model: str, noise: float = 15) -> np.ndarray:
        """
        Run the model's tensor convention and preserve rectangular display geometry.
        """
        entry = self.session.get_inputs()[0]
        if model_spec(model).get("task") == "style":
            # Use each export's bounded working size and tensor convention.
            # Fit the full rectangular view without cropping/stretching it.
            spec = MODELS[model]
            height, width = blob.shape[2:]
            target_h, target_w = MODELS[model]["fixed_input"]
            ratio = min(target_h / height, target_w / width)
            fit_h = min(target_h, max(1, round(height * ratio)))
            fit_w = min(target_w, max(1, round(width * ratio)))
            rgb = blob[0].transpose(1, 2, 0)
            fitted = cv2.resize(
                rgb, (fit_w, fit_h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
            )
            top, left = (target_h - fit_h) // 2, (target_w - fit_w) // 2
            padded = cv2.copyMakeBorder(
                fitted,
                top,
                target_h - fit_h - top,
                left,
                target_w - fit_w - left,
                cv2.BORDER_REPLICATE,
            )
            nhwc = spec.get("layout") == "nhwc"
            feed = padded[None] if nhwc else padded.transpose(2, 0, 1)[None]
            result = self.session.run(None, {entry.name: np.ascontiguousarray(feed)})[0]
            channels = spec.get("output_channels", 3)
            expected = (
                (1, target_h, target_w, channels) if nhwc else (1, channels, target_h, target_w)
            )
            if result.shape != expected or not np.isfinite(result).all():
                raise ValueError("Unexpected style output dimensions or non-finite pixels")
            if nhwc:
                result = result.transpose(0, 3, 1, 2)
            if channels == 1:
                result = np.repeat(result, 3, axis=1)
            cropped = result[0, :, top : top + fit_h, left : left + fit_w].transpose(1, 2, 0)
            restored = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
            return np.ascontiguousarray(restored.transpose(2, 0, 1)[None])
        if model == "ffdnet-gray":
            height, width = blob.shape[2:]
            padded = np.pad(blob, ((0, 0), (0, 0), (0, height % 2), (0, width % 2)), mode="edge")
            result = self.session.run(
                None,
                {
                    entry.name: np.ascontiguousarray(padded),
                    "sigma": np.full((1, 1, 1, 1), noise / 255.0, dtype=np.float32),
                },
            )[0]
            if result.shape != padded.shape:
                raise ValueError("Unexpected FFDNet output dimensions")
            return result[:, :, :height, :width]
        shape = getattr(entry, "shape", ())
        if (
            model != "espcn"
            or len(shape) != 4
            or not all(isinstance(n, int) and n > 0 for n in shape[2:])
        ):
            return self.session.run(None, {entry.name: blob})[0]
        # The published ESPCN export hard-codes 224x224 in its pixel shuffle.
        # Use overlapping tiles rather than squashing a rectangular camera feed.
        tile_h, tile_w = shape[2:]
        halo = 8  # More than the convolution stack's five-pixel receptive radius.
        step_h, step_w = tile_h - 2 * halo, tile_w - 2 * halo
        if min(step_h, step_w) <= 0:
            raise ValueError("ESPCN tile size is too small")
        height, width = blob.shape[2:]
        padded = cv2.copyMakeBorder(blob[0, 0], halo, tile_h, halo, tile_w, cv2.BORDER_REFLECT_101)
        factor = MODELS[model]["factor"]
        output = np.empty((1, 1, height * factor, width * factor), dtype=np.float32)
        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                patch = np.ascontiguousarray(padded[y : y + tile_h, x : x + tile_w])[None, None]
                result = self.session.run(None, {entry.name: patch})[0]
                if (
                    result.shape != (1, 1, tile_h * factor, tile_w * factor)
                    or not np.isfinite(result).all()
                ):
                    raise ValueError("Unexpected ESPCN tile output or non-finite pixels")
                h, w = min(step_h, height - y) * factor, min(step_w, width - x) * factor
                start = halo * factor
                output[:, :, y * factor : y * factor + h, x * factor : x * factor + w] = result[
                    :, :, start : start + h, start : start + w
                ]
        return output

    def apply(
        self,
        image: np.ndarray,
        model: str,
        backend: str = "cpu",
        compute: str = "CPUAndGPU",
        amount: float = 1.0,
        noise: float = 15,
    ) -> np.ndarray:
        """
        Enhance display pixels using an explicitly initialized execution provider.
        """
        if not amount:
            return image
        if backend not in ("cpu", "coreml", "cuda"):
            raise ValueError(f"Unsupported ONNX backend: {backend}")
        spec = model_spec(model)
        if model == "ffdnet-gray" and (not np.isfinite(noise) or not 0 <= noise <= 75):
            raise ValueError("FFDNet noise sigma must be between 0 and 75")
        height, width = image.shape[:2]
        factor = spec["factor"]
        size = width * factor, height * factor
        if size[0] * size[1] > MAX_PIXELS:
            raise ValueError(
                "ONNX upscale exceeds 4 megapixels; choose Native input or reduce preceding scales"
            )
        specialize = backend == "coreml" and (
            model.startswith("mewzoom-v1-") or spec.get("static_coreml", False)
        )
        key = (model, backend, compute, height, width) if specialize else (model, backend, compute)
        if self.key != key:
            import onnxruntime as ort

            data = model_data(model)
            providers = ["CPUExecutionProvider"]
            if backend == "cuda":
                if (
                    sys.platform != "linux"
                    or "CUDAExecutionProvider" not in ort.get_available_providers()
                ):
                    raise ValueError("NVIDIA CUDA provider unavailable")
                providers = cuda_providers()
            if backend == "coreml":
                if (
                    sys.platform != "darwin"
                    or "CoreMLExecutionProvider" not in ort.get_available_providers()
                ):
                    raise ValueError("Apple Core ML provider unavailable")
                # Same provider padding workaround as ACNet, on an in-memory
                # Apple copy only. Cached weights and CPU execution stay original.
                data = explicit_coreml_padding(data)
                providers.insert(
                    0,
                    (
                        "CoreMLExecutionProvider",
                        {
                            "ModelFormat": "MLProgram",
                            "MLComputeUnits": compute,
                            "RequireStaticInputShapes": "1" if specialize else "0",
                            "EnableOnSubgraphs": "0",
                        },
                    ),
                )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            if specialize:
                input_height = height + height % 2 if model == "ffdnet-gray" else height
                input_width = width + width % 2 if model == "ffdnet-gray" else width
                for name, value in coreml_input_shape_overrides(
                    data,
                    (1, 3 if spec["rgb"] else 1, input_height, input_width),
                    **({"input_name": spec["input_name"]} if "input_name" in spec else {}),
                ).items():
                    options.add_free_dimension_override_by_name(name, value)
            session = ort.InferenceSession(data, sess_options=options, providers=providers)
            session.disable_fallback()
            if backend == "coreml" and "CoreMLExecutionProvider" not in session.get_providers():
                raise ValueError("Core ML session failed; select CPU execution to try this model")
            if backend == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
                raise ValueError("CUDA session failed; select CPU execution to try this model")
            inputs = session.get_inputs()
            if len(inputs) != (2 if model == "ffdnet-gray" else 1) or (
                model == "ffdnet-gray" and inputs[1].name != "sigma"
            ):
                raise ValueError("Unexpected ONNX model inputs")
            self.session = session
            self.key = key
            LOG.info(
                "Visual ONNX %s: %s (unsupported accelerator operators may run on CPU)",
                model,
                backend,
            )
        started = perf_counter()
        color = image.ndim == 3
        if spec["rgb"]:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if color else cv2.COLOR_GRAY2RGB)
            blob = rgb.transpose(2, 0, 1)[None]
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) if color else None
            gray = ycrcb[..., 0] if color else image
            blob = gray[None, None]
        pixel_range = spec.get("pixel_range", 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) * (pixel_range / 255.0)
        offset = spec.get("input_offset", 0)
        if offset:
            blob += offset
        tensor = self._infer(blob, model, noise)
        expected = (1, 3 if spec["rgb"] else 1, size[1], size[0])
        if tensor.shape != expected or not np.isfinite(tensor).all():
            raise ValueError("Unexpected ONNX output dimensions or non-finite pixels")
        pixels = (
            np.clip((tensor[0] - offset) * (255.0 / pixel_range), 0, 255).round().astype(np.uint8)
        )
        if spec["rgb"]:
            result = cv2.cvtColor(pixels.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            if not color:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            result = pixels[0]
            if model in BUNDLED_ACNET and amount != 1:
                baseline = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)
                result = cv2.addWeighted(result, amount, baseline, 1 - amount, 0)
            if color:
                ycrcb = cv2.resize(ycrcb, size, interpolation=cv2.INTER_CUBIC)
                ycrcb[..., 0] = result
                result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        if amount != 1 and model not in BUNDLED_ACNET:
            baseline = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
            result = cv2.addWeighted(result, amount, baseline, 1 - amount, 0)
        self.elapsed_ms = (perf_counter() - started) * 1000
        return result


def _worker(connection, model, backend, compute):
    runtime = ONNXRuntime()
    try:
        while True:
            image, amount, noise = connection.recv()
            try:
                output = runtime.apply(image, model, backend, compute, amount, noise)
                connection.send((output, runtime.elapsed_ms, ""))
            except Exception as exc:  # noqa: BLE001 - isolate third-party native runtime failures
                connection.send((None, 0.0, str(exc)))
                return
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class ONNXUpsampler:
    """
    Persistent spawned CPU/Apple/NVIDIA helper; contain native runtime crashes.
    """

    def __init__(self):
        self.process = self.connection = self.key = None
        self.error = ""
        self.elapsed_ms = 0.0

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.5)
            self.process.close()
            self.process = None

    def __del__(self):
        with suppress(OSError, ValueError, AssertionError, AttributeError):
            self.close()

    def apply(
        self,
        image: np.ndarray,
        model: str,
        backend: str = "cpu",
        compute: str = "CPUAndGPU",
        amount: float = 1.0,
        noise: float = 15,
    ) -> np.ndarray:
        """
        Reuse an isolated worker until its model or execution settings change.
        """
        if not amount:
            return image
        key = model, backend, compute
        retry_download = self.key != key
        if self.key != key:
            self.close()
            self.key, self.error = key, ""
        if self.error:
            raise ValueError(self.error)
        factor = model_spec(model)["factor"]
        if image.shape[0] * image.shape[1] * factor * factor > MAX_PIXELS:
            raise ValueError("ONNX upscale exceeds 4 megapixels; reduce preceding scales")
        if model not in BUNDLED_ACNET and not MODEL_DOWNLOADS.request(model, retry=retry_download):
            # Keep the feed and downstream geometry live while installing.
            self.elapsed_ms = 0.0
            return (
                image
                if factor == 1
                else cv2.resize(
                    image,
                    (image.shape[1] * factor, image.shape[0] * factor),
                    interpolation=cv2.INTER_CUBIC,
                )
            )
        try:
            if self.process is None:
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, *key), daemon=True)
                try:
                    process.start()
                except Exception:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.connection, self.process = parent, process
            self.connection.send((image, amount, noise))
            if not self.connection.poll(90):
                raise ValueError("ONNX helper timed out after 90 seconds; bypass this node")
            output, self.elapsed_ms, error = self.connection.recv()
            if error:
                raise ValueError(error)
            return output
        except Exception as exc:
            message = str(exc) or "native helper exited unexpectedly"
            self.error = f"Visual ONNX {model} failed: {message}"
            self.elapsed_ms = 0.0
            self.close()
            raise ValueError(self.error) from exc
