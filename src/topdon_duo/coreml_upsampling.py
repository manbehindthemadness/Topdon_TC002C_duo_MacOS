"""Optional, isolated Core ML ACNet inference; never reconfigure OpenCV's CPU DNN."""

import logging
import multiprocessing
import sys
from contextlib import suppress
from importlib.resources import files
from time import perf_counter

import cv2
import numpy as np

from .coreml_model import explicit_coreml_padding

logger = logging.getLogger(__name__)


class _CoreMLRuntime:
    """One lazily loaded session per node/configuration, with explicit failure reporting."""

    def __init__(self):
        self._session = None
        self._key = None
        self.error = ""
        self.elapsed_ms = 0.0

    def close(self):
        self._session = None

    def _load(self, denoise, compute):
        key = denoise, compute
        if self._key == key:
            if self.error:
                raise ValueError(self.error)
            return
        self._key = key
        self._session = None
        self.error = ""
        try:
            if sys.platform != "darwin":
                raise ValueError("requires macOS; bypass this node or use CPU AI enhancement")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ValueError("Apple runtime missing; run uv sync in the project directory") from exc
            if "CoreMLExecutionProvider" not in ort.get_available_providers():
                raise ValueError("installed ONNX Runtime does not provide CoreMLExecutionProvider")
            data = (
                files("topdon_duo")
                .joinpath("models", f"acnet-legacy-hdn{denoise}.onnx")
                .read_bytes()
            )
            data = explicit_coreml_padding(data)
            options = ort.SessionOptions()
            # CPU handles unsupported operators only within this isolated node.
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                data,
                sess_options=options,
                providers=[
                    (
                        "CoreMLExecutionProvider",
                        {
                            "ModelFormat": "MLProgram",
                            "MLComputeUnits": compute,
                            "RequireStaticInputShapes": "0",
                            "EnableOnSubgraphs": "0",
                        },
                    ),
                    "CPUExecutionProvider",
                ],
            )
            session.disable_fallback()
            if "CoreMLExecutionProvider" not in session.get_providers():
                raise ValueError(
                    "Core ML session initialization failed (CPU-only fallback rejected)"
                )
            self._session = session
            logger.info(
                "Apple ACNet: Core ML enabled (%s); unsupported operators may use CPU. "
                "Device selection is not proof of GPU execution.",
                compute,
            )
        except Exception as exc:
            self.error = f"Apple Core ML ACNet unavailable: {exc}"
            raise ValueError(self.error) from exc

    def apply(self, image, denoise=0, compute="CPUAndGPU", amount=1.0):
        if not amount:
            self.elapsed_ms = 0.0
            return image
        self._load(denoise, compute)
        started = perf_counter()
        try:
            color = image.ndim == 3
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) if color else None
            gray = ycrcb[..., 0] if color else image
            blob = np.ascontiguousarray(gray, dtype=np.float32)[None, None] / 255.0
            name = self._session.get_inputs()[0].name
            tensor = self._session.run(None, {name: blob})[0]
            expected = (1, 1, gray.shape[0] * 2, gray.shape[1] * 2)
            if tensor.shape != expected or not np.isfinite(tensor).all():
                raise ValueError("Unexpected model output dimensions or non-finite values")
            enhanced = np.clip(tensor[0, 0] * 255, 0, 255).round().astype(np.uint8)
            size = enhanced.shape[1], enhanced.shape[0]
            if amount != 1:
                baseline = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)
                enhanced = cv2.addWeighted(enhanced, amount, baseline, 1 - amount, 0)
            if color:
                ycrcb = cv2.resize(ycrcb, size, interpolation=cv2.INTER_CUBIC)
                ycrcb[..., 0] = enhanced
                enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            self.elapsed_ms = (perf_counter() - started) * 1000
            return enhanced
        except Exception as exc:
            self.error = f"Apple Core ML ACNet failed: {exc}"
            self.elapsed_ms = 0.0
            raise ValueError(self.error) from exc


def _coreml_worker(connection, denoise, compute):
    """Spawn, never fork: do not inherit GUI, USB queues or native thread state."""
    runtime = _CoreMLRuntime()
    try:
        while True:
            image, amount = connection.recv()
            try:
                result = runtime.apply(image, denoise, compute, amount)
                connection.send((result, runtime.elapsed_ms, ""))
            except ValueError as exc:
                connection.send((None, 0.0, str(exc)))
                return
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class CoreMLUpsampler:
    """Contain native crashes in one persistent helper process per Apple node."""

    def __init__(self):
        self._process = self._connection = self._key = None
        self.error = ""
        self.elapsed_ms = 0.0

    def close(self):
        connection, process = self._connection, self._process
        self._connection = self._process = None
        if connection is not None:
            connection.close()
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            process.close()

    def __del__(self):
        with suppress(OSError, ValueError, AssertionError, AttributeError):
            self.close()

    def _start(self, denoise, compute):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_coreml_worker, args=(child, denoise, compute), daemon=True
        )
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()
        self._connection, self._process = parent, process

    def apply(self, image, denoise=0, compute="CPUAndGPU", amount=1.0):
        if not amount:
            self.elapsed_ms = 0.0
            return image
        key = denoise, compute
        if key != self._key:
            self.close()
            self._key = key
            self.error = ""
        if self.error:
            raise ValueError(self.error)
        try:
            if sys.platform != "darwin":
                raise ValueError("requires macOS; bypass this node or use CPU AI enhancement")
            if self._process is None:
                self._start(denoise, compute)
            self._connection.send((image, amount))
            # Includes first-use compilation; UI/USB remain on their existing threads.
            if not self._connection.poll(45):
                raise ValueError("helper timed out after 45 seconds; bypass this node")
            output, self.elapsed_ms, error = self._connection.recv()
            if error:
                raise ValueError(error)
            return output
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            process = self._process
            if process is not None:
                process.join(timeout=0.2)
            code = process.exitcode if process is not None else None
            self.error = (
                f"Apple Core ML helper exited unexpectedly (exit {code}); "
                "bypass this node to continue using the CPU pipeline"
            )
            self.close()
            self.elapsed_ms = 0.0
            raise ValueError(self.error) from exc
        except Exception as exc:
            message = str(exc)
            self.error = (
                message
                if message.startswith("Apple Core ML")
                else f"Apple Core ML ACNet unavailable: {message}"
            )
            self.close()
            self.elapsed_ms = 0.0
            raise ValueError(self.error) from exc
