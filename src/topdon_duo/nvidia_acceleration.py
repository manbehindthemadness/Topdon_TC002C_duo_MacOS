"""
Probe NVIDIA inference in a disposable process before exposing CUDA controls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from typing import Any


def cuda_providers() -> list[Any]:
    """
    Select GPU zero with CPU support for operators unsupported by CUDA.
    """
    return [
        (
            "CUDAExecutionProvider",
            {"device_id": "0", "cudnn_conv_algo_search": "HEURISTIC", "use_tf32": "0"},
        ),
        "CPUExecutionProvider",
    ]


def _probe() -> dict[str, Any]:
    """
    Require provider initialization and a successful bundled ACNet inference.
    """
    result = {"available": False, "cuda": False, "reason": ""}
    if sys.platform != "linux":
        return {**result, "reason": "NVIDIA acceleration requires Linux"}
    try:
        import numpy as np
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return {**result, "reason": "Install a JetPack-compatible ONNX Runtime GPU build"}
        from .onnx_upsampling import ONNXRuntime

        runtime = ONNXRuntime()
        output = runtime.apply(np.zeros((8, 8), np.uint8), "acnet-legacy-hdn0", "cuda")
        if output.shape != (16, 16):
            raise ValueError("Invalid CUDA probe output")
        return {"available": True, "cuda": True, "reason": "NVIDIA CUDA inference ready"}
    except Exception as exc:  # noqa: BLE001 - optional native runtime must fail soft at startup
        return {**result, "reason": f"NVIDIA capability check failed: {exc}"}


@lru_cache(maxsize=1)
def nvidia_acceleration() -> dict[str, Any]:
    """
    Cache a bounded startup probe and contain native crashes and malformed replies.
    """
    unavailable = {"available": False, "cuda": False}
    if sys.platform != "linux":
        return {**unavailable, "reason": "NVIDIA acceleration requires Linux"}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "topdon_duo.nvidia_acceleration"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode:
            return {
                **unavailable,
                "reason": f"NVIDIA capability probe exited {completed.returncode}",
            }
        result = json.loads(completed.stdout)
        if (
            not isinstance(result, dict)
            or any(type(result.get(key)) is not bool for key in unavailable)
            or not isinstance(result.get("reason"), str)
            or result["available"] != result["cuda"]
        ):
            raise ValueError("invalid capability response")
        return result
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {**unavailable, "reason": f"NVIDIA capability probe unavailable: {exc}"}


if __name__ == "__main__":
    print(json.dumps(_probe()))
