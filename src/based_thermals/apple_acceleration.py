"""Small, isolated startup capability probe; never import Apple libraries on Linux."""

import ctypes
import importlib
import json
import subprocess
import sys
from functools import lru_cache
from typing import Any


def _metal_available() -> bool:
    """
    Ask Metal for a device inside the disposable capability probe.
    """
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    create = metal.MTLCreateSystemDefaultDevice
    create.argtypes = []
    create.restype = ctypes.c_void_p
    # This function runs only in the short-lived probe process. Let process exit
    # release native resources rather than bridge Objective-C ownership manually.
    return bool(create())


def _probe() -> dict[str, Any]:
    """
    Discover Apple providers without importing their libraries on other platforms.
    """
    result = {"available": False, "metal": False, "coreml": False, "reason": ""}
    if sys.platform != "darwin":
        return {**result, "reason": "Apple acceleration requires macOS"}
    try:
        result["metal"] = _metal_available()
        if not result["metal"]:
            return {**result, "reason": "No Metal device available"}
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError:
            return {
                **result,
                "reason": "Apple runtime missing; run uv sync in the project directory",
            }
        result["coreml"] = "CoreMLExecutionProvider" in ort.get_available_providers()
        result["available"] = result["coreml"]
        result["reason"] = (
            "Metal device and Core ML runtime detected"
            if result["available"]
            else "Installed ONNX Runtime has no Core ML provider"
        )
    except (OSError, AttributeError, ValueError, RuntimeError) as exc:
        result["reason"] = f"Apple capability check failed: {exc}"
    return result


@lru_cache(maxsize=1)
def apple_acceleration() -> dict[str, Any]:
    """
    Probe once per viewer start, with native-library crashes/timeouts contained.
    """
    unavailable = {"available": False, "metal": False, "coreml": False}
    if sys.platform != "darwin":
        return {**unavailable, "reason": "Apple acceleration requires macOS"}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "based_thermals.apple_acceleration"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if completed.returncode:
            return {
                **unavailable,
                "reason": f"Apple capability probe exited {completed.returncode}",
            }
        result = json.loads(completed.stdout)
        if (
            not isinstance(result, dict)
            or any(type(result.get(key)) is not bool for key in unavailable)
            or not isinstance(result.get("reason"), str)
            or result["available"] != (result["metal"] and result["coreml"])
        ):
            raise ValueError("invalid capability response")
        return result
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {**unavailable, "reason": f"Apple capability probe unavailable: {exc}"}


if __name__ == "__main__":
    print(json.dumps(_probe()))
