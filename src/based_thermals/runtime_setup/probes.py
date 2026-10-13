"""
Discover driver capabilities and verify runtimes without importing native ORT in the viewer.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def environment(path: Path | None = None) -> dict[str, str]:
    """
    Pass a selected runtime to subprocesses without retaining an earlier selection.
    """
    env = dict(os.environ)
    previous = env.pop("BASED_THERMALS_ORT_PATH", None)
    entries = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != previous]
    if path is not None:
        entries.insert(0, str(path))
        env["BASED_THERMALS_ORT_PATH"] = str(path)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def query(mode: str, path: Path | None = None) -> dict[str, Any]:
    """
    Bound native probes and reject malformed replies or native process crashes.
    """
    try:
        # Native providers can write to stdout even during process teardown.
        # Keep their diagnostics separate from the structured capability reply.
        with tempfile.TemporaryDirectory(prefix="based-thermals-probe-") as directory:
            response = Path(directory) / "result.json"
            completed = subprocess.run(
                [sys.executable, "-m", "based_thermals.runtime_setup.probes", mode, str(response)],
                env=environment(path), capture_output=True, text=True, check=False, timeout=45,
            )
            if completed.returncode:
                raise ValueError(
                    f"Runtime probe exited {completed.returncode}: {completed.stderr[-2000:]}"
                )
            result = json.loads(response.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("reason"), str):
            raise TypeError("Malformed runtime probe response")
        if mode == "driver":
            if type(result.get("cuda")) is not int or result["cuda"] < 0:
                raise ValueError("Malformed CUDA driver version")
        elif type(result.get("available")) is not bool:
            raise ValueError("Malformed runtime availability")
        return result
    except (OSError, TypeError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "cuda": 0, "reason": str(exc)}


def driver() -> dict[str, Any]:
    """
    Ask the CUDA driver directly, including hosts without nvidia-smi or a CUDA toolkit.
    """
    try:
        library = ctypes.CDLL("libcuda.so.1")
        initialize = library.cuInit
        initialize.argtypes = [ctypes.c_uint]
        initialize.restype = ctypes.c_int
        version = library.cuDriverGetVersion
        version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        version.restype = ctypes.c_int
        value = ctypes.c_int()
        if initialize(0) != 0 or version(ctypes.byref(value)) != 0:
            raise ValueError("CUDA driver could not initialize a device")
        return {"cuda": value.value, "reason": "CUDA driver initialized"}
    except (OSError, AttributeError, ValueError) as exc:
        return {"cuda": 0, "reason": str(exc)}


def runtime(backend: str) -> dict[str, Any]:
    """
    Require provider initialization and finite output from a bundled model, without downloads.
    """
    try:
        import numpy as np

        if backend in {"metal", "coreml"}:
            from ..apple_acceleration import _metal_available

            if not _metal_available():
                return {"available": False, "reason": "No Metal device available"}
            if backend == "metal":
                return {"available": True, "reason": "Metal device available"}
        from ..onnx_upsampling import ONNXRuntime

        engine = ONNXRuntime()
        output = engine.apply(np.zeros((8, 8), np.uint8), "acnet-legacy-hdn0", backend)
        if output.shape != (16, 16) or not np.isfinite(output).all():
            raise ValueError("Invalid runtime probe output")
        return {"available": True, "reason": f"{backend} bundled inference verified"}
    except Exception as exc:  # noqa: BLE001 - native optional backends must fail soft
        return {"available": False, "reason": str(exc)}


def write_result(mode: str, destination: Path) -> None:
    """
    Write an isolated capability reply that native stdout/stderr cannot corrupt.
    """
    result = driver() if mode == "driver" else runtime(mode)
    destination.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    selected = sys.argv[1]
    if len(sys.argv) > 2:
        write_result(selected, Path(sys.argv[2]))
    else:
        print(json.dumps(driver() if selected == "driver" else runtime(selected)))
