"""
Portable Custom device preferences and worker-side inference provider choices.
"""

from typing import Any

from .. import apple_acceleration, nvidia_acceleration
from ..acceleration import select_backend

BACKENDS = (("cpu", "CPU"), ("coreml", "Apple Core ML"), ("cuda", "NVIDIA CUDA"))
APPLE_COMPUTE = (
    ("CPUOnly", "CPU only"),
    ("CPUAndGPU", "CPU + GPU"),
    ("ALL", "CPU + GPU + Neural Engine"),
    ("CPUAndNeuralEngine", "CPU + Neural Engine"),
)


def validate_device(value: object) -> dict[str, str]:
    """
    Validate saved device settings without probing hardware or importing package code.
    """
    if not isinstance(value, dict) or set(value) != {"backend", "apple_compute"}:
        raise ValueError("Custom device requires backend and apple_compute")
    backend, compute = value["backend"], value["apple_compute"]
    if not isinstance(backend, str) or backend not in dict(BACKENDS):
        raise ValueError("Custom device backend must be cpu, coreml or cuda")
    if not isinstance(compute, str) or compute not in dict(APPLE_COMPUTE):
        raise ValueError("Invalid Custom Apple compute devices")
    settings = {"backend": backend, "apple_compute": compute}
    return settings


def resolve_device(
    value: object, *, apple_available: bool | None = None, nvidia_available: bool | None = None,
) -> dict[str, str]:
    """
    Resolve a GPU preference through the viewer's capability probes, retaining CPU intent.

    Explicit capability flags reuse a branch's startup results. Standalone callers
    probe only for GPU requests; saved settings are never mutated.
    """
    settings = validate_device(value)
    requested = settings["backend"]
    if requested == "cpu":
        backend = "cpu"
    else:
        apple = (
            apple_acceleration.apple_acceleration()["available"]
            if apple_available is None else apple_available
        )
        nvidia = (
            nvidia_acceleration.nvidia_acceleration()["available"]
            if nvidia_available is None else nvidia_available
        )
        backend = select_backend(requested, apple, nvidia)
    effective = {
        "backend": backend,
        "apple_compute": settings["apple_compute"] if backend == "coreml" else "CPUOnly",
    }
    return effective


def onnx_providers(device: object) -> list[Any]:
    """
    Build ONNX provider options from a callback's resolved device configuration.

    This worker-only helper uses the built-in CUDA preload/options and Apple
    compute units. The package still owns model compatibility and session caching.
    """
    settings = validate_device(device)
    if settings["backend"] == "cuda":
        providers = nvidia_acceleration.cuda_providers()
    elif settings["backend"] == "coreml":
        providers = [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": settings["apple_compute"],
                "RequireStaticInputShapes": "0",
                "EnableOnSubgraphs": "0",
            }),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return providers
