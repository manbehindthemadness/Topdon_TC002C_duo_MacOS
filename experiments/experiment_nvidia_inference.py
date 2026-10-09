"""
Verify CUDA execution and compare bundled ACNet against CPU on a live NVIDIA host.

Run with the project GPU environment; no camera, network, or elevation is needed.
Profiles default to /usr/src/codex/scratch/jetpack and stay outside pytest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort

from topdon_duo.nvidia_acceleration import cuda_providers, nvidia_acceleration
from topdon_duo.onnx_upsampling import ONNXUpsampler, model_data
from topdon_duo.upsampling import VisionUpsampler


def check_model(model: str, output: Path) -> dict[str, Any]:
    """
    Profile a native-size inference, compare CPU tensors, and exercise the worker.
    """
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output / model)
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        model_data(model), sess_options=options, providers=cuda_providers()
    )
    session.disable_fallback()
    if "CUDAExecutionProvider" not in session.get_providers():
        raise ValueError("CUDA initialization fell back to CPU")
    blob = np.random.default_rng(17).random((1, 1, 192, 256), dtype=np.float32)
    feed = {session.get_inputs()[0].name: blob}
    session.run(None, feed)
    started = perf_counter()
    gpu = session.run(None, feed)[0]
    elapsed = (perf_counter() - started) * 1000
    profile = Path(session.end_profiling())
    providers: dict[str, int] = {}
    for event in json.loads(profile.read_text()):
        provider = event.get("args", {}).get("provider")
        if provider:
            providers[provider] = providers.get(provider, 0) + 1
    if not providers.get("CUDAExecutionProvider"):
        raise ValueError("Profile contains no CUDA operator execution")
    cpu_options = ort.SessionOptions()
    cpu_options.intra_op_num_threads = 2
    cpu = ort.InferenceSession(
        model_data(model), sess_options=cpu_options, providers=["CPUExecutionProvider"]
    )
    reference = cpu.run(None, {cpu.get_inputs()[0].name: blob})[0]
    np.testing.assert_allclose(gpu, reference, atol=1e-5, rtol=1e-5)
    image = np.random.default_rng(18).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    original = image.copy()
    engine = ONNXUpsampler()
    try:
        for amount in (1.0, 0.35):
            pixels = engine.apply(image, model, "cuda", amount=amount)
            baseline = VisionUpsampler().apply(image, model, amount=amount)
            # Independent OpenCV and ORT implementations may round a pixel differently.
            np.testing.assert_allclose(pixels, baseline, atol=1, rtol=0)
        first_pass = engine.apply(image, model, "cuda")
        second_pass = engine.apply(first_pass, model, "cuda")
        if second_pass.shape != (768, 1024, 3):
            raise ValueError("Unexpected two-pass output geometry")
        np.testing.assert_array_equal(image, original)
    finally:
        engine.close()
    return {
        "model": model,
        "executed_nodes": providers,
        "gpu_warm_ms": elapsed,
        "max_absolute_error": float(np.max(np.abs(gpu - reference))),
        "profile": str(profile),
    }


def main() -> None:
    """
    Check all four bundled ACNet denoising levels on the configured NVIDIA runtime.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/usr/src/codex/scratch/jetpack"))
    args = parser.parse_args()
    capability = nvidia_acceleration()
    if not capability["available"]:
        raise SystemExit(capability["reason"])
    args.output.mkdir(parents=True, exist_ok=True)
    results = [check_model(f"acnet-legacy-hdn{level}", args.output) for level in range(4)]
    print(json.dumps({"runtime": ort.__version__, "results": results}, indent=2))


if __name__ == "__main__":
    main()
