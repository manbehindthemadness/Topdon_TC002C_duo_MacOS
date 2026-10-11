"""
Check installed external ONNX weights separately from deterministic application tests.
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort

from topdon_duo import onnx_models
from topdon_duo.coreml_model import coreml_input_shape_overrides
from topdon_duo.onnx_upsampling import ONNXUpsampler

V1_MODELS = ("mewzoom-v1-2x", "mewzoom-v1-4x")


def check_output(model: str, backend: str) -> None:
    """
    Verify installed weights before running a bounded local inference smoke check.
    """
    onnx_models.verified_model(model)
    engine = ONNXUpsampler()
    image = np.full((24, 32, 3), 100, np.uint8)
    try:
        output = engine.apply(image, model, backend)
        factor = onnx_models.MODELS[model]["factor"]
        if output.shape != (24 * factor, 32 * factor, 3):
            raise ValueError(f"Unexpected {model} output shape: {output.shape}")
    finally:
        engine.close()


def check_v1_overrides(model: str) -> None:
    """
    Compare dynamic and fixed shape sessions using independently generated input values.
    """
    data = onnx_models.verified_model(model)
    shape = (1, 3, 24, 32)
    overrides = coreml_input_shape_overrides(data, shape)
    if set(overrides.values()) != {1, 24, 32}:
        raise ValueError("Unexpected V1 shape overrides")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    dynamic = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    for name, value in overrides.items():
        options.add_free_dimension_override_by_name(name, value)
    fixed = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    blob = np.random.default_rng(40).random(shape, dtype=np.float32)
    if not np.allclose(dynamic.run(None, {"x": blob})[0],
                       fixed.run(None, {"x": blob})[0], atol=1e-6):
        raise ValueError("Fixed V1 dimensions changed CPU outputs")


def main() -> int:
    """
    Report missing local weights and failures; never install or convert models.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(onnx_models.MODELS))
    parser.add_argument("--coreml", action="store_true", help="Also check V1 Core ML on macOS")
    args = parser.parse_args()
    if args.coreml and sys.platform != "darwin":
        parser.error("--coreml requires macOS")
    failed = False
    for model in args.models or tuple(onnx_models.MODELS):
        if not onnx_models.model_path(model).exists():
            print(f"SKIP {model}: optional weights not installed")
            continue
        try:
            check_output(model, "cpu")
            if model in V1_MODELS:
                check_v1_overrides(model)
                if args.coreml:
                    if "CoreMLExecutionProvider" not in ort.get_available_providers():
                        raise ValueError("Core ML execution provider is unavailable")
                    check_output(model, "coreml")
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"FAIL {model}: {exc}", file=sys.stderr)
            failed = True
        else:
            print(f"PASS {model}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
