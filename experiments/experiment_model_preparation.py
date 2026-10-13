"""
Check first-use denoiser preparation and local CPU inference without accessing a camera.

This explicitly downloads conversion dependencies and official weights when missing.
Use a separate model directory to avoid changing viewer data; ordinary pytest uses fakes.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from based_thermals.model_exports import EXPORT_MODELS, prepare_model
from based_thermals.onnx_upsampling import ONNXRuntime


def main() -> None:
    """
    Prepare requested models, verify cache reuse and run same-size CPU display inference.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=EXPORT_MODELS)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["BASED_THERMALS_MODEL_DIR"] = str(args.model_dir.resolve())
    for model in args.models:
        print(f"Preparing {model}", flush=True)
        path = prepare_model(model, lambda status: print(status, flush=True))
        assert prepare_model(model) == path
        runtime = ONNXRuntime()
        image = np.full((17, 23, 3), 100, dtype=np.uint8)
        output = runtime.apply(image, model, backend="cpu")
        assert output.shape == image.shape and np.isfinite(output).all()
        provenance = json.loads(path.with_suffix(".json").read_text())
        print(json.dumps({"model": model, "path": str(path), "shape": list(output.shape),
                          "checksum": path.with_suffix(".sha256").read_text().strip(),
                          "parity_error": provenance["max_parity_error"]}), flush=True)


if __name__ == "__main__":
    main()
