"""
Test external DDColor ONNX weights through the Custom worker without camera access.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import FunctionType
from typing import Any

import cv2
import numpy as np

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import default_pipeline, node, validate_pipeline


def main() -> None:
    """
    Write actual CPU/GPU timings, colorized stills and a portable local-weight preset.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "coreml", "cuda"),
                        default=["cpu", "coreml"])
    parser.add_argument("--rotation", choices=("0", "90", "180", "270"), default="0")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    loaded = cv2.imread(str(args.image))
    if loaded is None:
        raise ValueError("Cannot read display test image")
    image = loaded.astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    folder = Path(__file__).parents[1] / "examples" / "custom_nodes" / "DDColor"
    params = load_folder(folder)
    report: dict[str, Any] = {"image": str(args.image), "results": []}
    tiles = [("Thermal input", image)]
    for model in args.models:
        checksum = hashlib.sha256(model.read_bytes()).hexdigest()
        for backend in args.backends:
            config = json.loads(params["config"])
            config.update(model_path=str(model.resolve()), input_rotation=args.rotation,
                          invert_input=args.invert)
            config["device"]["backend"] = backend
            item = node("software", "custom", **params)
            item["params"]["config"] = json.dumps(config)
            pipeline = default_pipeline()
            pipeline["hardware"] = []
            pipeline["software"][0]["params"]["source"] = "raw"
            pipeline["software"] = [pipeline["software"][0], item, node("software", "output")]
            document = validate_pipeline(json.loads(json.dumps(pipeline)))
            item = document["software"][1]
            processor = CustomProcessor(apple_available=True, nvidia_available=True)
            try:
                durations = []
                output = None
                first = None
                for index in range(args.runs + 3):
                    started = time.perf_counter()
                    output = processor.apply(image, item)
                    elapsed = (time.perf_counter() - started) * 1000
                    if index == 0:
                        first = elapsed
                    if index >= 3:
                        durations.append(elapsed)
                if output is None or output.shape != image.shape or output.dtype != np.float32:
                    raise ValueError("Output lost native geometry or dtype")
                label = f"{model.stem}-{backend}"
                cv2.imwrite(str(args.output / f"{label}.png"), output.astype(np.uint8))
                np.save(args.output / f"{label}.npy", output)
                tiles.append((label, output))
                callback = processor.packages[item["id"]].process
                if not isinstance(callback, FunctionType):
                    raise TypeError("Expected the DDColor package's process function")
                engine = callback.__globals__["_colorizer"]
                result = {
                    "model": model.name, "sha256": checksum, "backend": backend,
                    "input_shape": engine.shape, "providers": engine.session.get_providers(),
                    "rotation": args.rotation, "inverted": args.invert,
                    "first_call_ms": first, "median_ms": float(np.median(durations)),
                    "p95_ms": float(np.percentile(durations, 95)),
                    "shape": list(output.shape), "mean": float(output.mean()),
                    "std": float(output.std()),
                }
                report["results"].append(result)
                print(json.dumps(result), flush=True)
                if backend == "coreml":
                    preset = args.output / f"{model.stem}-pipeline.json"
                    preset.write_text(json.dumps(document, indent=2) + "\n")
            finally:
                processor.close()
    cell_w, cell_h = 420, 370
    sheet = np.full((cell_h, len(tiles) * cell_w, 3), 245, dtype=np.uint8)
    for column, (label, pixels) in enumerate(tiles):
        x = column * cell_w
        cv2.putText(sheet, label, (x + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (20, 20, 20), 1, cv2.LINE_AA)
        height, width = pixels.shape[:2]
        scale = min(400 / width, 320 / height)
        resized = cv2.resize(pixels.astype(np.uint8),
                             (round(width * scale), round(height * scale)))
        sheet[40:40 + resized.shape[0], x + 8:x + 8 + resized.shape[1]] = resized
    cv2.imwrite(str(args.output / "comparison.png"), sheet)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
