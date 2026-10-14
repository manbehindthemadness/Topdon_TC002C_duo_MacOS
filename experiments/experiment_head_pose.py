"""
Run Head Pose on saved images, reporting cold/warm timings and portable presets.
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
import onnxruntime as ort

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.labels import confidence_mask
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.pipeline import default_pipeline, node, validate_pipeline


def file_hash(path: Path) -> str:
    """
    Record the precise external models used for each experiment.
    """
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest


def run_image(args: argparse.Namespace, image_path: Path, backend: str) -> dict[str, Any]:
    """
    Run the real Custom worker, preserving image geometry and collecting predictions.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = image.astype(np.float32)
    folder = Path(__file__).resolve().parents[1] / "examples/custom_nodes/head-pose/nodes/Head Pose"
    params = load_folder(folder)
    config = json.loads(params["config"])
    config.update(detector_path=str(args.detector.resolve()), pose_path=str(args.pose.resolve()),
                  score_threshold=args.threshold, input_rotation=args.rotation,
                  max_heads=args.max_heads)
    config["detector_device"]["backend"] = backend
    config["pose_device"]["backend"] = args.pose_backend
    params["config"] = json.dumps(config)
    item = node("software", "custom", **params)
    pipeline = default_pipeline()
    pipeline["hardware"] = []
    source = pipeline["software"][0]
    source["params"]["source"] = "raw"
    pipeline["software"] = [source, item, node("software", "output")]
    document = validate_pipeline(json.loads(json.dumps(pipeline)))
    item = document["software"][1]
    available = set(ort.get_available_providers())
    processor = CustomProcessor(apple_available="CoreMLExecutionProvider" in available,
                                nvidia_available="CUDAExecutionProvider" in available)
    try:
        durations, stages = [], []
        first = 0.0
        output = image
        callback: FunctionType | None = None
        for index in range(args.runs + 2):
            processor.labels.clear()
            started = time.perf_counter()
            output = processor.apply(image, item)
            elapsed = (time.perf_counter() - started) * 1000
            worker_callback = processor.packages[item["id"]].process
            if not isinstance(worker_callback, FunctionType):
                raise TypeError("Expected a Head Pose package function")
            callback = worker_callback
            if index == 0:
                first = elapsed
            if index >= 2:
                durations.append(elapsed)
                stages.append(worker_callback.__globals__["last_timings"])
        if output.shape != image.shape or output.dtype != np.float32:
            raise ValueError("Head Pose changed the input geometry or dtype")
        if callback is None:
            raise ValueError("Experiment did not execute any frames")
        for label in processor.labels:
            anchor = (round(label.x * (image.shape[1] - 1)),
                      round(label.y * (image.shape[0] - 1)))
            mask = confidence_mask((int(image.shape[0]), int(image.shape[1])), [(label.text, anchor)])
            output[mask > 0] = 255 - output[mask > 0]
        namespace = callback.__globals__
        detector, estimator = namespace["_detector"], namespace["_estimator"]
        predictions = [{"box": list(pose.head.box), "score": pose.head.score,
                        "yaw": pose.angles[0], "pitch": pose.angles[1], "roll": pose.angles[2],
                        "kappa": list(pose.kappa)} for pose in namespace["last_poses"]]
        result = {
            "image": str(image_path), "requested_backend": backend,
            "requested_pose_backend": args.pose_backend,
            "detector_providers": detector.session.get_providers(),
            "pose_providers": estimator.session.get_providers(),
            "input_rotation_ccw": args.rotation, "threshold": args.threshold,
            "first_call_ms": first, "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "detector_median_ms": float(np.median([s["detector_ms"] for s in stages])),
            "pose_median_ms": float(np.median([s["pose_ms"] for s in stages])),
            "shape": list(output.shape), "predictions_in_rotated_input_pixels": predictions,
        }
        label = f"{image_path.stem}-{backend}"
        if not cv2.imwrite(str(args.output / f"{label}.png"), output.astype(np.uint8)):
            raise ValueError("Could not write experiment image")
        (args.output / f"{label}.pipeline.json").write_text(json.dumps(document, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return result
    finally:
        processor.close()


def main() -> None:
    """
    Measure explicit local weights and stills without opening a camera or changing preferences.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path,
                        default=Path("diagnostics/head-pose/models/detector.onnx"))
    parser.add_argument("--pose", type=Path, default=Path("diagnostics/head-pose/models/pose.onnx"))
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/head-pose/results"))
    parser.add_argument("--backends", nargs="+", choices=("cpu", "coreml", "cuda"), default=["cpu"])
    parser.add_argument("--pose-backend", choices=("cpu", "coreml", "cuda"), default="cpu")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-heads", type=int, default=1)
    parser.add_argument("--rotation", choices=("0", "90", "180", "270"), default="0")
    args = parser.parse_args()
    if args.runs < 1 or not 0 <= args.threshold <= 1 or not 1 <= args.max_heads <= 16:
        parser.error("Invalid run count, confidence threshold or maximum heads")
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
              "detector_sha256": file_hash(args.detector), "pose_sha256": file_hash(args.pose),
              "onnxruntime": ort.__version__, "results": []}
    for image in args.images:
        for backend in args.backends:
            result = run_image(args, image, backend)
            report["results"].append(result)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
