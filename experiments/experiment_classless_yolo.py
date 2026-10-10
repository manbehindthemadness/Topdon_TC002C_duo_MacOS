"""
Run the optional Classless YOLO example against local weights and a still image.

Requires an explicit PINTO Mobile Object Localizer ONNX export and readable image.
No model downloads, camera access, or viewer settings are performed.
"""

import argparse
import json
from pathlib import Path

import cv2

from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import node


def main() -> None:
    """
    Run a full folder-load/inference path and write the annotated still image.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1] / "examples/custom_nodes/Classless YOLO"
    params = load_folder(folder)
    config = json.loads(params["config"])
    config["model_path"] = str(args.model.resolve())
    params["config"] = json.dumps(config)
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")
    processor = CustomProcessor()
    try:
        item = node("software", "custom", **params)
        output = processor.apply(image.astype("float32"), item)
        if output.shape != image.shape:
            raise ValueError("Example changed image dimensions")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), output.astype("uint8")):
            raise ValueError(f"Could not write image: {args.output}")
        print(f"Classless YOLO: {output.shape}, {output.dtype}; output: {args.output}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()
