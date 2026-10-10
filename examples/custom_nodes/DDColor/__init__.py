# This folder is a display label; the worker generates the Python module name.
# ruff: noqa: N999
"""
DDColor chroma prediction for display images, with original-resolution lightness.
"""

from pathlib import Path
from typing import Any

import numpy as np

from topdon_duo.custom_nodes.models import ModelResolver

from .colorizer import Colorizer

CONFIG_JSON = {
    "defaults": {
        "model_path": "",
        "device": {"backend": "coreml", "apple_compute": "CPUAndGPU"},
        "strength": 1.0,
        "invert_input": False,
        "input_rotation": "0",
    },
    "controls": [
        {"key": "model_path", "label": "Local ONNX model override", "type": "model"},
        {"key": "device", "label": "Inference device", "type": "device"},
        {"key": "strength", "label": "Color strength", "type": "number",
         "min": 0, "max": 1, "step": 0.05},
        {"key": "invert_input", "label": "Invert model input", "type": "boolean"},
        {"key": "input_rotation", "label": "Model input rotation (CCW)", "type": "choice",
         "options": ["0", "90", "180", "270"]},
    ],
}

MODEL_SOURCE = {
    "name": "DDColor tiny FP16 (512)",
    "url": "https://huggingface.co/edgetools/ddcolor/resolve/c4c98361d19eda29908cc960be39e8eeb44fc530/ddcolor-tiny-fp16.onnx",
    "sha256": "2653da00dc15e54a45e5200b61dbf82ee9ceaf56b02bb9b9657569ac775e82e6",
    "max_bytes": 150_000_000,
}

_models = ModelResolver()
_colorizer: Colorizer | None = None


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Predict color while leaving source lightness, dimensions and radiometry separate.
    """
    global _colorizer
    strength = config.get("strength", 1.0)
    inverted = config.get("invert_input", False)
    rotation = config.get("input_rotation", "0")
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError("Color strength must be between 0 and 1")
    if type(inverted) is not bool or rotation not in ("0", "90", "180", "270"):
        raise ValueError("Invalid model input polarity or rotation")
    if strength == 0:
        return image
    local = config.get("model_path", "")
    if not isinstance(local, str):
        raise TypeError("Local model path must be text")
    if local:
        path = Path(local).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Selected DDColor ONNX model does not exist")
    else:
        source = config.get("model", MODEL_SOURCE)
        if not isinstance(source, (str, dict)):
            raise TypeError("Model source must be a catalog name or pinned HTTPS descriptor")
        path = _models.resolve(source)
        if path is None:
            return image
    device = config["device"]
    key = (str(path), device["backend"], device["apple_compute"])
    engine = _colorizer
    if engine is None or engine.key != key:
        engine = Colorizer(path, device)
        _colorizer = engine
    output = engine.apply(image, float(strength), inverted, int(rotation))
    return output
