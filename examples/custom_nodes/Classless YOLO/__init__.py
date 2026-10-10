# This folder names a Custom node; runtime generates its Python module name.
# ruff: noqa: N999
"""
Inspector-inspired classless object localization for the Custom pipeline node.
"""

from typing import Any

import numpy as np

from topdon_duo.custom_nodes.models import ModelResolver

from .detector import Detector
from .drawing import draw_detections
from .nesting import select_detections
from .settings import Settings

CONFIG_JSON = r'''
{
  "defaults": {
    "model_path": "",
    "providers": ["CPUExecutionProvider"],
    "score_threshold": 0.2,
    "score_maximum": 1.0,
    "filter_scores": true,
    "nested_coverage": 0.85,
    "suppress_duplicates": true,
    "duplicate_iou": 0.5,
    "max_detections": 25,
    "line_thickness": 1,
    "show_scores": false,
    "nesting": "Outside-in",
    "border_color": "#00ff00",
    "fill_color": "#00ff00",
    "fill_opacity": 0.0
  },
  "controls": [
    {"key": "filter_scores", "label": "Filter confidence scores", "type": "boolean"},
    {"key": "score_threshold", "label": "Minimum confidence", "type": "number",
     "min": 0, "max": 1, "step": 0.01},
    {"key": "score_maximum", "label": "Maximum confidence", "type": "number",
     "min": 0, "max": 1, "step": 0.01},
    {"key": "max_detections", "label": "Maximum detections", "type": "integer",
     "min": 1, "max": 100, "step": 1},
    {"key": "line_thickness", "label": "Box thickness", "type": "integer",
     "min": 1, "max": 8, "step": 1},
    {"key": "show_scores", "label": "Show confidence scores", "type": "boolean"},
    {"key": "nesting", "label": "Detection nesting", "type": "choice",
     "options": ["Inside-out", "Outside-in"]},
    {"key": "nested_coverage", "label": "Nested box coverage", "type": "number",
     "min": 0.5, "max": 1, "step": 0.05},
    {"key": "suppress_duplicates", "label": "Suppress duplicate boxes", "type": "boolean"},
    {"key": "duplicate_iou", "label": "Duplicate overlap (IoU)", "type": "number",
     "min": 0.01, "max": 1, "step": 0.05},
    {"key": "border_color", "label": "Box border color", "type": "color"},
    {"key": "fill_color", "label": "Box fill color", "type": "color"},
    {"key": "fill_opacity", "label": "Fill opacity", "type": "number",
     "min": 0, "max": 1, "step": 0.05},
    {"key": "model_path", "label": "Local model override", "type": "model"}
  ]
}
'''

MODEL_SOURCE = {
    "name": "Classless YOLO",
    "url": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/151_object_detection_mobile_object_localizer/resources.tar.gz",
    "sha256": "4f6f761beb78e4c2aac0033e4792564cf4d2e00cd8c866878c74ee236b51c837",
    "archive_member": "saved_model/model_float32.onnx",
    "archive_sha256": "4f2c91baab9066fd66902186ea428685a3b6394422143d90af78540b6f1b291f",
    "max_bytes": 70_000_000,
}

_detector: Detector | None = None
_models = ModelResolver()


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Detect objects in the current display image and overlay boxes without class labels.
    """
    global _detector
    options = Settings.from_config(config)
    model = options.model_path
    if not model:
        source = config.get("model", MODEL_SOURCE)
        if not isinstance(source, (str, dict)):
            raise ValueError("model must be a catalog name or download source object")
        path = _models.resolve(source)
        if path is None:
            return image
        model = str(path)
    engine = _detector
    if engine is None or engine.key != (model, options.providers):
        engine = Detector(model, options.providers)
        _detector = engine
    detections = engine.detect(image, options.threshold if options.filter_scores else 0.0)
    if options.filter_scores:
        detections = [box for box in detections if box[4] <= options.score_maximum]
    detections = select_detections(
        detections, options.nesting, options.max_detections, options.nested_coverage,
        options.duplicate_iou if options.suppress_duplicates else None,
    )
    output = draw_detections(image, detections, options.thickness, options.show_scores,
                             options.border_color, options.fill_color, options.fill_opacity)
    return output
