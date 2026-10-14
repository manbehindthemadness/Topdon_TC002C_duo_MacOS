# This folder is the Custom node display name.
# ruff: noqa: N999
"""
HFHPE head detection, crop inference and display-only pose overlays.

Crop/axes adapted from PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation.
MIT License
Copyright (c) 2026 Katsuya Hyodo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from based_thermals.custom_nodes.labels import confidence_mask, defer_labels
from based_thermals.custom_nodes.models import ModelResolver

from .geometry import Pose, draw_axes, original_anchor
from .inference import Detector, Estimator
from .settings import Settings
from .sources import detector_source

CONFIG_JSON = {
    "defaults": {
        "detector_model": "HGNet-N (fast)",
        "detector_path": "",
        "pose_path": "",
        "detector_device": {"backend": "coreml", "apple_compute": "CPUAndGPU"},
        "pose_device": {"backend": "cpu", "apple_compute": "CPUAndGPU"},
        "score_threshold": 0.7,
        "max_heads": 1,
        "input_rotation": "0",
        "show_boxes": True,
        "show_axes": True,
        "show_angles": True,
    },
    "controls": [
        {"key": "detector_model", "label": "Head detector", "type": "choice",
         "options": ["HGNet-N (fast)", "DINOv3-S (reference)"]},
        {"key": "detector_path", "label": "Detector ONNX override (empty = download)",
         "type": "model", "models": [
             {"name": "HGNet-N head detector",
              "sha256": "155c3c6f37dcf398288735f96a18738b67eb8f7d02ec54f024c75bed0a169e31",
              "filenames": ["deimv2_hgnetv2_n_wholebody49_boxes_only.onnx"]},
             {"name": "DINOv3-S head detector",
              "sha256": "9b64e68cd6cf2e085613ae1dd06a42941fc0b7fce57cd80441dfb45ce90a1323",
              "filenames": ["deimv2_dinov3_s_wholebody49_boxes_only.onnx"]},
         ]},
        {"key": "pose_path", "label": "Pose ONNX override (empty = download)", "type": "model",
         "models": [
             {"name": "HFHPE ViT-T 64×64 (rollgate v10)",
              "sha256": "be054658797c1610f8ac82a2ec0a4a5b5aff6df7c9efb459120133b1fd4a6895",
              "filenames": ["hfhpe_vitt_rollgate_1x3x64x64.onnx"]},
         ]},
        {"key": "detector_device", "label": "Detector device", "type": "device"},
        {"key": "pose_device", "label": "Pose device", "type": "device"},
        {"key": "score_threshold", "label": "Head confidence", "type": "number",
         "min": 0, "max": 1, "step": 0.05},
        {"key": "max_heads", "label": "Maximum heads (highest confidence first)", "type": "integer",
         "min": 1, "max": 16, "step": 1},
        {"key": "input_rotation", "label": "Model input rotation (CCW)", "type": "choice",
         "options": ["0", "90", "180", "270"]},
        {"key": "show_boxes", "label": "Head boxes", "type": "boolean"},
        {"key": "show_axes", "label": "Pose axes", "type": "boolean"},
        {"key": "show_angles", "label": "Yaw/pitch/roll labels", "type": "boolean"},
    ],
}

MODEL_SOURCE = {
    "name": "HFHPE ViT-T 64 rollgate v10",
    "url": "https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/vitt_distill_064_yp_unified_v10.tar.gz",
    "archive_member": "vitt_distill_064_yp_unified_v10/hfhpe_vitt_rollgate_1x3x64x64.onnx",
    "archive_sha256": "ac06a442b9a0d9e308250ff7792a2cd3819c8597dfe3031c931bacd0b1379f39",
    "sha256": "be054658797c1610f8ac82a2ec0a4a5b5aff6df7c9efb459120133b1fd4a6895",
    "max_bytes": 200_000_000,
}

_models = ModelResolver()
_detector: Detector | None = None
_estimator: Estimator | None = None
_detector_key: tuple[Any, ...] | None = None
_pose_key: tuple[Any, ...] | None = None
last_poses: list[Pose] = []
last_timings: dict[str, float] = {}


def local_model(value: object, label: str) -> Path:
    """
    Resolve explicit external weights without importing or downloading in the UI.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Select {label}; run tools/prepare_head_pose.py first")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Selected {label} does not exist: {path}")
    return path


def session_key(path: Path, device: dict[str, str]) -> tuple[Any, ...]:
    """
    Rebuild sessions when model bytes on disk or the effective device change.
    """
    info = path.stat()
    key = (str(path), info.st_mtime_ns, info.st_ctime_ns, info.st_size,
           device["backend"], device["apple_compute"])
    return key


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Infer on upright display pixels, then restore source geometry and defer text.
    """
    global _detector, _estimator, _detector_key, _pose_key, last_poses, last_timings
    frame_settings = Settings.read(config)
    last_poses, last_timings = [], {}
    source = detector_source(config.get("detector_model", "HGNet-N (fast)"))
    detector_override = config.get("detector_path", "")
    if not isinstance(detector_override, str):
        raise TypeError("Detector ONNX override must be a path string")
    override = config.get("pose_path", "")
    if not isinstance(override, str):
        raise TypeError("Pose ONNX override must be a path string")
    detector_path = (local_model(detector_override, "detector ONNX") if detector_override
                     else _models.resolve(source))
    pose_path = (local_model(override, "pose ONNX") if override
                 else _models.resolve(config.get("model", MODEL_SOURCE)))
    if detector_path is None or pose_path is None:
        return image
    detector_key = session_key(detector_path, config["detector_device"])
    pose_key = session_key(pose_path, config["pose_device"])
    detector = _detector
    if detector is None or detector_key != _detector_key:
        detector = Detector(detector_path, config["detector_device"])
        _detector = detector
        _detector_key = detector_key
    estimator = _estimator
    if estimator is None or pose_key != _pose_key:
        estimator = Estimator(pose_path, config["pose_device"])
        _estimator = estimator
        _pose_key = pose_key
    upright = np.ascontiguousarray(np.rot90(np.clip(image, 0, 255), frame_settings.rotation),
                                   dtype=np.float32)
    heads = detector.detect(upright, frame_settings.threshold, frame_settings.max_heads)
    poses = estimator.estimate(upright, heads)
    last_poses = poses
    last_timings = {"detector_ms": detector.last_ms, "pose_ms": estimator.last_ms}
    output = upright.copy()
    labels = []
    for pose in poses:
        x1, y1, x2, y2 = pose.head.box
        if frame_settings.boxes:
            cv2.rectangle(output, (round(x1), round(y1)), (round(x2), round(y2)),
                          (0, 255, 0), 1, cv2.LINE_AA)
        if frame_settings.axes:
            draw_axes(output, pose)
        if frame_settings.angles:
            yaw, pitch, roll = pose.angles
            text = f"y{yaw:.1f} p{pitch:+.1f} r{roll:+.1f}"
            anchor = original_anchor(round(x1), max(0, round(y1) - 6),
                                     (int(image.shape[0]), int(image.shape[1])),
                                     frame_settings.rotation)
            labels.append((text, anchor))
    restored = np.ascontiguousarray(np.rot90(output, -frame_settings.rotation), dtype=np.float32)
    if labels and not defer_labels(labels, image.shape[:2]):
        mask = confidence_mask(image.shape[:2], labels)
        restored[mask > 0] = 255 - restored[mask > 0]
    return restored
