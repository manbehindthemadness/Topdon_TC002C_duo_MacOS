"""
Validated static ONNX sessions for a pruned DEIMv2 detector and integrated HFHPE.
"""

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from based_thermals.custom_nodes.devices import onnx_providers

from .geometry import Head, Pose, crop_head
from .sources import NORMALIZATIONS


def make_session(path: Path, device: dict[str, str]) -> Any:
    """
    Use the viewer's effective providers with bounded CPU parallelism.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    providers = onnx_providers(device)
    names = {entry if isinstance(entry, str) else entry[0] for entry in providers}
    if names - set(ort.get_available_providers()):
        raise ValueError("Requested ONNX provider is unavailable")
    session = ort.InferenceSession(str(path), options, providers=providers)
    session.disable_fallback()
    if names - set(session.get_providers()):
        raise ValueError("GPU initialization failed; select CPU to diagnose this model")
    return session


def image_input(session: Any) -> tuple[str, int, int]:
    """
    Accept only single-image float32 RGB exports with bounded static dimensions.
    """
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("Use a single-input batch-1 ONNX export")
    value = inputs[0]
    shape = value.shape
    if (value.type != "tensor(float)" or len(shape) != 4 or shape[:2] != [1, 3]
            or any(type(size) is not int or not 32 <= size <= 1024 for size in shape[2:])):
        raise ValueError("Use float32 [1,3,H,W] ONNX input with fixed dimensions")
    return value.name, shape[2], shape[3]


class Detector:
    """
    Cache a segmentation-free DEIMv2 detector with explicit normalization metadata.
    """

    def __init__(self, path: Path, device: dict[str, str]) -> None:
        """
        Accept prepared graphs or the exact audited release bytes with known normalization.
        """
        self.session = make_session(path, device)
        self.name, self.height, self.width = image_input(self.session)
        meta = self.session.get_modelmeta().custom_metadata_map
        normalization = meta.get("based_thermals.detector_norm")
        if meta.get("based_thermals.detector_pruned") != "1" or normalization not in (
            "imagenet", "div255",
        ):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            normalization = NORMALIZATIONS.get(digest)
            if normalization is None:
                raise ValueError("Use a pinned detector release or prepare its graph with "
                                 "tools/prepare_head_pose.py before loading")
        self.normalization = normalization
        outputs = self.session.get_outputs()
        if (len(outputs) != 1 or outputs[0].name != "label_xyxy_score"
                or outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 3
                or outputs[0].shape[0] != 1 or outputs[0].shape[-1] != 6):
            raise ValueError("Use a DEIMv2 normalized label_xyxy_score [1,Q,6] export")
        self.last_ms = 0.0

    def detect(self, image: np.ndarray, threshold: float, maximum: int) -> list[Head]:
        """
        Filter class 7 and select the highest-confidence heads before pose inference.
        """
        resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if self.normalization == "imagenet":
            rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32,
            )
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        started = time.perf_counter()
        values = self.session.run(["label_xyxy_score"], {self.name: tensor})
        self.last_ms = (time.perf_counter() - started) * 1000
        if len(values) != 1 or not isinstance(values[0], np.ndarray):
            raise ValueError("Detector returned invalid boxes")
        boxes = values[0]
        if (boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 6
                or not np.isfinite(boxes).all()):
            raise ValueError("Detector returned invalid boxes")
        height, width = image.shape[:2]
        heads = []
        for label, x1, y1, x2, y2, score in boxes[0]:
            if label != 7 or not threshold <= score <= 1:
                continue
            x1, x2 = np.clip([x1 * width, x2 * width], 0, width - 1)
            y1, y2 = np.clip([y1 * height, y2 * height], 0, height - 1)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                heads.append(Head((float(x1), float(y1), float(x2), float(y2)), float(score)))
        heads.sort(key=lambda head: -head.score)
        return heads[:maximum]


class Estimator:
    """
    Retain a fixed batch-1, center05, integrated yaw/pitch/roll HFHPE session.
    """

    def __init__(self, path: Path, device: dict[str, str]) -> None:
        """
        Reject standalone yaw/pitch and dynamic-batch exports explicitly.
        """
        self.session = make_session(path, device)
        self.name, height, width = image_input(self.session)
        outputs = {value.name: value for value in self.session.get_outputs()}
        meta = self.session.get_modelmeta().custom_metadata_map
        if (height != width or height not in (64, 128)
                or set(outputs) != {"cos_sin", "kappa"}
                or outputs["cos_sin"].shape != [1, 6] or outputs["kappa"].shape != [1, 3]
                or any(value.type != "tensor(float)" for value in outputs.values())
                or "mean=" in meta.get("input.images", "")):
            raise ValueError("Use integrated center05 HFHPE batch-1 64/128 ONNX with cos_sin/kappa")
        self.size = height
        self.last_ms = 0.0

    def estimate(self, image: np.ndarray, heads: list[Head]) -> list[Pose]:
        """
        Preserve upstream crop, RGB normalization and atan2 angle conventions.
        """
        poses = []
        self.last_ms = 0.0
        for head in heads:
            crop = crop_head(image, head, self.size)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            tensor = np.ascontiguousarray((rgb / 127.5 - 1).transpose(2, 0, 1)[None])
            started = time.perf_counter()
            values = self.session.run(["cos_sin", "kappa"], {self.name: tensor})
            self.last_ms += (time.perf_counter() - started) * 1000
            if (len(values) != 2 or not all(isinstance(value, np.ndarray) for value in values)
                    or values[0].shape != (1, 6) or values[1].shape != (1, 3)
                    or not all(np.isfinite(value).all() for value in values)
                    or (values[1] < 0).any()):
                raise ValueError("HFHPE returned invalid pose/confidence tensors")
            pairs = values[0][0].reshape(3, 2)
            if np.any(np.less(np.linalg.norm(pairs, axis=1), 1e-6)):
                raise ValueError("HFHPE returned an undefined angle")
            angles = np.degrees(np.arctan2(pairs[:, 1], pairs[:, 0]))
            angles[0] %= 360
            yaw, pitch, roll = (float(angle) for angle in angles)
            yaw_kappa, pitch_kappa, roll_kappa = (float(value) for value in values[1][0])
            poses.append(Pose(head, (yaw, pitch, roll), (yaw_kappa, pitch_kappa, roll_kappa)))
        return poses
