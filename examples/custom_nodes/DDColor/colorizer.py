"""
DDColor ONNX preprocessing and Lab reconstruction adapted from instant-high/DDColor-onnx.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from topdon_duo.custom_nodes.devices import onnx_providers, validate_device


class Colorizer:
    """
    Retain a validated DDColor session for one model and effective device.
    """

    def __init__(self, path: Path, device: dict[str, str]) -> None:
        """
        Accept fixed float32 RGB input and same-resolution two-channel Lab output.
        """
        settings = validate_device(device)
        self.key = (str(path), settings["backend"], settings["apple_compute"])
        providers = onnx_providers(settings)
        names = {value if isinstance(value, str) else value[0] for value in providers}
        if names - set(ort.get_available_providers()):
            raise ValueError("Requested ONNX provider is unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.session.disable_fallback()
        if names - set(self.session.get_providers()):
            raise ValueError("GPU initialization failed; select CPU to diagnose the model")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1
                or inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4 or inputs[0].shape[:2] != [1, 3]
                or any(type(size) is not int or size < 32 or size > 1024
                       for size in inputs[0].shape[2:])
                or outputs[0].shape != [1, 2, *inputs[0].shape[2:]]):
            raise ValueError("Use a DDColor export with float32 NCHW RGB input and Lab ab output")
        self.shape = tuple(inputs[0].shape)
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def apply(
        self, image: np.ndarray, strength: float, inverted: bool, rotation: int,
    ) -> np.ndarray:
        """
        Resize predicted chroma back to source pixels and retain the original Lab L.

        Input rotation/polarity affect the model only. Undo rotation before combining
        chroma with source lightness so later viewer transforms still align correctly.
        """
        source = np.clip(image, 0, 255).astype(np.float32) / 255
        original_l = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)[:, :, :1]
        model_image = 1 - source if inverted else source
        turns = rotation // 90
        rotated = np.ascontiguousarray(np.rot90(model_image, turns))
        resized = cv2.resize(rotated, (self.shape[3], self.shape[2]))
        lightness = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        gray_lab = np.concatenate((lightness, np.zeros_like(lightness),
                                   np.zeros_like(lightness)), axis=-1)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
        tensor = np.ascontiguousarray(gray_rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
            raise ValueError("DDColor returned an invalid chroma tensor")
        chroma = np.asarray(outputs[0])
        if (chroma.shape != (1, 2, self.shape[2], self.shape[3])
                or chroma.dtype != np.float32 or not np.isfinite(chroma).all()):
            raise ValueError("DDColor returned an invalid chroma tensor")
        resized_ab = cv2.resize(chroma[0].transpose(1, 2, 0),
                               (rotated.shape[1], rotated.shape[0]))
        original_ab = np.ascontiguousarray(np.rot90(resized_ab, -turns)) * strength
        lab = np.concatenate((original_l, original_ab), axis=-1)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        output = np.clip(bgr * 255, 0, 255).astype(np.float32)
        return output
