"""
CPU-first ONNX adaptation of Inspector's classless Mobile Object Localizer stage.
"""

import cv2
import numpy as np
import onnxruntime as ort

type Detection = tuple[float, float, float, float, float]


def decode_outputs(
    outputs: list[np.ndarray], threshold: float, maximum: int | None = None
) -> list[Detection]:
    """
    Decode PINTO's four-output export into normalized y1/x1/y2/x2 boxes and confidence.
    """
    if len(outputs) != 4:
        raise ValueError("Use the four-output PINTO model_float32.onnx export")
    boxes, classes, scores, count = (np.asarray(value) for value in outputs)
    if (
        boxes.ndim != 3
        or boxes.shape[0] != 1
        or boxes.shape[2] != 4
        or scores.shape != boxes.shape[:2]
        or classes.shape != scores.shape
        or count.shape != (1,)
        or any(not np.isfinite(value).all() for value in outputs)
    ):
        raise ValueError("Unexpected Mobile Object Localizer output shapes or non-finite values")
    number = float(count[0])
    if not number.is_integer() or not 0 <= number <= boxes.shape[1]:
        raise ValueError("Invalid Mobile Object Localizer detection count")
    selected = []
    for index in range(int(number)):
        score = float(scores[0, index])
        if not 0 <= score <= 1:
            raise ValueError("Invalid Mobile Object Localizer confidence")
        if score < threshold:
            continue
        y1, x1, y2, x2 = np.clip(boxes[0, index], 0, 1)
        if y2 > y1 and x2 > x1:
            selected.append((float(y1), float(x1), float(y2), float(x2), score))
    selected.sort(key=lambda box: box[4], reverse=True)
    return selected if maximum is None else selected[:maximum]


class Detector:
    """
    Cache one model session; weights are supplied explicitly outside the package.
    """

    def __init__(self, model_path: str, providers: tuple[str, ...]) -> None:
        """
        Load a local model for inference on the current pipeline image.
        """
        self.key = model_path, providers
        missing = set(providers) - set(ort.get_available_providers())
        if missing:
            raise ValueError(f"ONNX providers unavailable: {', '.join(sorted(missing))}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=list(providers)
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].shape != [1, 3, 192, 192]
            or inputs[0].type != "tensor(float)"
        ):
            raise ValueError("Expected float32 NCHW 1×3×192×192 Mobile Object Localizer input")
        if len(outputs) != 4:
            raise ValueError(
                "Use PINTO's four-output model_float32.onnx, not the two-output Luxonis export"
            )
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]

    def detect(
        self, image: np.ndarray, threshold: float, maximum: int | None = None
    ) -> list[Detection]:
        """
        Resize the entire BGR view, infer raw 0–255 RGB pixels, and decode classless boxes.
        """
        resized = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(np.clip(resized, 0, 255).astype(np.float32), cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        if any(not isinstance(output, np.ndarray) for output in outputs):
            raise ValueError("Expected dense NumPy output tensors from Mobile Object Localizer")
        arrays = [np.asarray(output) for output in outputs]
        detections = decode_outputs(arrays, threshold, maximum)
        return detections
