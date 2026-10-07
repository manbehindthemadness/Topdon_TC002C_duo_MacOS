"""Bounded classic OpenCV display filters; no models or optional contrib modules."""

import cv2
import numpy as np

from .view_settings import IMAGE_FILTERS

FILTER_MODES = {
    **IMAGE_FILTERS,
    "box": "Box blur",
    "sobel": "Sobel edges",
    "scharr": "Scharr edges",
    "laplacian": "Laplacian edges",
    "canny": "Canny edges",
    "clahe": "Local contrast (CLAHE)",
    "equalize": "Histogram equalization",
    "threshold": "Threshold / Otsu",
    "adaptive": "Adaptive threshold",
    "morphology": "Morphology",
    "emboss": "Emboss",
    "highpass": "High-pass detail",
}
KERNELS = (0, 3, 5, 7, 9, 11, 15, 21)
FILTER_FIELDS = {
    "none": set(),
    "bilateral": {"kernel", "sigma_color", "sigma_space", "border", "channels"},
    "median": {"kernel", "channels"},
    "gaussian": {"kernel", "sigma", "border", "channels"},
    "sharpen": {"kernel", "sigma", "amount", "border", "channels"},
    "box": {"kernel", "border", "channels"},
    "sobel": {"kernel", "direction", "gain", "border"},
    "scharr": {"direction", "gain", "border"},
    "laplacian": {"kernel", "gain", "border"},
    "canny": {"kernel", "edge_low", "edge_high", "l2_gradient"},
    "clahe": {"clip_limit", "tile_size"},
    "equalize": set(),
    "threshold": {"threshold", "threshold_type", "maximum"},
    "adaptive": {"block_size", "adaptive_method", "adaptive_c", "invert", "maximum"},
    "morphology": {"kernel", "morph_operation", "shape", "iterations", "border", "channels"},
    "emboss": {"emboss_direction", "gain", "offset", "border"},
    "highpass": {"kernel", "sigma", "gain", "offset", "border", "channels"},
}


def filter_fields(params):
    fields = {"filter", *FILTER_FIELDS[params["filter"]]}
    if params["filter"] != "none":
        fields.add("mix")
    if params["filter"] == "threshold" and params["threshold_type"].startswith("otsu"):
        fields.discard("threshold")
    return fields


def allowed_kernels(mode):
    maximum = (
        7
        if mode in ("sobel", "laplacian", "canny")
        else 9
        if mode in ("bilateral", "median")
        else 21
    )
    return tuple(k for k in KERNELS if k <= maximum)


def validate_filter(params):
    if params["kernel"] not in allowed_kernels(params["filter"]):
        raise ValueError("Kernel too large for this filter; use Auto or a smaller kernel")
    if params["filter"] == "canny" and params["edge_low"] >= params["edge_high"]:
        raise ValueError("Canny lower threshold must be below upper threshold")


def _bytes(image):
    return np.clip(image, 0, 255).round().astype(np.uint8)


def _core(image, p):
    mode = p["filter"]
    border = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
    }[p["border"]]
    kernel = int(p["kernel"] or (5 if mode == "bilateral" else 3))
    size = (kernel, kernel)
    if mode == "box":
        return cv2.blur(image, size, borderType=border)
    if mode == "median":
        return cv2.medianBlur(_bytes(image), kernel).astype(np.float32)
    if mode == "bilateral":
        return cv2.bilateralFilter(
            image.astype(np.float32), kernel, p["sigma_color"], p["sigma_space"], borderType=border
        )
    if mode in ("gaussian", "sharpen", "highpass"):
        size = (0, 0) if not p["kernel"] else size
        blurred = cv2.GaussianBlur(image, size, p["sigma"], borderType=border)
        if mode == "gaussian":
            return blurred
        if mode == "sharpen":
            return np.clip(image * (1 + p["amount"]) - blurred * p["amount"], 0, 255)
        return np.clip((image - blurred) * p["gain"] + p["offset"], 0, 255)
    if mode == "morphology":
        shape = {
            "rectangle": cv2.MORPH_RECT,
            "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS,
        }[p["shape"]]
        operation = {
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }[p["morph_operation"]]
        return cv2.morphologyEx(
            image,
            operation,
            cv2.getStructuringElement(shape, size),
            iterations=int(p["iterations"]),
            borderType=border,
        )
    if mode in ("sobel", "scharr"):

        def derivative(dx, dy):
            if mode == "scharr":
                return cv2.Scharr(image, cv2.CV_32F, dx, dy, borderType=border)
            return cv2.Sobel(image, cv2.CV_32F, dx, dy, ksize=kernel, borderType=border)

        direction = p["direction"]
        result = (
            np.abs(derivative(1, 0))
            if direction == "x"
            else np.abs(derivative(0, 1))
            if direction == "y"
            else cv2.magnitude(derivative(1, 0), derivative(0, 1))
        )
        return np.clip(result * p["gain"], 0, 255)
    if mode == "laplacian":
        return np.clip(
            np.abs(cv2.Laplacian(image, cv2.CV_32F, ksize=kernel, borderType=border)) * p["gain"],
            0,
            255,
        )
    if mode == "canny":
        return cv2.Canny(
            _bytes(image),
            p["edge_low"],
            p["edge_high"],
            apertureSize=kernel,
            L2gradient=p["l2_gradient"],
        ).astype(np.float32)
    if mode == "clahe":
        grid = int(p["tile_size"])
        return (
            cv2.createCLAHE(clipLimit=p["clip_limit"], tileGridSize=(grid, grid))
            .apply(_bytes(image))
            .astype(np.float32)
        )
    if mode == "equalize":
        return cv2.equalizeHist(_bytes(image)).astype(np.float32)
    if mode == "threshold":
        threshold_type = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV,
            "otsu": cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            "otsu_inv": cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        }[p["threshold_type"]]
        return cv2.threshold(_bytes(image), p["threshold"], p["maximum"], threshold_type)[1].astype(
            np.float32
        )
    if mode == "adaptive":
        method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if p["adaptive_method"] == "mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        threshold_type = cv2.THRESH_BINARY_INV if p["invert"] else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            _bytes(image),
            p["maximum"],
            method,
            threshold_type,
            int(p["block_size"]),
            p["adaptive_c"],
        ).astype(np.float32)
    if mode == "emboss":
        matrix = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], np.float32)
        matrix = np.rot90(matrix, ("se", "ne", "nw", "sw").index(p["emboss_direction"])).copy()
        return np.clip(
            cv2.filter2D(
                image, cv2.CV_32F, matrix * p["gain"], delta=p["offset"], borderType=border
            ),
            0,
            255,
        )
    raise ValueError("Unknown image filter")


def apply_filter(image, p):
    if p["filter"] == "none" or not p["mix"]:
        return image
    mode = p["filter"]
    color = None
    grayscale = mode in ("sobel", "scharr", "laplacian", "canny", "threshold", "adaptive", "emboss")
    luma = grayscale or mode in ("clahe", "equalize") or p["channels"] == "luminance"
    if luma:
        color = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2YCrCb)
        processed = _core(color[..., 0], p)
        if grayscale:
            filtered = np.repeat(processed[..., None], 3, axis=2)
        else:
            color[..., 0] = processed
            filtered = np.clip(cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR), 0, 255)
    else:
        filtered = _core(image.astype(np.float32), p)
    return np.clip(
        cv2.addWeighted(
            image.astype(np.float32), 1 - p["mix"], filtered.astype(np.float32), p["mix"], 0
        ),
        0,
        255,
    )
