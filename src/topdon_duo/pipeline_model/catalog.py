"""
Pipeline parameter definitions and available hardware/software nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..hardware_controls import HARDWARE_CONTROLS
from ..image_filters import FILTER_MODES, KERNELS
from ..view_settings import COLOR_PALETTES


@dataclass(frozen=True)
class Parameter:
    title: str
    default: object
    minimum: float = 0
    maximum: float = 1
    step: float = 1
    options: tuple = ()
    unit: str = ""

    def validate(self, value: Any) -> None:
        """
        Validate.
        """
        if self.options:
            if isinstance(self.default, bool):
                valid_type = type(value) is bool
            elif isinstance(self.default, (int, float)):
                valid_type = not isinstance(value, bool) and isinstance(value, (int, float))
            else:
                valid_type = type(value) is type(self.default)
            if not valid_type or value not in dict(self.options):
                raise ValueError(f"Invalid {self.title}")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not self.minimum <= value <= self.maximum
            or (self.step == 1 and int(value) != value)
        ):
            raise ValueError(f"Invalid {self.title}")


def choice(title: Any, default: Any, options: Any) -> Any:
    """
    Choice.
    """
    return Parameter(title, default, options=tuple(options))


def switch(title: Any, default: Any = False) -> Any:
    """
    Switch.
    """
    return choice(title, default, ((False, "Off"), (True, "On")))


def hardware_parameter(name: Any) -> Any:
    """
    Hardware parameter.
    """
    spec = HARDWARE_CONTROLS[name]
    return Parameter(
        spec.title, spec.minimum, spec.minimum, spec.maximum, spec.step, spec.options, spec.unit
    )


# Camera-style palettes are display approximations, never claimed to be the SDK LUTs.
CAMERA_GRADIENTS = {
    f"camera_{value}": label
    for value, label in HARDWARE_CONTROLS["palette"].options
    if value not in (1, 2)
}
PALETTES = {**COLOR_PALETTES, **CAMERA_GRADIENTS}
HARDWARE_NODES = {
    "preset": (
        "Camera processing preset",
        {
            "value": choice(
                "Preset",
                "balanced",
                (("balanced", "Balanced"), ("shadow", "Shadow"), ("soft", "Soft")),
            )
        },
    ),
    "brightness": ("Camera brightness", {"value": hardware_parameter("brightness")}),
    "contrast": ("Camera contrast", {"value": hardware_parameter("contrast")}),
    "gamma": ("Camera gamma", {"value": Parameter("Gamma", 50, 0, 100)}),
    "boost": (
        "Camera boost",
        {
            "value": choice(
                "Boost mode", 0, tuple((i, "Off" if i == 0 else f"Mode {i}") for i in range(4))
            )
        },
    ),
    "detail": (
        "Detail enhancement",
        {
            "enabled": switch("Detail enhancement", True),
            "amount": hardware_parameter("detail"),
            "fixed": switch("Fixed detail mode"),
        },
    ),
    "camera_colors": ("Camera colors", {"palette": hardware_parameter("palette")}),
    "noise": (
        "Camera noise reduction",
        {
            name: hardware_parameter(name)
            for name in ("noise_mode", "noise_general", "noise_spatial", "noise_temporal")
        },
    ),
    "humidity": ("Relative humidity", {"value": hardware_parameter("humidity")}),
}


def feature_parameters() -> Any:
    """
    Feature parameters.
    """
    return {
        "resolution": choice(
            "Detection max dimension", 512, ((256, "256 px"), (512, "512 px"), (1024, "1024 px"))
        ),
        "blur": choice(
            "Pre-detection smoothing", 3, ((0, "Off"), (3, "3 × 3"), (5, "5 × 5"), (7, "7 × 7"))
        ),
        "region": choice(
            "Selection region", "full", (("full", "Full image"), ("center", "Centered region"))
        ),
        "region_size": Parameter("Centered region size", 75, 1, 100, 1, unit="%"),
        "exclude_border": switch("Exclude features touching image border", True),
        "max_count": Parameter("Maximum selected features", 100, 1, 256),
        "color": choice(
            "Feature color",
            "cyan",
            (
                ("white", "White"),
                ("cyan", "Cyan"),
                ("yellow", "Yellow"),
                ("red", "Red"),
                ("black", "Black"),
            ),
        ),
        "thickness": Parameter("Outline / edge thickness", 1, 1, 5),
        "mix": Parameter("Blend amount", 1.0, 0, 1, 0.01),
        "lower": Parameter("Canny lower threshold", 50, 0, 255),
        "upper": Parameter("Canny upper threshold", 150, 0, 255),
    }


SOFTWARE_NODES = {
    "source": (
        "Image source",
        {
            "source": choice(
                "Image source",
                "preview",
                (("preview", "Camera preview"), ("raw", "Raw thermal image")),
            )
        },
    ),
    "output": ("Output → viewer", {}),
    "preview": ("Pipeline preview", {}),
    "combine": (
        "Combine pipeline",
        {
            "tab": choice(
                "Blend input",
                "B",
                (
                    ("preview", "Camera preview"),
                    ("raw", "Raw thermal"),
                    *((t, f"Tab {t}") for t in "ABCD"),
                ),
            ),
            "mode": choice(
                "Blend mode",
                "opacity",
                tuple(
                    (k, v)
                    for k, v in (
                        ("opacity", "Opacity"),
                        ("weighted", "Weighted sum"),
                        ("add", "Add"),
                        ("subtract", "Subtract"),
                        ("difference", "Difference"),
                        ("multiply", "Multiply"),
                        ("screen", "Screen"),
                        ("overlay", "Overlay"),
                        ("lighten", "Lighten (maximum)"),
                        ("darken", "Darken (minimum)"),
                        ("and", "Bitwise AND"),
                        ("or", "Bitwise OR"),
                        ("xor", "Bitwise XOR"),
                        ("mask", "Luminance mask"),
                    )
                ),
            ),
            "opacity": Parameter("Opacity", 0.5, 0, 1, 0.01),
            "mask_source": choice(
                "Optional mix mask",
                "none",
                (
                    ("none", "None"),
                    ("input", "Blend input"),
                    ("preview", "Camera preview"),
                    ("raw", "Raw thermal"),
                    *((t, f"Tab {t} output") for t in "ABCD"),
                ),
            ),
            "mask_kind": choice(
                "Mix mask type",
                "luminance",
                (
                    ("luminance", "Luminance (soft mix)"),
                    ("threshold", "Threshold (binary mask)"),
                ),
            ),
            "mask_threshold": Parameter("Mix mask threshold", 127, 0, 255),
            "mask_invert": switch("Invert mix mask"),
            "raw_low": Parameter("Raw input / mask from", 15.0, -20, 550, 0.1, unit="°C"),
            "raw_high": Parameter("Raw input / mask to", 45.0, -20, 550, 0.1, unit="°C"),
            "base_weight": Parameter("Current image weight", 1.0, 0, 2, 0.01),
            "input_weight": Parameter("Blend input weight", 1.0, 0, 2, 0.01),
            "offset": Parameter("Brightness offset", 0, -255, 255, 0.1),
            "threshold": Parameter("Mask threshold", 127, 0, 255),
            "invert": switch("Invert mask"),
            "interpolation": choice(
                "Match input size",
                "linear",
                (
                    ("nearest", "Nearest"),
                    ("linear", "Linear"),
                    ("bicubic", "Bicubic"),
                    ("lanczos", "Lanczos"),
                ),
            ),
        },
    ),
    "brightness": (
        "Software brightness",
        {"amount": Parameter("Brightness", 0, -100, 100, 0.1, unit="%")},
    ),
    "contrast": ("Software contrast", {"amount": Parameter("Contrast", 1.0, 0, 3, 0.01)}),
    "gamma": ("Software gamma", {"amount": Parameter("Gamma", 1.0, 0.1, 3, 0.01)}),
    "colors": ("App colors", {"palette": choice("Palette", "inferno", tuple(PALETTES.items()))}),
    "edges": (
        "Edge features",
        {
            **feature_parameters(),
            "method": choice(
                "Edge detector",
                "canny",
                (("canny", "Canny"), ("sobel", "Sobel"), ("scharr", "Scharr")),
            ),
            "auto_threshold": switch("Automatic Canny thresholds", True),
            "l2": switch("Accurate L2 gradient", True),
            "strength": Parameter("Minimum gradient strength", 25, 0, 255),
            "direction": choice(
                "Edge direction",
                "all",
                (
                    ("all", "All directions"),
                    ("horizontal", "Horizontal edges"),
                    ("vertical", "Vertical edges"),
                ),
            ),
            "min_pixels": Parameter("Minimum connected edge pixels", 10, 1, 10000),
            "rank": choice(
                "Select by",
                "longest",
                (("longest", "Longest connected edges"), ("center", "Nearest image center")),
            ),
            "output": choice(
                "Output",
                "overlay",
                (
                    ("overlay", "Colored edges over input"),
                    ("mask", "Binary edge mask"),
                    ("cutout", "Input at selected edges"),
                ),
            ),
        },
    ),
    "contours": (
        "Contour regions",
        {
            **feature_parameters(),
            "segmentation": choice(
                "Region detection",
                "otsu",
                (
                    ("otsu", "Otsu automatic threshold"),
                    ("threshold", "Manual threshold"),
                    ("adaptive", "Adaptive threshold"),
                    ("canny", "Canny closed boundaries"),
                ),
            ),
            "threshold": Parameter("Region threshold", 127, 0, 255),
            "invert": switch("Select dark regions", False),
            "block_size": choice(
                "Adaptive block size",
                11,
                tuple((k, f"{k} × {k}") for k in (3, 5, 7, 9, 11, 15, 21, 31)),
            ),
            "adaptive_c": Parameter("Adaptive offset C", 0.0, -30, 30, 0.1),
            "close_kernel": choice(
                "Close gaps", 3, ((0, "Off"), (3, "3 × 3"), (5, "5 × 5"), (7, "7 × 7"))
            ),
            "min_area": Parameter("Minimum region area", 0.05, 0, 100, 0.01, unit="%"),
            "max_area": Parameter("Maximum region area", 80.0, 0, 100, 0.01, unit="%"),
            "shape": choice(
                "Select shape",
                "any",
                (
                    ("any", "Any region"),
                    ("rectangle", "Rectangles / squares"),
                    ("circle", "Round regions"),
                    ("convex", "Convex regions"),
                ),
            ),
            "min_aspect": Parameter("Minimum long / short side ratio", 1.0, 1, 50, 0.01),
            "max_aspect": Parameter("Maximum long / short side ratio", 20.0, 1, 50, 0.01),
            "solidity": Parameter("Minimum solidity", 0, 0, 1, 0.01),
            "circularity": Parameter("Minimum circularity", 0, 0, 1, 0.01),
            "simplify": Parameter("Polygon simplification", 2.0, 0.1, 10, 0.1, unit="%"),
            "rank": choice(
                "Select by",
                "largest",
                (
                    ("largest", "Largest area"),
                    ("smallest", "Smallest area"),
                    ("brightest", "Brightest display region"),
                    ("darkest", "Darkest display region"),
                    ("center", "Nearest image center"),
                ),
            ),
            "geometry": choice(
                "Region geometry",
                "contour",
                (
                    ("contour", "Original contour"),
                    ("polygon", "Simplified polygon"),
                    ("hull", "Convex hull"),
                    ("box", "Rotated bounding box"),
                ),
            ),
            "output": choice(
                "Output",
                "overlay",
                (
                    ("overlay", "Colored outlines over input"),
                    ("fill", "Colored regions over input"),
                    ("mask", "Binary region mask"),
                    ("cutout", "Isolate selected regions"),
                    ("mean", "Flatten regions to average color"),
                ),
            ),
        },
    ),
    "filter": (
        "Image filter",
        {
            "filter": choice("Filter", "sharpen", tuple(FILTER_MODES.items())),
            "mix": Parameter("Blend amount", 1.0, 0, 1, 0.01),
            "kernel": choice(
                "Kernel size", 0, tuple((k, "Auto" if k == 0 else f"{k} × {k}") for k in KERNELS)
            ),
            "sigma": Parameter("Gaussian sigma", 0.8, 0.1, 10, 0.1),
            "sigma_color": Parameter("Color sigma", 21.0, 0.1, 150, 0.1),
            "sigma_space": Parameter("Spatial sigma", 3.0, 0.1, 10, 0.1),
            "channels": choice(
                "Channels",
                "color",
                (("color", "All color channels"), ("luminance", "Luminance only")),
            ),
            "border": choice(
                "Border handling",
                "reflect",
                (("reflect", "Reflect"), ("replicate", "Replicate edge"), ("constant", "Constant")),
            ),
            "direction": choice(
                "Edge direction",
                "magnitude",
                (
                    ("magnitude", "Magnitude (both axes)"),
                    ("x", "X derivative"),
                    ("y", "Y derivative"),
                ),
            ),
            "gain": Parameter("Response gain", 1.0, 0, 4, 0.01),
            "edge_low": Parameter("Canny lower threshold", 50, 0, 255),
            "edge_high": Parameter("Canny upper threshold", 150, 0, 255),
            "l2_gradient": switch("Accurate L2 gradient"),
            "clip_limit": Parameter("CLAHE clip limit", 2.0, 0.1, 40, 0.1),
            "tile_size": Parameter("CLAHE tiles per axis", 8, 2, 32),
            "threshold": Parameter("Threshold", 127, 0, 255),
            "threshold_type": choice(
                "Threshold method",
                "binary",
                (
                    ("binary", "Binary"),
                    ("binary_inv", "Binary inverted"),
                    ("trunc", "Truncate"),
                    ("tozero", "To zero"),
                    ("tozero_inv", "To zero inverted"),
                    ("otsu", "Otsu automatic"),
                    ("otsu_inv", "Otsu inverted"),
                ),
            ),
            "maximum": Parameter("Maximum output", 255, 1, 255),
            "block_size": choice(
                "Adaptive block size",
                11,
                tuple((k, f"{k} × {k}") for k in (3, 5, 7, 9, 11, 15, 21, 31)),
            ),
            "adaptive_method": choice(
                "Adaptive method", "gaussian", (("mean", "Mean"), ("gaussian", "Gaussian"))
            ),
            "adaptive_c": Parameter("Adaptive offset C", 2.0, -30, 30, 0.1),
            "invert": switch("Invert threshold"),
            "morph_operation": choice(
                "Morphology operation",
                "open",
                (
                    ("erode", "Erode"),
                    ("dilate", "Dilate"),
                    ("open", "Open"),
                    ("close", "Close"),
                    ("gradient", "Gradient"),
                    ("tophat", "Top hat"),
                    ("blackhat", "Black hat"),
                ),
            ),
            "shape": choice(
                "Kernel shape",
                "ellipse",
                (("rectangle", "Rectangle"), ("ellipse", "Ellipse"), ("cross", "Cross")),
            ),
            "iterations": Parameter("Iterations", 1, 1, 5),
            "emboss_direction": choice(
                "Emboss direction",
                "se",
                (
                    ("se", "Southeast"),
                    ("ne", "Northeast"),
                    ("nw", "Northwest"),
                    ("sw", "Southwest"),
                ),
            ),
            "offset": Parameter("Response offset", 128, 0, 255),
            "amount": Parameter("Amount", 0.7, 0, 3, 0.01),
        },
    ),
    "mirror": (
        "Mirror",
        {"horizontal": switch("Left / right"), "vertical": switch("Top / bottom")},
    ),
    "antialiasing": ("Antialiasing", {"amount": Parameter("Edge smoothing", 0.35, 0, 2, 0.01)}),
    "range": (
        "Temperature range",
        {
            "low": Parameter("From", 15.0, -20, 550, 0.1, unit="°C"),
            "high": Parameter("To", 45.0, -20, 550, 0.1, unit="°C"),
        },
    ),
    "interpolation": (
        "Sensor interpolation",
        {
            "method": choice(
                "Interpolation",
                "bicubic",
                (
                    ("nearest", "Nearest"),
                    ("linear", "Linear"),
                    ("bicubic", "Bicubic"),
                    ("lanczos", "Lanczos"),
                ),
            ),
            "scale": choice("Scale", 2, ((1, "1×"), (2, "2×"), (4, "4×"))),
        },
    ),
    "coreml_acnet": (
        "Apple Core ML ACNet (legacy)",
        {
            "compute": choice(
                "Compute devices",
                "CPUAndGPU",
                (
                    ("CPUOnly", "CPU only"),
                    ("CPUAndGPU", "CPU + GPU"),
                    ("ALL", "CPU + GPU + Neural Engine"),
                    ("CPUAndNeuralEngine", "CPU + Neural Engine"),
                ),
            ),
            "denoise": choice(
                "ACNet denoising", 0, ((0, "None"), (1, "Light"), (2, "Medium"), (3, "Strong"))
            ),
            "input": choice(
                "Input size",
                "native",
                (
                    ("current", "Current image"),
                    ("native", "Native sensor"),
                    ("preview", "Full preview"),
                ),
            ),
            "amount": Parameter("Enhancement amount", 1.0, 0, 1, 0.01),
        },
    ),
    "onnx_superresolution": (
        "ONNX super-resolution (legacy)",
        {
            "model": choice(
                "Model",
                "espcn",
                (
                    ("espcn", "ESPCN 3×"),
                    ("mewzoom", "MewZoom V0 4× (legacy)"),
                    ("mewzoom-v0-2x", "MewZoom V0 2× (legacy)"),
                    ("mewzoom-v1-2x", "MewZoom V1 2× (TrunkNet)"),
                    ("mewzoom-v1-4x", "MewZoom V1 4× (TrunkNet)"),
                    ("realesr-general-x4v3", "Real-ESRGAN general 4× v3"),
                ),
            ),
            "input": choice(
                "Input size",
                "native",
                (
                    ("current", "Current image"),
                    ("native", "Native sensor"),
                    ("preview", "Full preview"),
                ),
            ),
            "amount": Parameter("Enhancement amount", 1.0, 0, 1, 0.01),
            "backend": choice(
                "Execution",
                "cpu",
                (
                    ("cpu", "CPU (ONNX Runtime)"),
                    ("coreml", "Apple Core ML"),
                    ("cuda", "NVIDIA CUDA"),
                ),
            ),
            "apple_compute": choice(
                "Compute devices",
                "CPUAndGPU",
                (
                    ("CPUOnly", "CPU only"),
                    ("CPUAndGPU", "CPU + GPU"),
                    ("ALL", "CPU + GPU + Neural Engine"),
                    ("CPUAndNeuralEngine", "CPU + Neural Engine"),
                ),
            ),
        },
    ),
    "onnx_style": (
        "AI image styling",
        {
            "model": choice(
                "Style",
                "style-mosaic",
                (
                    ("style-mosaic", "Mosaic"),
                    ("style-candy", "Candy"),
                    ("style-rain-princess", "Rain Princess"),
                    ("style-udnie", "Udnie"),
                    ("style-pointillism", "Pointillism"),
                    ("style-line-art", "Line Art (Informative Drawings)"),
                    ("style-animegan-sketch", "AnimeGAN Portrait Sketch (non-commercial)"),
                ),
            ),
            "amount": Parameter("Style blend", 0.5, 0, 1, 0.01),
            "backend": choice(
                "Execution",
                "cpu",
                (
                    ("cpu", "CPU (ONNX Runtime)"),
                    ("coreml", "Apple Core ML"),
                    ("cuda", "NVIDIA CUDA"),
                ),
            ),
            "apple_compute": choice(
                "Compute devices",
                "CPUAndGPU",
                (
                    ("CPUOnly", "CPU only"),
                    ("CPUAndGPU", "CPU + GPU"),
                    ("ALL", "CPU + GPU + Neural Engine"),
                    ("CPUAndNeuralEngine", "CPU + Neural Engine"),
                ),
            ),
        },
    ),
    "onnx_denoise": (
        "ONNX denoising (legacy)",
        {
            "model": choice(
                "Model",
                "ffdnet-gray",
                (
                    ("ffdnet-gray", "FFDNet luminance (adjustable noise)"),
                    ("dncnn-25", "DnCNN luminance (fixed noise 25)"),
                ),
            ),
            "noise": Parameter("FFDNet noise sigma (not temperature)", 15, 0, 75, 1),
            "amount": Parameter("Denoising blend", 1.0, 0, 1, 0.01),
            "backend": choice(
                "Execution",
                "cpu",
                (
                    ("cpu", "CPU (ONNX Runtime)"),
                    ("coreml", "Apple Core ML"),
                    ("cuda", "NVIDIA CUDA"),
                ),
            ),
            "apple_compute": choice(
                "Compute devices",
                "CPUAndGPU",
                (
                    ("CPUOnly", "CPU only"),
                    ("CPUAndGPU", "CPU + GPU"),
                    ("ALL", "CPU + GPU + Neural Engine"),
                    ("CPUAndNeuralEngine", "CPU + Neural Engine"),
                ),
            ),
        },
    ),
    "enhance": (
        "AI enhancement",
        {
            "model": choice(
                "Model",
                "anime4k09",
                (
                    ("off", "Off"),
                    ("anime4k09", "Anime4K09 (CPU only)"),
                    ("acnet", "ACNet"),
                    ("espcn", "ESPCN 3×"),
                    ("mewzoom", "MewZoom V0 4× (legacy)"),
                    ("mewzoom-v0-2x", "MewZoom V0 2× (legacy)"),
                    ("mewzoom-v1-2x", "MewZoom V1 2× (TrunkNet)"),
                    ("mewzoom-v1-4x", "MewZoom V1 4× (TrunkNet)"),
                    ("realesr-general-x4v3", "Real-ESRGAN general 4× v3"),
                    ("dncnn-25", "DnCNN luminance (fixed noise 25)"),
                    ("ffdnet-gray", "FFDNet luminance (adjustable noise)"),
                ),
            ),
            "denoise": choice(
                "ACNet denoising", 0, ((0, "None"), (1, "Light"), (2, "Medium"), (3, "Strong"))
            ),
            "noise": Parameter("FFDNet noise sigma (not temperature)", 15, 0, 75, 1),
            "input": choice(
                "Input size",
                "current",
                (
                    ("current", "Current image"),
                    ("native", "Native sensor"),
                    ("preview", "Full preview"),
                ),
            ),
            "amount": Parameter("Enhancement amount", 1.0, 0, 1, 0.01),
            "passes": Parameter("Passes", 3, 1, 5),
            "backend": choice(
                "Execution",
                "cpu",
                (("cpu", "CPU"), ("coreml", "Apple Core ML"), ("cuda", "NVIDIA CUDA")),
            ),
            "apple_compute": choice(
                "Compute devices",
                "CPUAndGPU",
                (
                    ("CPUOnly", "CPU only"),
                    ("CPUAndGPU", "CPU + GPU"),
                    ("ALL", "CPU + GPU + Neural Engine"),
                    ("CPUAndNeuralEngine", "CPU + Neural Engine"),
                ),
            ),
        },
    ),
}
CATALOG = {"hardware": HARDWARE_NODES, "software": SOFTWARE_NODES}
# Import/runtime compatibility only: all new instances belong in AI enhancement.
LEGACY_SOFTWARE_NODES = frozenset({"coreml_acnet", "onnx_superresolution", "onnx_denoise"})
