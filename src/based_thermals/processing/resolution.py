"""
Bound display raster promotion without changing saved working-canvas coordinates.
"""

from typing import Any

from ..enhancement_limits import MAX_PIXELS
from ..onnx_models import MODELS

MODEL_NODES = {"onnx_superresolution", "onnx_denoise", "onnx_style", "enhance", "coreml_acnet"}


def model_geometry(item: dict[str, Any]) -> tuple[int, bool]:
    """
    Return the declared model scale and whether it explicitly resets its input resolution.
    """
    kind, p = item["type"], item["params"]
    if kind not in MODEL_NODES or not p["amount"]:
        return 1, False
    model = p.get("model", "acnet")
    if model == "off":
        return 1, False
    factor = MODELS[model]["factor"] if model in MODELS else (
        2 ** int(p["passes"]) if model == "acnet" and kind != "coreml_acnet" else 2
    )
    resets = kind != "onnx_denoise" and factor > 1 and p["input"] != "current"
    return factor, resets


def fits_processing(
    size: tuple[int, int], following: list[dict[str, Any]], native: tuple[int, int],
) -> bool:
    """
    Reserve space for declared downstream resolution operations before promoting a raster.

    Custom output dimensions remain subject to the worker's existing runtime size validation.
    """
    width, height = size
    if width * height > MAX_PIXELS:
        return False
    for item in following:
        kind, p = item["type"], item["params"]
        factor = 1
        if kind == "interpolation":
            factor = p["scale"]
        elif kind == "transform" and p["resize_mode"] == "resolution":
            width = max(1, round(width * p["width"] / 100))
            height = max(1, round(height * p["height"] / 100))
        elif kind in MODEL_NODES:
            factor, resets = model_geometry(item)
            if resets:
                multiplier = 1 if p["input"] == "native" else 2
                width, height = native[0] * multiplier, native[1] * multiplier
        width, height = width * factor, height * factor
        if width * height > MAX_PIXELS:
            return False
    return True


def bounded_promotion(
    current: tuple[int, int], desired: tuple[int, int], following: list[dict[str, Any]],
    native: tuple[int, int],
) -> tuple[int, int]:
    """
    Increase sampling density only as far as the remaining declared allocations permit.
    """
    desired = max(current[0], desired[0]), max(current[1], desired[1])
    if fits_processing(desired, following, native):
        return desired
    if not fits_processing(current, following, native):
        return current
    lower, upper = 0.0, 1.0
    result = current
    for _ in range(24):
        fraction = (lower + upper) / 2
        candidate = (
            current[0] + int((desired[0] - current[0]) * fraction),
            current[1] + int((desired[1] - current[1]) * fraction),
        )
        if fits_processing(candidate, following, native):
            lower, result = fraction, candidate
        else:
            upper = fraction
    return result


def raster_transform_params(
    params: dict[str, Any], logical: tuple[int, int], raster: tuple[int, int],
) -> dict[str, Any]:
    """
    Keep pixel/percent positions in the declared canvas units when its raster is denser.
    """
    if logical == raster:
        return params
    result = dict(params)
    for axis, original, actual in zip(("x", "y"), logical, raster):
        distance = params[f"position_{axis}"]
        if params["position_unit"] == "percent":
            distance *= original / 100
        result[f"position_{axis}"] = distance * max(1, actual - 1) / max(1, original - 1)
    result["position_unit"] = "pixels"
    return result
