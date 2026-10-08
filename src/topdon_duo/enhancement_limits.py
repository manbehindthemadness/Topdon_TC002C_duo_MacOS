"""Shared ACNet size budget for CPU/GPU execution and pipeline controls."""

from .onnx_models import MODELS as ONNX_MODELS

MAX_PIXELS = 4_000_000


def acnet_pass_limit(width, height):
    passes = 0
    pixels = width * height
    while passes < 5 and pixels * 4 <= MAX_PIXELS:
        pixels *= 4
        passes += 1
    return passes


def enhancement_pass_limits(document, *, clamp=False):
    """Estimate input sizes before each enhancement, including preceding resizes.

    Preview is conservatively full 512×384; raw/range sources are 256×192.
    Combining tabs resizes the incoming image to the current image's dimensions.
    """
    limits = {}
    for nodes in (document["software"], *document.get("branches", {}).values()):
        active = [item for item in nodes if not item["bypass"]]
        thermal = nodes[0]["params"]["source"] == "raw" or any(
            item["type"] == "range" for item in active
        )
        width, height = (256, 192) if thermal else (512, 384)
        for item in nodes:
            kind, p = item["type"], item["params"]
            if kind == "enhance":
                iw, ih = (
                    (256, 192)
                    if p["input"] == "native"
                    else (512, 384)
                    if p["input"] == "preview"
                    else (width, height)
                )
                maximum = acnet_pass_limit(iw, ih) if p["model"] == "acnet" else 5
                limits[item["id"]] = maximum
                if clamp and p["model"] == "acnet" and maximum:
                    p["passes"] = min(p["passes"], maximum)
                if item["bypass"] or p["model"] == "off" or not p["amount"]:
                    continue
                factor = 2 ** min(p["passes"], maximum) if p["model"] == "acnet" else 2
                width, height = iw * factor, ih * factor
            elif item["bypass"]:
                continue
            elif kind == "interpolation":
                width, height = width * p["scale"], height * p["scale"]
            elif kind == "onnx_superresolution" and p["amount"]:
                if p["input"] != "current":
                    width, height = (256, 192) if p["input"] == "native" else (512, 384)
                factor = ONNX_MODELS[p["model"]]["factor"]
                width, height = width * factor, height * factor
            elif kind == "coreml_acnet" and p["amount"]:
                if p["input"] != "current":
                    width, height = (256, 192) if p["input"] == "native" else (512, 384)
                width, height = width * 2, height * 2
    return limits
