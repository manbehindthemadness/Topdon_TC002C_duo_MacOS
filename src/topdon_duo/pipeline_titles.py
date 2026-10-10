"""Concise node descriptions derived from settings, without changing node identities."""

from typing import Any

from .pipeline import CATALOG


def node_title(stack: str, item: dict[str, Any], temperature_unit: str = "C") -> str:
    """
    Describe a node's operation and settings for its title.
    """
    kind, p = item["type"], item["params"]
    title, definitions = CATALOG[stack][kind]

    def option(key: str) -> str:
        """
        Resolve a validated setting to its display label.
        """
        options: dict[Any, Any] = dict(definitions[key].options)
        value = options.get(p[key], p[key])
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        return "Unknown"

    def number(key: str) -> str:
        """
        Format a numeric node setting compactly.
        """
        return f"{p[key]:g}"

    suffix = ""
    if stack == "hardware":
        if kind == "device_control":
            suffix = f"{p['control'] or 'Choose control'} · {p['value']}"
        elif kind == "camera_colors":
            suffix = option("palette")
        elif kind == "detail":
            suffix = f"On · {number('amount')}" if p["enabled"] else "Off"
            if p["enabled"] and p["fixed"]:
                suffix += " · Fixed"
        elif kind == "noise":
            suffix = option("noise_mode")
        else:
            suffix = option("value")
    elif kind == "custom":
        suffix = p["name"] or "Choose module folder"
    elif kind == "source":
        suffix = option("source")
    elif kind == "colors":
        suffix = option("palette")
    elif kind == "filter":
        suffix = option("filter")
        if p["filter"] == "sharpen":
            suffix += f" · {number('amount')}"
        elif p["filter"] == "morphology":
            suffix += f" · {option('morph_operation')}"
        elif p["filter"] == "threshold":
            suffix += f" · {option('threshold_type')}"
        if p["kernel"] and p["filter"] != "none":
            suffix += f" · {int(p['kernel'])}×{int(p['kernel'])}"
    elif kind in ("brightness", "contrast", "gamma", "antialiasing"):
        suffix = number("amount")
        if kind == "brightness":
            suffix = f"{p['amount']:+g}%"
        elif kind == "contrast":
            suffix += "×"
    elif kind == "mirror":
        suffix = (
            " + ".join(
                name
                for key, name in (("horizontal", "Horizontal"), ("vertical", "Vertical"))
                if p[key]
            )
            or "Off"
        )
    elif kind == "range":
        low, high = float(p["low"]), float(p["high"])
        if temperature_unit == "F":
            low, high = low * 1.8 + 32, high * 1.8 + 32
        suffix = f"{low:.1f}–{high:.1f} °{temperature_unit}"
    elif kind == "interpolation":
        suffix = f"{option('method')} · {option('scale')}"
    elif kind == "coreml_acnet":
        suffix = f"{option('compute')} · 2× · {option('denoise')} denoise"
    elif kind == "onnx_superresolution":
        suffix = f"{option('model')} · {option('input')}"
    elif kind == "onnx_style":
        suffix = f"{option('model')} · {p['amount'] * 100:g}% blend"
    elif kind == "onnx_denoise":
        suffix = option("model")
        if p["model"] == "ffdnet-gray":
            suffix += f" · sigma {p['noise']:g}"
    elif kind == "enhance":
        suffix = option("model")
        if p["model"] != "off":
            if p["model"] in ("acnet", "anime4k09"):
                suffix += f" · {int(p['passes'])} pass{'es' if p['passes'] != 1 else ''}"
            elif p["model"] == "ffdnet-gray":
                suffix += f" · sigma {p['noise']:g}"
            elif p["model"] != "dncnn-25":
                suffix += f" · {option('input')}"
            if p["model"] == "acnet" and p["denoise"]:
                suffix += f" · {option('denoise')} denoise"
    elif kind == "combine":
        suffix = f"{option('tab')} · {option('mode')} · {p['opacity'] * 100:g}%"
        if p["mask_source"] != "none":
            suffix += f" · Mask: {option('mask_source')}"
    elif kind in ("edges", "contours"):
        detector = (
            option("method")
            if kind == "edges"
            else {
                "any": "Any shape",
                "rectangle": "Rectangles",
                "circle": "Round",
                "convex": "Convex",
            }[p["shape"]]
        )
        output = {
            "overlay": "Overlay" if kind == "edges" else "Outlines",
            "fill": "Fill",
            "mask": "Mask",
            "cutout": "Isolate",
            "mean": "Average fill",
        }[p["output"]]
        suffix = f"{detector} · {output} · ≤{int(p['max_count'])}"
    return title + (f": {suffix}" if suffix else "")
