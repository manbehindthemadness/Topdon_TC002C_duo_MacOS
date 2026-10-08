"""Concise node descriptions derived from settings, without changing node identities."""

from .pipeline import CATALOG


def node_title(stack, item, temperature_unit="C"):
    kind, p = item["type"], item["params"]
    title, definitions = CATALOG[stack][kind]

    def option(key):
        return str(dict(definitions[key].options).get(p[key], p[key]))

    def number(key):
        return f"{p[key]:g}"

    suffix = ""
    if stack == "hardware":
        if kind == "camera_colors":
            suffix = option("palette")
        elif kind == "detail":
            suffix = f"On · {number('amount')}" if p["enabled"] else "Off"
            if p["enabled"] and p["fixed"]:
                suffix += " · Fixed"
        elif kind == "noise":
            suffix = option("noise_mode")
        else:
            suffix = option("value")
            if kind == "humidity":
                suffix += "%"
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
        low, high = p["low"], p["high"]
        if temperature_unit == "F":
            low, high = low * 1.8 + 32, high * 1.8 + 32
        suffix = f"{low:.1f}–{high:.1f} °{temperature_unit}"
    elif kind == "interpolation":
        suffix = f"{option('method')} · {option('scale')}"
    elif kind == "coreml_acnet":
        suffix = f"{option('compute')} · 2× · {option('denoise')} denoise"
    elif kind == "onnx_superresolution":
        suffix = f"{option('model')} · {option('input')}"
        if p["backend"] == "coreml":
            suffix += " · Apple preferred (CPU fallback)"
    elif kind == "onnx_denoise":
        suffix = option("model")
        if p["model"] == "ffdnet-gray":
            suffix += f" · sigma {p['noise']:g}"
        if p["backend"] == "coreml":
            suffix += " · Apple preferred (CPU fallback)"
    elif kind == "enhance":
        suffix = option("model")
        if p["model"] != "off":
            suffix += f" · {int(p['passes'])} pass{'es' if p['passes'] != 1 else ''}"
            if p["model"] == "acnet" and p["denoise"]:
                suffix += f" · {option('denoise')} denoise"
            if p["model"] == "acnet" and p["backend"] == "coreml":
                suffix += " · Apple preferred (CPU fallback)"
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
