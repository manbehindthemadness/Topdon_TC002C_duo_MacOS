"""Versioned, portable camera pipelines and legacy preference migration.

Hardware order is presentation only. Software order is execution order.
Temperature parameters are always stored in Celsius.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from .hardware_controls import HARDWARE_CONTROLS
from .view_settings import COLOR_PALETTES, IMAGE_FILTERS, VIEW_DEFAULTS


@dataclass(frozen=True)
class Parameter:
    title: str
    default: object
    minimum: float = 0
    maximum: float = 1
    step: float = 1
    options: tuple = ()
    unit: str = ""

    def validate(self, value):
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


def choice(title, default, options):
    return Parameter(title, default, options=tuple(options))


def switch(title, default=False):
    return choice(title, default, ((False, "Off"), (True, "On")))


def hardware_parameter(name):
    spec = HARDWARE_CONTROLS[name]
    return Parameter(
        spec.title, spec.minimum, spec.minimum, spec.maximum, spec.step, spec.options, spec.unit
    )


# Camera-style palettes are display approximations, never claimed to be the SDK LUTs.
CAMERA_GRADIENTS = {
    f"camera_{value}": f"{label} (camera-style approximation)"
    for value, label in HARDWARE_CONTROLS["palette"].options
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
    "filter": (
        "Image filter",
        {
            "filter": choice("Filter", "sharpen", tuple(IMAGE_FILTERS.items())),
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
    "enhance": (
        "AI enhancement",
        {
            "model": choice(
                "Upsampler",
                "anime4k09",
                (("off", "Off"), ("anime4k09", "Anime4K09"), ("acnet", "ACNet")),
            ),
            "denoise": choice(
                "ACNet denoising", 0, ((0, "None"), (1, "Light"), (2, "Medium"), (3, "Strong"))
            ),
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
        },
    ),
}
CATALOG = {"hardware": HARDWARE_NODES, "software": SOFTWARE_NODES}


def node(stack, kind, **params):
    definitions = CATALOG[stack][kind][1]
    return {
        "id": uuid4().hex,
        "type": kind,
        "params": {key: params.get(key, spec.default) for key, spec in definitions.items()},
        "bypass": False,
        "expanded": False,
    }


def default_pipeline():
    # The legacy unconfigured preview uses Inferno and antialiased display scaling.
    return {
        "version": 3,
        "hardware": [],
        "software": [
            node("software", "source"),
            node("software", "colors"),
            node("software", "antialiasing"),
            node("software", "output"),
        ],
        "branches": {t: [node("software", "source")] for t in "BCD"},
    }


def validate_pipeline(document):
    if (
        not isinstance(document, dict)
        or set(document)
        != (
            {"version", "hardware", "software", "branches"}
            if document.get("version") == 3
            else {"version", "hardware", "software"}
        )
        or type(document["version"]) is not int
        or document["version"] not in (1, 2, 3)
    ):
        raise ValueError("Unsupported pipeline document/version")
    document = deepcopy(document)
    if document["version"] == 1:
        hardware, software = document["hardware"], document["software"]
        if (
            not isinstance(hardware, list)
            or not hardware
            or not isinstance(hardware[0], dict)
            or hardware[0].get("type") != "source"
            or not isinstance(software, list)
        ):
            raise ValueError("Image source must be first in the legacy camera stack")
        document["software"] = [hardware[0], *software]
        document["hardware"] = hardware[1:]
        document["version"] = 2
    if document["version"] == 2:
        if not isinstance(document["software"], list):
            raise ValueError("Invalid software stack")
        document["software"].append(node("software", "output"))
        document["branches"] = {t: [node("software", "source")] for t in "BCD"}
        document["version"] = 3
    if not isinstance(document["branches"], dict) or set(document["branches"]) != set("BCD"):
        raise ValueError("Pipelines must have tabs A, B, C and D")
    ids = set()
    stacks = [("hardware", document["hardware"], HARDWARE_NODES)] + [
        ("software", nodes, SOFTWARE_NODES) for nodes in software_tabs(document).values()
    ]
    for stack, nodes, catalog in stacks:
        if stack == "hardware":
            # Read older pipelines, but transmission is now a standalone control.
            catalog = {
                **catalog,
                "transmission": (
                    "Optical transmission",
                    {"value": hardware_parameter("transmission")},
                ),
            }
        if not isinstance(nodes, list) or len(nodes) > 128:
            raise ValueError("Invalid pipeline stack")
        kinds = set()
        for item in nodes:
            if not isinstance(item, dict) or set(item) != {
                "id",
                "type",
                "params",
                "bypass",
                "expanded",
            }:
                raise ValueError("Invalid pipeline node")
            if (
                not isinstance(item["id"], str)
                or not 1 <= len(item["id"]) <= 80
                or item["id"] in ids
            ):
                raise ValueError("Duplicate/invalid node identity")
            ids.add(item["id"])
            kind = item["type"]
            if not isinstance(kind, str) or kind not in catalog:
                raise ValueError("Unknown pipeline node")
            if (stack == "hardware" or kind in ("source", "output")) and kind in kinds:
                raise ValueError("Camera controls may only be included once")
            kinds.add(kind)
            if type(item["bypass"]) is not bool or type(item["expanded"]) is not bool:
                raise ValueError("Invalid node state")
            definitions = catalog[kind][1]
            if not isinstance(item["params"], dict) or set(item["params"]) != set(definitions):
                raise ValueError("Unknown/missing node parameter")
            for key, spec in definitions.items():
                spec.validate(item["params"][key])
            if stack == "hardware":
                fields = (
                    {"value": kind}
                    if kind in ("brightness", "contrast", "humidity", "transmission")
                    else {"palette": "palette"}
                    if kind == "camera_colors"
                    else {"amount": "detail"}
                    if kind == "detail"
                    else {key: key for key in definitions}
                    if kind == "noise"
                    else {}
                )
                for parameter, field in fields.items():
                    spec = HARDWARE_CONTROLS[field]
                    spec.apply(bytearray(80), item["params"][parameter])
            if (
                stack == "software"
                and kind == "range"
                and item["params"]["low"] >= item["params"]["high"]
            ):
                raise ValueError("From temperature must be below To temperature")
            if kind == "combine" and item["params"]["raw_low"] >= item["params"]["raw_high"]:
                raise ValueError("Raw input / mask From must be below To")
    hardware = document["hardware"]
    for tab, software in software_tabs(document).items():
        if not software or software[0]["type"] != "source" or software[0]["bypass"]:
            raise ValueError(f"Tab {tab}: Image source must be first and enabled")
        outputs = [n for n in software if n["type"] == "output"]
        if tab == "A":
            if len(outputs) != 1 or software[-1]["type"] != "output" or outputs[0]["bypass"]:
                raise ValueError("Tab A must end with an enabled Output node")
        elif outputs:
            raise ValueError("Only tab A can contain the viewer Output node")
    execution_dependencies(document, all_tabs=True)  # Also reject cycles in disconnected tabs.
    enabled = {n["type"]: n["params"] for n in hardware if not n["bypass"]}
    detail = enabled.get("detail", {})
    if detail.get("fixed") and (
        not detail.get("enabled")
        or enabled.get("preset", {}).get("value", "balanced") != "balanced"
        or enabled.get("gamma", {}).get("value", 50) != 50
        or enabled.get("boost", {}).get("value", 0) != 0
    ):
        raise ValueError(
            "Fixed detail requires detail enhancement, Balanced, gamma 50 and boost Off"
        )
    result = deepcopy(document)
    result["hardware"] = [item for item in result["hardware"] if item["type"] != "transmission"]
    return result


def legacy_transmission_value(document):
    """Preserve an active old node on preference reload, not on pipeline import."""
    validate_pipeline(document)
    return next(
        (
            item["params"]["value"]
            for item in document["hardware"]
            if item["type"] == "transmission" and not item["bypass"]
        ),
        None,
    )


def software_tabs(document):
    return {"A": document["software"], **document.get("branches", {})}


def execution_dependencies(document, all_tabs=False, roots=("A",)):
    """Reachable DAG, with bypassed combines creating no connection."""
    tabs = software_tabs(document)
    dependencies = {}
    visiting = set()

    def visit(tab):
        if tab in visiting:
            raise ValueError("Combine connections cannot form a cycle")
        if tab in dependencies:
            return
        if tab not in tabs:
            raise ValueError("Unknown input tab")
        visiting.add(tab)
        targets = {
            target
            for n in tabs[tab]
            if n["type"] == "combine" and not n["bypass"]
            for target in (n["params"]["tab"], n["params"]["mask_source"])
            if target in tabs
        }
        for target in sorted(targets):
            visit(target)
        visiting.remove(tab)
        dependencies[tab] = targets

    for tab in tabs if all_tabs else roots:
        visit(tab)
    return dependencies


def preview_roots(document):
    return tuple(
        tab
        for tab, nodes in software_tabs(document).items()
        if any(n["type"] == "preview" and n["expanded"] and not n["bypass"] for n in nodes)
    )


def collapse_previews(document):
    """Opening Camera always starts its read-only preview taps off."""
    for nodes in software_tabs(document).values():
        for item in nodes:
            if item["type"] == "preview":
                item["expanded"] = False


def preview_required(document):
    tabs = software_tabs(document)
    active = execution_dependencies(document)
    return any(
        not thermal_source({"software": tabs[t]})
        or any(
            n["type"] == "combine"
            and not n["bypass"]
            and (n["params"]["tab"] == "preview" or n["params"]["mask_source"] == "preview")
            for n in tabs[t]
        )
        for t in active
    )


def active_nodes(document, stack):
    return [n for n in document[stack] if not n["bypass"]]


def geometry(document, rotation=0):
    horizontal = vertical = False
    for item in active_nodes(document, "software"):
        if item["type"] == "mirror":
            horizontal ^= item["params"]["horizontal"]
            vertical ^= item["params"]["vertical"]
    # Legacy measurement coordinates mirror AFTER rotation; swap axes at 90/270.
    return (vertical, horizontal) if rotation in (90, 270) else (horizontal, vertical)


def thermal_source(document):
    return document["software"][0]["params"]["source"] == "raw" or any(
        n["type"] == "range" for n in active_nodes(document, "software")
    )


def migrate_pipeline(saved):
    document = default_pipeline()
    display = {**VIEW_DEFAULTS, **saved.get("display", {})}
    document["software"][0]["params"]["source"] = display["image_source"]
    hardware = saved.get("hardware", {})
    if not display["analyze_mode"] and display["palette_source"] == "app" and "palette" in hardware:
        document["software"][0]["params"]["source"] = "raw"
    for kind, key in (
        ("brightness", "brightness"),
        ("contrast", "contrast"),
        ("humidity", "humidity"),
    ):
        if key in hardware:
            document["hardware"].append(node("hardware", kind, value=hardware[key]))
    if "palette" in hardware:
        document["hardware"].append(node("hardware", "camera_colors", palette=hardware["palette"]))
    if any(key in hardware for key in ("detail", "detail_enabled")) or saved.get("fixed_range"):
        document["hardware"].append(
            node(
                "hardware",
                "detail",
                enabled=bool(hardware.get("detail_enabled", 1)),
                amount=hardware.get("detail", 0),
                fixed=saved.get("fixed_range", False),
            )
        )
    noise = {
        key: hardware[key]
        for key in ("noise_mode", "noise_general", "noise_spatial", "noise_temporal")
        if key in hardware
    }
    if noise:
        document["hardware"].append(node("hardware", "noise", **noise))
    for kind, key, neutral in (
        ("preset", "processing_preset", "balanced"),
        ("gamma", "camera_gamma", 50),
        ("boost", "camera_boost", 0),
    ):
        if saved.get(key, neutral) != neutral:
            document["hardware"].append(node("hardware", kind, value=saved[key]))
    software = []
    if display["analyze_mode"]:
        software.append(
            node(
                "software",
                "range",
                low=display["raw_temperature_low"],
                high=display["raw_temperature_high"],
            )
        )
        if display["raw_upsampling"] == "bicubic":
            software.append(node("software", "interpolation"))
        if display["raw_sharpen_amount"]:
            software.append(
                node("software", "filter", filter="sharpen", amount=display["raw_sharpen_amount"])
            )
        model = display["raw_upsampling"]
        passes = display["raw_anime4k_passes"]
        palette = display["raw_palette"]
    else:
        if display["image_filter"] != "none":
            software.append(node("software", "filter", filter=display["image_filter"]))
        model = display["upsampling"]
        passes = display["anime4k_passes"]
        palette = (
            display["color_palette"]
            if display["palette_source"] == "app"
            or display["image_source"] == "raw"
            or "palette" not in hardware
            else None
        )

    # Match legacy learned enhancement before color; Anime04 enhancement after color.
    def enhancement():
        return node(
            "software",
            "enhance",
            model="acnet" if model.startswith("acnet") else model,
            denoise=int(model[-1]) if model.startswith("acnet") else 0,
            amount=display["enhancement_amount"],
            passes=passes,
            input="current" if display["analyze_mode"] else display["enhancement_input"],
        )

    if model.startswith("acnet") or (display["analyze_mode"] and model == "anime4k09"):
        software.append(enhancement())
    if palette:
        software.append(node("software", "colors", palette=palette))
    if model == "anime4k09" and not display["analyze_mode"]:
        software.append(enhancement())
    # Convert legacy after-rotation mirrors into pipeline sensor-axis mirrors.
    horizontal, vertical = display["mirror_horizontal"], display["mirror_vertical"]
    if saved.get("rotation", 0) in (90, 270):
        horizontal, vertical = vertical, horizontal
    if horizontal or vertical:
        software.append(node("software", "mirror", horizontal=horizontal, vertical=vertical))
    if display["antialiasing"]:
        software.append(node("software", "antialiasing"))
    document["software"] = [document["software"][0], *software, document["software"][-1]]
    return validate_pipeline(document)
