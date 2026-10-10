"""
Portable pipeline validation, migration, geometry, and dependencies.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from ..custom_nodes.bundle import validate_custom
from ..feature_processing import validate_feature
from ..hardware_controls import HARDWARE_CONTROLS
from ..image_filters import validate_filter
from ..view_settings import VIEW_DEFAULTS
from .catalog import (
    CATALOG,
    HARDWARE_NODES,
    SOFTWARE_NODES,
    hardware_parameter,
)


def node(stack: Any, kind: Any, **params: Any) -> Any:
    """
    Node.
    """
    definitions = CATALOG[stack][kind][1]
    if stack == "software" and kind == "enhance" and params.get("model") == "acnet":
        # Each ACNet pass doubles both dimensions; Anime4K passes instead
        # refine a single 2x image. Keep explicit saved pass counts intact.
        params.setdefault("passes", 1)
    if stack == "software" and kind == "filter" and "amount" in params:
        if params.get("filter") == "gaussian" and "sigma" not in params:
            params["sigma"] = max(0.1, params["amount"])
            if not params["amount"]:
                params.setdefault("mix", 0.0)
        elif params.get("filter") == "bilateral" and "sigma_color" not in params:
            params["sigma_color"] = 30 * params["amount"] if params["amount"] else 1.0
    return {
        "id": uuid4().hex,
        "type": kind,
        "params": {key: params.get(key, spec.default) for key, spec in definitions.items()},
        "bypass": False,
        "expanded": False,
    }


def default_pipeline() -> Any:
    # The legacy unconfigured preview uses Inferno and antialiased display scaling.
    """
    Default pipeline.
    """
    return {
        "version": 4,
        "hardware": [],
        "software": [
            node("software", "source"),
            node("software", "colors"),
            node("software", "antialiasing"),
            node("software", "output"),
        ],
        "branches": {t: [node("software", "source")] for t in "BCD"},
    }


def validate_pipeline(document: Any) -> Any:
    """
    Validate pipeline.
    """
    if (
        not isinstance(document, dict)
        or set(document)
        != (
            {"version", "hardware", "software", "branches"}
            if document.get("version") in (3, 4)
            else {"version", "hardware", "software"}
        )
        or type(document["version"]) is not int
        or document["version"] not in (1, 2, 3, 4)
    ):
        raise ValueError("Unsupported pipeline document/version")
    document = deepcopy(document)
    legacy_filters = document["version"] < 4
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
    if document["version"] == 3:
        document["version"] = 4
    if not isinstance(document["branches"], dict) or set(document["branches"]) != set("BCD"):
        raise ValueError("Pipelines must have tabs A, B, C and D")
    ids = set()
    stacks = [("hardware", document["hardware"], HARDWARE_NODES)] + [
        ("software", nodes, SOFTWARE_NODES) for nodes in software_tabs(document).values()
    ]
    for stack, nodes, catalog in stacks:
        if stack == "hardware":
            # Read older pipelines; these fields now belong to permanent calibration controls.
            catalog = {
                **catalog,
                **{
                    name: (HARDWARE_CONTROLS[name].title, {"value": hardware_parameter(name)})
                    for name in ("transmission", "humidity")
                },
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
            if stack == "software" and kind == "custom" and isinstance(item["params"], dict):
                item["params"].setdefault("controls", "[]")
            if stack == "software" and kind == "enhance" and isinstance(item["params"], dict):
                # Read existing version-4 presets without changing model/pass choices.
                # Hardware-specific preferences remain serialized on every platform.
                for key in ("backend", "apple_compute", "noise"):
                    item["params"].setdefault(key, definitions[key].default)
            if (
                legacy_filters
                and stack == "software"
                and kind == "filter"
                and isinstance(item["params"], dict)
                and set(item["params"]) == {"filter", "amount"}
            ):
                old = item["params"]
                item["params"] = node("software", "filter", **old)["params"]
            if stack == "software" and kind == "colors" and isinstance(item["params"], dict):
                # Preserve old saved/imported white/black-hot choices after deduplicating the menu.
                aliases = {"camera_1": "white_hot", "camera_2": "black_hot"}
                palette = item["params"].get("palette")
                if isinstance(palette, str) and palette in aliases:
                    item["params"]["palette"] = aliases[palette]
            if not isinstance(item["params"], dict) or set(item["params"]) != set(definitions):
                raise ValueError("Unknown/missing node parameter")
            for key, spec in definitions.items():
                spec.validate(item["params"][key])
            if stack == "software" and kind == "custom":
                validate_custom(item["params"])
            if stack == "software" and kind == "filter":
                validate_filter(item["params"])
            if stack == "software" and kind in ("edges", "contours"):
                validate_feature(kind, item["params"])
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
    result["hardware"] = [
        item for item in result["hardware"] if item["type"] not in ("transmission", "humidity")
    ]
    return result


def legacy_transmission_value(document: Any) -> Any:
    """
    Preserve an active old node on preference reload, not on pipeline import.
    """
    return legacy_calibration_values(document).get("transmission")


def legacy_calibration_values(document: Any) -> dict[str, int | float]:
    """
    Extract validated active correction nodes for migration of local preferences only.
    """
    validate_pipeline(document)
    values = {
        item["type"]: item["params"]["value"] for item in document["hardware"]
        if item["type"] in ("transmission", "humidity") and not item["bypass"]
    }
    return values


def software_tabs(document: Any) -> Any:
    """
    Software tabs.
    """
    return {"A": document["software"], **document.get("branches", {})}


def execution_dependencies(document: Any, all_tabs: Any = False, roots: Any = ("A",)) -> Any:
    """
    Reachable DAG, with bypassed combines creating no connection.
    """
    tabs = software_tabs(document)
    dependencies = {}
    visiting = set()

    def visit(tab: Any) -> None:
        """
        Visit.
        """
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

    for root_tab in tabs if all_tabs else roots:
        visit(root_tab)
    return dependencies


def preview_roots(document: Any) -> Any:
    """
    Preview roots.
    """
    return tuple(
        tab
        for tab, nodes in software_tabs(document).items()
        if any(n["type"] == "preview" and n["expanded"] and not n["bypass"] for n in nodes)
    )


def collapse_previews(document: Any) -> None:
    """
    Opening Camera always starts its read-only preview taps off.
    """
    for nodes in software_tabs(document).values():
        for item in nodes:
            if item["type"] == "preview":
                item["expanded"] = False


def preview_required(document: Any) -> Any:
    """
    Preview required.
    """
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


def active_nodes(document: Any, stack: Any) -> Any:
    """
    Active nodes.
    """
    return [n for n in document[stack] if not n["bypass"]]


def geometry(document: Any, rotation: Any = 0) -> Any:
    """
    Geometry.
    """
    horizontal = vertical = False
    for item in active_nodes(document, "software"):
        if item["type"] == "mirror":
            horizontal ^= item["params"]["horizontal"]
            vertical ^= item["params"]["vertical"]
    # Legacy measurement coordinates mirror AFTER rotation; swap axes at 90/270.
    return (vertical, horizontal) if rotation in (90, 270) else (horizontal, vertical)


def thermal_source(document: Any) -> Any:
    """
    Thermal source.
    """
    return document["software"][0]["params"]["source"] == "raw" or any(
        n["type"] == "range" for n in active_nodes(document, "software")
    )


def migrate_pipeline(saved: Any) -> Any:
    """
    Migrate pipeline.
    """
    document = default_pipeline()
    display = {**VIEW_DEFAULTS, **saved.get("display", {})}
    document["software"][0]["params"]["source"] = display["image_source"]
    hardware = saved.get("hardware", {})
    if not display["analyze_mode"] and display["palette_source"] == "app" and "palette" in hardware:
        document["software"][0]["params"]["source"] = "raw"
    for kind, key in (
        ("brightness", "brightness"),
        ("contrast", "contrast"),
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
            software.append(node("software", "filter", filter=display["image_filter"], amount=0.7))
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
    def enhancement() -> Any:
        """
        Enhancement.
        """
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
