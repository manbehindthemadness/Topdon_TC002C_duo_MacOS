"""
Capability-driven hardware choices without probing devices in the Qt process.
"""

from typing import Any

from PySide6.QtWidgets import QVBoxLayout, QWidget


def node_capability(item: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve per-node support while retaining compatibility with older viewer messages.
    """
    capabilities = state.get("camera_capabilities")
    if not isinstance(capabilities, dict):
        return {"supported": True, "available": bool(state.get("processing_preset_available")),
                "reason": "Camera settings are not ready"}
    if item["type"] == "device_control":
        if capabilities.get("features", {}).get("duo_nodes"):
            return {"supported": False, "available": False,
                    "reason": "Use the Duo's existing hardware nodes"}
        spec = capabilities.get("controls", {}).get(item["params"]["control"])
        if not isinstance(spec, dict) or spec.get("effect") != "preview" or (
            spec.get("scope", "device") != "device"
        ):
            return {"supported": False, "available": False,
                    "reason": "Choose a supported display control"}
        return spec
    return capabilities.get("nodes", {}).get(item["type"], {
        "supported": False, "available": False, "reason": "Not supported by this camera",
    })


def row_options(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Translate presentation metadata into the shared wheel-safe control widget.
    """
    return {"minimum": spec.get("minimum", 0), "maximum": spec.get("maximum", 100),
            "step": spec.get("step", 1), "unit": spec.get("unit", ""),
            "options": ((False, "Off"), (True, "On")) if spec.get("kind") == "boolean"
            else tuple(tuple(option) for option in spec.get("options", ()))}


class HardwareControlFields(QWidget):
    """
    Edit arbitrary backend-declared display controls while preserving unavailable selections.
    """

    def __init__(self, editor: Any, item: dict[str, Any]) -> None:
        """
        Create fields using metadata already published by the viewer.
        """
        super().__init__()
        self.editor, self.item = editor, item
        self.layout_rows = QVBoxLayout(self)
        self.signature = None
        self.rows: dict[str, Any] = {}
        self.update_state(getattr(editor, "last_state", {}), editor.locked)

    def select(self, name: str) -> None:
        """
        Initialize a new selection from its own default without rewriting saved selections.
        """
        if self.editor.locked:
            return
        spec = self.editor.last_state.get("camera_capabilities", {}).get("controls", {}).get(name)
        if not isinstance(spec, dict):
            return
        self.item["params"].update(control=name, value=spec.get("default", spec.get("minimum", 0)))
        self.editor.publish()
        self.update_state(self.editor.last_state, self.editor.locked)

    def update_state(self, state: dict[str, Any], locked: bool) -> None:
        """
        Rebuild only when metadata changes, retaining saved values and respecting locks.
        """
        specs = state.get("camera_capabilities", {}).get("controls", {})
        choices = [(name, spec["title"]) for name, spec in specs.items()
                   if spec.get("supported") and spec.get("effect") == "preview"
                   and spec.get("scope", "device") == "device"]
        name = self.item["params"]["control"]
        if name not in dict(choices):
            choices.insert(0, (name, name or "Choose control"))
        spec = specs.get(name, {})
        signature = choices, spec
        if signature != self.signature:
            while self.layout_rows.count():
                entry = self.layout_rows.takeAt(0)
                if entry is None:
                    break
                widget = entry.widget()
                if widget is not None:
                    if hasattr(widget, "timer"):
                        widget.timer.stop()
                    widget.deleteLater()
            control = self.editor.row_class("Control", self.select, options=choices)
            self.layout_rows.addWidget(control)
            self.rows = {"control": control}
            if spec:
                value = self.editor.row_class(
                    spec["title"], lambda selected: self.editor.change(self.item, "value", selected),
                    **row_options(spec),
                )
                self.layout_rows.addWidget(value)
                self.rows["value"] = value
            self.signature = signature
        self.rows["control"].update_state(name, not locked and not self.item["bypass"])
        if "value" in self.rows:
            self.rows["value"].set_display_unit(state.get("temperature_unit", "C"))
            self.rows["value"].update_state(
                self.item["params"]["value"], not locked and not self.item["bypass"]
                and bool(spec.get("available")),
            )
