"""
Present runtime availability and values without changing saved selections.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox

if TYPE_CHECKING:
    from .editor import PipelineEditor

from ..enhancement_limits import enhancement_pass_limits
from ..feature_processing import feature_fields
from ..image_filters import allowed_kernels, filter_fields
from ..pipeline import preview_required
from ..pipeline_hardware import desired_hardware
from .compute import backend_status, configure_backend_row, state_backend, update_compute_devices
from .hardware import node_capability
from .presets import pipeline_content


def update_editor_state(self: PipelineEditor, state: Any, locked: Any) -> None:
    """
    Update state.
    """
    self.last_state = state
    download_status = state.get("model_download_status", "")
    self.download_status.setText(download_status)
    self.download_status.setVisible(bool(download_status))
    capability = state.get("apple_acceleration", {})
    self.apple_status.setText(
        "Apple acceleration ready · Metal + Core ML"
        if capability.get("available")
        else capability.get("reason", "")
    )
    nvidia = state.get("nvidia_acceleration", {})
    if nvidia.get("available") or (
        nvidia
        and not capability.get("available")
        and nvidia.get("reason") != "NVIDIA acceleration requires Linux"
    ):
        self.apple_status.setText(nvidia.get("reason", ""))
    self.apple_status.setVisible(bool(capability or nvidia))
    self.hardware_state = state.get("hardware", {})
    incoming = state.get("pipeline")
    if incoming is not None and state.get("pipeline_serial", 0) >= self.edit_serial:
        incoming = self.validate_document(incoming)
        self.edit_serial = max(self.edit_serial, state.get("pipeline_serial", 0))
        if incoming != self.document:
            if pipeline_content(incoming) != pipeline_content(self.document):
                self.current_preset_name = None
                self.recognize_preset = True
            old_structure = [
                (n["id"], n["expanded"])
                for stack in ("hardware", "software")
                for n in self.nodes(stack)
            ]
            new_structure = [
                (n["id"], n["expanded"])
                for stack in ("hardware", "software")
                for n in self.nodes(stack, incoming)
            ]
            if old_structure != new_structure:
                self.document = incoming
                self.rebuild()
            else:
                for stack in ("hardware", "software"):
                    for old, new in zip(self.nodes(stack), self.nodes(stack, incoming)):
                        old.update(new)
                if self.tab != "A":
                    self.document["software"] = incoming["software"]
                for tab in "BCD":
                    if tab != self.tab:
                        self.document["branches"][tab] = incoming["branches"][tab]
            self.accepted_document = deepcopy(self.document)
    self.last_desired = desired_hardware(self.document)
    self.update_tab_status()
    self.update_titles()
    self.locked = locked
    for custom in self.custom_widgets.values():
        custom.refresh()
    pass_limits = enhancement_pass_limits(self.document, clamp=True)
    self.update_preset_selection()
    current = state.get("pipeline_serial", 0) >= self.edit_serial
    for item in self.nodes("software"):
        if item["type"] == "preview":
            payload = state.get("pipeline_previews", {}).get(item["id"]) if current else None
            thumbnail = self.preview_widgets[item["id"]]
            if item["bypass"] and not thumbnail.image.isNull():
                continue  # A bypassed preview keeps its last image and viewport.
            thumbnail.show_image(
                payload if item["expanded"] and not item["bypass"] else None,
                "Preview bypassed"
                if item["bypass"]
                else f"Preview failed: {state['pipeline_preview_errors'][item['id']]}"
                if state.get("pipeline_preview_errors", {}).get(item["id"])
                else "Waiting for pipeline image…",
            )
            elapsed = state.get("pipeline_preview_timings", {}).get(item["id"]) if current else None
            thumbnail.elapsed_ms = (
                float(elapsed)
                if not thumbnail.image.isNull()
                and isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                and math.isfinite(elapsed)
                and elapsed >= 0
                else None
            )
    thermal = not preview_required(self.document)
    for stack in ("hardware", "software"):
        for item in self.nodes(stack):
            _, controls, bypass, title, badge = self.widgets[item["id"]]
            inactive = stack == "hardware" and thermal
            capability = node_capability(item, state) if stack == "hardware" else {}
            unavailable = stack == "hardware" and not capability.get("available", False)
            badge.setText(
                capability.get("reason") or "Camera setting unavailable; value retained"
                if unavailable
                else "Inactive for thermal rendering; settings retained"
                if inactive
                else "Bypassed"
                if item["bypass"]
                else backend_status(item, state)
                if "backend" in item["params"]
                else state.get("apple_acceleration", {}).get("reason", "")
                if item["type"] == "coreml_acnet"
                and not state.get("apple_acceleration", {}).get("available", False)
                else ""
            )
            badge.setVisible(bool(badge.text()))
            bypass.blockSignals(True)
            bypass.setChecked(item["bypass"])
            bypass.blockSignals(False)
            bypass.setEnabled(not locked)
            title.setEnabled(not locked)
            if item["id"] in self.hardware_widgets:
                self.hardware_widgets[item["id"]].update_state(state, locked or inactive)
            if item["type"] == "preview":
                elapsed = self.preview_widgets[item["id"]].elapsed_ms
                self.preview_timing_widgets[item["id"]].setText(
                    f"{elapsed:.1f} ms" if elapsed is not None else "— ms"
                )
            for key, row in controls.items():
                row.set_display_unit(state.get("temperature_unit", "C"))
                available = not locked and not inactive and not unavailable and not item["bypass"]
                if item["type"] == "source" and key == "source" and "camera_profile" in state:
                    profile = state["camera_profile"]
                    combo = row.input
                    assert isinstance(combo, QComboBox)
                    model = combo.model()
                    assert isinstance(model, QStandardItemModel)
                    for index in range(combo.count()):
                        source = combo.itemData(index)
                        supported = profile.get("radiometry") if source == "raw" else profile.get("preview")
                        entry = model.item(index)
                        if entry is not None:
                            entry.setEnabled(bool(supported))
                if item["type"] == "filter":
                    relevant = key in filter_fields(item["params"])
                    row.setVisible(relevant)
                    available &= relevant
                    if key == "kernel":
                        allowed = allowed_kernels(item["params"]["filter"])
                        combo = row.input
                        assert isinstance(combo, QComboBox)
                        model = combo.model()
                        assert isinstance(model, QStandardItemModel)
                        for index in range(combo.count()):
                            entry = model.item(index)
                            if entry is not None:
                                entry.setEnabled(combo.itemData(index) in allowed)
                if stack == "hardware" and item["type"] == "detail" and key == "fixed":
                    features = state.get("camera_capabilities", {}).get("features", {})
                    if "camera_capabilities" in state:
                        available &= bool(features.get("fixed_detail"))
                if key in ("backend", "apple_compute"):
                    available = configure_backend_row(row, item, key, state, available)
                if item["type"] == "onnx_denoise" and key == "noise":
                    relevant = item["params"]["model"] == "ffdnet-gray"
                    row.setVisible(relevant)
                    available &= relevant
                if item["type"] in ("edges", "contours"):
                    relevant = key in feature_fields(item["type"], item["params"])
                    row.setVisible(relevant)
                    available &= relevant
                if item["type"] == "combine":
                    mode = item["params"]["mode"]
                    has_mask = item["params"]["mask_source"] != "none"
                    if key in ("mask_kind", "mask_invert"):
                        available &= has_mask
                    if key == "mask_threshold":
                        available &= has_mask and item["params"]["mask_kind"] == "threshold"
                    if key in ("raw_low", "raw_high"):
                        available &= (
                            item["params"]["tab"] == "raw"
                            or item["params"]["mask_source"] == "raw"
                            or item["params"]["mask_source"] == "input"
                            and item["params"]["tab"] == "raw"
                        )
                    available &= (
                        key not in ("base_weight", "input_weight", "offset") or mode == "weighted"
                    )
                    available &= key not in ("threshold", "invert") or mode == "mask"
                if item["type"] == "enhance":
                    model = item["params"]["model"]
                    if key in ("noise", "denoise", "passes", "input"):
                        relevant = (
                            model == "ffdnet-gray"
                            if key == "noise"
                            else model == "acnet"
                            if key == "denoise"
                            else model in ("acnet", "anime4k09")
                            if key == "passes"
                            else model not in ("dncnn-25", "ffdnet-gray")
                        )
                        row.setVisible(relevant)
                        available &= relevant
                    if key == "passes" and model in ("acnet", "anime4k09"):
                        maximum = pass_limits[item["id"]]
                        row.set_numeric_limits(1, max(1, maximum))
                        available &= maximum > 0
                        row.setToolTip(
                            f"Maximum {maximum} ACNet passes for this input (4 megapixel limit)."
                            if item["params"]["model"] == "acnet" and maximum
                            else "No upscale fits; choose Native input or reduce preceding scales."
                            if not maximum
                            else "Anime4K09 allows 1–5 refinement passes with one 2× output."
                        )
                if key == "apple_compute":
                    update_compute_devices(
                        row, item["params"], available, state_backend(item["params"], state)
                    )
                else:
                    value = (
                        state_backend(item["params"], state)
                        if key == "backend"
                        else item["params"][key]
                    )
                    row.update_state(value, available)
            # Child size changes (badges) need a refreshed item height.
    for listing in self.stacks.values():
        listing.setDragEnabled(not locked)
        listing.fit_contents()
    for button in (self.import_button, self.export_button, self.defaults_button):
        button.setEnabled(not locked)
    self.preset_combo.setEnabled(not locked)
