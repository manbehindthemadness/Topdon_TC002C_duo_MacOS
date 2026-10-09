"""
Qt pipeline editing, import/export, presets, and node controls.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..view_window import ControlRow

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from ..dialog_preferences import load_dialog_directory, remember_dialog_directory
from ..enhancement_limits import enhancement_pass_limits
from ..feature_processing import feature_fields
from ..image_filters import allowed_kernels, filter_fields
from ..onnx_models import MODELS as ONNX_MODELS
from ..pipeline import (
    CATALOG,
    default_pipeline,
    execution_dependencies,
    node,
    software_tabs,
    validate_pipeline,
)
from ..pipeline_hardware import desired_hardware
from ..pipeline_presets import load_presets, save_presets
from ..pipeline_titles import node_title
from .state import update_editor_state
from .widgets import PipelinePreview, StackList


class PipelineEditor(QWidget):
    last_state: dict[str, Any]

    def __init__(self, send: Any, row_class: Any) -> None:
        """
        Init.
        """
        super().__init__()
        self.send, self.row_class = send, row_class
        self.document = default_pipeline()
        self.accepted_document = deepcopy(self.document)
        self.tab = "A"
        self.edit_serial = 0
        self.hardware_state = {}
        self.last_desired = desired_hardware(self.document)
        self.locked = False
        self.rebuilding = False
        self.widgets: dict[str, tuple[str, dict[str, ControlRow], Any, Any, Any]] = {}
        self.preview_widgets = {}
        self.preview_timing_widgets = {}
        self.preview_cache = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        instruction = QLabel(
            "Click a node header to expand. Drag its grip to rearrange. Right-click to add, insert or clear nodes."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self.apple_status = QLabel()
        self.apple_status.setWordWrap(True)
        layout.addWidget(self.apple_status)
        self.download_status = QLabel()
        self.download_status.setWordWrap(True)
        self.download_status.hide()
        layout.addWidget(self.download_status)
        self.stacks = {}
        for stack, title in (
            ("hardware", "Camera hardware · order organizes controls"),
            ("software", "Software processing · runs from top to bottom"),
        ):
            if stack == "software":
                self.tab_bar = QTabBar()
                for tab in "ABCD":
                    self.tab_bar.addTab(tab)
                self.tab_bar.currentChanged.connect(self.select_tab)
                layout.addWidget(self.tab_bar)
                self.tab_status = QLabel()
                self.tab_status.setWordWrap(True)
                layout.addWidget(self.tab_status)
            layout.addWidget(QLabel(title))
            listing = StackList(self, stack)
            self.stacks[stack] = listing
            layout.addWidget(listing)
        buttons = QHBoxLayout()
        self.import_button = QPushButton("Import pipeline")
        self.export_button = QPushButton("Export pipeline")
        self.defaults_button = QPushButton("Restore default pipeline")
        self.import_button.clicked.connect(self.import_file)
        self.export_button.clicked.connect(self.export_file)
        self.defaults_button.clicked.connect(self.defaults)
        for button in (self.import_button, self.export_button, self.defaults_button):
            buttons.addWidget(button)
        self.presets = load_presets()
        self.preset_combo = QComboBox()
        self.preset_combo.setAccessibleName("Pipeline presets")
        self.preset_combo.setToolTip("Save your current pipeline or apply a named preset.")
        self.refresh_presets()
        self.preset_combo.activated.connect(self.select_preset)
        buttons.addWidget(self.preset_combo)
        layout.addLayout(buttons)
        self.rebuild()

    def refresh_presets(self) -> None:
        """
        Refresh presets.
        """
        self.preset_combo.clear()
        self.preset_combo.addItem("Pipeline presets…", None)
        self.preset_combo.addItem("Save current pipeline…", ("save", ""))
        if self.presets:
            self.preset_combo.insertSeparator(self.preset_combo.count())
        for name in sorted(self.presets, key=str.casefold):
            self.preset_combo.addItem(name, ("load", name))

    def select_preset(self, index: Any) -> None:
        """
        Select preset.
        """
        choice = self.preset_combo.itemData(index)
        self.preset_combo.setCurrentIndex(0)
        if self.locked or choice is None:
            return
        action, name = choice
        if action == "load":
            self.document = validate_pipeline(self.presets[name])
            self.rebuild()
            self.publish()
            return
        name, accepted = QInputDialog.getText(self, "Save pipeline preset", "Preset name:")
        name = name.strip()
        if not accepted or not name:
            return
        if (
            name in self.presets
            and QMessageBox.question(self, "Replace preset", f'Replace the saved preset "{name}"?')
            != QMessageBox.StandardButton.Yes
        ):
            return
        # Include spinbox edits that are still waiting on their debounce timer.
        for _, controls, *_ in list(self.widgets.values()):
            for row in controls.values():
                if hasattr(row.input, "interpretText"):
                    row.input.interpretText()
                if row.timer.isActive():
                    row.timer.timeout.emit()
        candidate = {**self.presets, name: validate_pipeline(self.document)}
        try:
            save_presets(candidate)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Could not save preset", str(exc))
            return
        self.presets = candidate
        self.refresh_presets()

    def nodes(self, stack: str, document: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Nodes.
        """
        document = self.document if document is None else document
        return (
            document[stack]
            if stack == "hardware" or self.tab == "A"
            else document["branches"][self.tab]
        )

    def set_nodes(self, stack: Any, nodes: Any) -> None:
        """
        Set nodes.
        """
        if stack == "hardware" or self.tab == "A":
            self.document[stack] = nodes
        else:
            self.document["branches"][self.tab] = nodes

    def select_tab(self, index: Any) -> None:
        # Commit debounced inputs before their widgets are destroyed by tab switching.
        """
        Select tab.
        """
        if not self.locked:
            for _, controls, *_ in list(self.widgets.values()):
                for row in controls.values():
                    if hasattr(row.input, "interpretText"):
                        row.input.interpretText()
                    if row.timer.isActive():
                        row.timer.timeout.emit()
        self.tab = "ABCD"[index]
        self.rebuild()
        if hasattr(self, "last_state"):
            self.update_state(self.last_state, self.locked)

    def publish(self) -> None:
        """
        Publish.
        """
        try:
            validate_pipeline(self.document)
        except ValueError as exc:
            QMessageBox.warning(self, "Pipeline rejected", str(exc))
            self.document = deepcopy(self.accepted_document)
            self.rebuild()
            self.send({"action": "pipeline_refresh"})
            return
        enhancement_pass_limits(self.document, clamp=True)
        self.accepted_document = deepcopy(self.document)
        self.update_tab_status()
        self.update_titles()
        self.edit_serial += 1
        desired = desired_hardware(self.document)
        hardware_operation = desired != self.last_desired
        self.last_desired = desired
        self.send(
            {
                "action": "pipeline",
                "document": deepcopy(self.document),
                "serial": self.edit_serial,
                "hardware_operation": hardware_operation,
            }
        )

    def insert(self, stack: Any, kind: Any, position: Any) -> None:
        """
        Insert.
        """
        if self.locked or kind in ("source", "output"):
            return
        if stack == "hardware" and any(n["type"] == kind for n in self.nodes(stack)):
            return
        item = node(stack, kind)
        if stack == "hardware":
            fields = (
                {"value": kind}
                if kind in ("brightness", "contrast", "humidity")
                else {"palette": "palette"}
                if kind == "camera_colors"
                else {"amount": "detail", "enabled": "detail_enabled"}
                if kind == "detail"
                else {key: key for key in item["params"]}
                if kind == "noise"
                else {}
            )
            for parameter, field in fields.items():
                value = self.hardware_state.get(field, {}).get("value")
                if value is not None:
                    item["params"][parameter] = bool(value) if parameter == "enabled" else value
        if kind == "combine":
            item["params"]["tab"] = next(t for t in "BCD" if t != self.tab)
        item["expanded"] = kind != "preview"
        if stack == "software":
            position = max(1, min(position, len(self.nodes(stack)) - (self.tab == "A")))
        self.nodes(stack).insert(position, item)
        self.rebuild()
        self.publish()

    def remove(self, stack: Any, position: Any) -> None:
        """
        Remove.
        """
        if self.locked or self.nodes(stack)[position]["type"] in ("source", "output"):
            return
        del self.nodes(stack)[position]
        self.rebuild()
        self.publish()

    def clear(self, stack: Any = None) -> None:
        """
        Clear.
        """
        if self.locked:
            return
        if stack in (None, "hardware"):
            self.document["hardware"] = []
        if stack in (None, "software"):
            tabs = (
                software_tabs(self.document)
                if stack is None
                else {self.tab: self.nodes("software")}
            )
            for tab, nodes in tabs.items():
                kept = [n for n in nodes if n["type"] in ("source", "output")]
                if tab == "A":
                    self.document["software"] = kept
                else:
                    self.document["branches"][tab] = kept
        self.rebuild()
        self.publish()

    def defaults(self) -> None:
        """
        Defaults.
        """
        if not self.locked:
            self.document = default_pipeline()
            self.rebuild()
            self.publish()

    def change(self, item: Any, key: Any, value: Any) -> None:
        """
        Change.
        """
        if self.locked:
            return
        if (
            item["type"] == "enhance"
            and key == "model"
            and value != item["params"][key]
            and value in ("acnet", "anime4k09")
        ):
            item["params"]["passes"] = 1 if value == "acnet" else 3
            widget = self.widgets.get(item["id"])
            row = widget[1].get("passes") if widget is not None else None
            if row is not None:
                row.timer.stop()
                row.update_state(item["params"]["passes"], True)
        item["params"][key] = value
        if (
            item["type"] == "filter"
            and key == "filter"
            and item["params"]["kernel"] not in allowed_kernels(value)
        ):
            item["params"]["kernel"] = 0
        if item["type"] == "detail" and key == "enabled" and not value:
            item["params"]["fixed"] = False
        self.publish()

    def expand(
        self, item: Any, body: Any, listing: Any, _entry: Any, _checked: bool = False
    ) -> None:
        """
        Expand.
        """
        if item["type"] == "output":
            return
        item["expanded"] = not item["expanded"]
        title = self.widgets[item["id"]][3]
        stack = self.widgets[item["id"]][0]
        title.setText(
            ("▾ " if item["expanded"] else "▸ ")
            + node_title(stack, item, getattr(self, "last_state", {}).get("temperature_unit", "C"))
        )
        body.setVisible(item["expanded"])
        listing.fit_contents()
        self.publish()

    def rebuild(self) -> None:
        """
        Rebuild.
        """
        for identity, thumbnail in self.preview_widgets.items():
            if not thumbnail.image.isNull():
                self.preview_cache[identity] = (
                    thumbnail.payload,
                    thumbnail.zoomed,
                    QPointF(thumbnail.pan),
                    thumbnail.elapsed_ms,
                )
        live_ids = {
            n["id"]
            for nodes in software_tabs(self.document).values()
            for n in nodes
            if n["type"] == "preview"
        }
        self.preview_cache = {
            identity: value
            for identity, value in self.preview_cache.items()
            if identity in live_ids
        }
        self.rebuilding = True
        self.widgets = {}
        self.preview_widgets = {}
        self.preview_timing_widgets = {}
        for stack, listing in self.stacks.items():
            listing.clear()
            for item in self.nodes(stack):
                entry = QListWidgetItem(listing)
                entry.setData(Qt.ItemDataRole.UserRole, item["id"])
                if item["type"] in ("source", "output"):
                    entry.setFlags(entry.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
                container = QWidget()
                container.setObjectName("pipelineNode")
                container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                base = listing.palette().color(QPalette.ColorRole.Window)
                background = QColor(
                    *(
                        round(channel * 0.95 + 255 * 0.05)
                        for channel in (base.red(), base.green(), base.blue())
                    )
                )
                border = listing.palette().color(QPalette.ColorRole.Mid)
                container.setStyleSheet(
                    f"QWidget#pipelineNode {{ background-color: {background.name()}; "
                    f"border: 1px solid {border.name()}; border-radius: 4px; }}"
                )
                container.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                container.customContextMenuRequested.connect(
                    lambda _p, bound_listing=listing, bound_entry=entry: bound_listing.context_menu(
                        bound_listing.visualItemRect(bound_entry).center()
                    )
                )
                layout = QVBoxLayout(container)
                header = QHBoxLayout()
                grip = QLabel("⠿")
                grip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                grip.setToolTip("Drag to rearrange")
                grip.setVisible(item["type"] not in ("source", "output"))
                header.addWidget(grip)
                title = QPushButton(
                    ("" if item["type"] == "output" else "▾ " if item["expanded"] else "▸ ")
                    + node_title(
                        stack, item, getattr(self, "last_state", {}).get("temperature_unit", "C")
                    )
                )
                title.setStyleSheet("text-align: left; font-weight: bold")
                title.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                title.customContextMenuRequested.connect(
                    lambda _p, bound_listing=listing, bound_entry=entry: bound_listing.context_menu(
                        bound_listing.visualItemRect(bound_entry).center()
                    )
                )
                header.addWidget(title, 1)
                if item["type"] == "preview":
                    timing = QLabel("— ms")
                    timing.setToolTip(
                        "Cumulative processing time in this tab up to this preview; excludes thumbnail encoding, UI and waiting for other tabs."
                    )
                    header.addWidget(timing)
                    self.preview_timing_widgets[item["id"]] = timing
                from ..view_window import NoWheelCheckBox

                bypass = NoWheelCheckBox("Bypass")
                bypass.setChecked(item["bypass"])
                bypass.setVisible(item["type"] not in ("source", "output"))
                bypass.toggled.connect(partial(self.bypass, item))
                header.addWidget(bypass)
                layout.addLayout(header)
                badge = QLabel()
                badge.setWordWrap(True)
                layout.addWidget(badge)
                body = QWidget()
                fields = QVBoxLayout(body)
                fields.setContentsMargins(4, 0, 4, 0)
                controls = {}
                for key, spec in CATALOG[stack][item["type"]][1].items():
                    row = self.row_class(
                        spec.title,
                        partial(self.change, item, key),
                        minimum=spec.minimum,
                        maximum=spec.maximum,
                        step=spec.step,
                        options=spec.options,
                        unit=spec.unit,
                    )
                    row.update_state(item["params"][key])
                    if item["type"] in (
                        "onnx_superresolution",
                        "onnx_denoise",
                        "onnx_style",
                    ) and key in ("backend", "apple_compute"):
                        row.setVisible(
                            bool(
                                getattr(self, "last_state", {})
                                .get("apple_acceleration", {})
                                .get("available")
                            )
                        )
                    if item["type"] == "enhance" and key in ("backend", "apple_compute"):
                        row.setVisible(
                            bool(
                                getattr(self, "last_state", {})
                                .get("apple_acceleration", {})
                                .get("available")
                            )
                            and (
                                item["params"]["model"] == "acnet"
                                or item["params"]["model"] in ONNX_MODELS
                            )
                        )
                    if item["type"] == "enhance" and key in ("noise", "denoise", "passes", "input"):
                        model = item["params"]["model"]
                        row.setVisible(
                            model == "ffdnet-gray"
                            if key == "noise"
                            else model == "acnet"
                            if key == "denoise"
                            else model in ("acnet", "anime4k09")
                            if key == "passes"
                            else model not in ("dncnn-25", "ffdnet-gray")
                        )
                    controls[key] = row
                    if item["type"] == "filter":
                        row.setVisible(key in filter_fields(item["params"]))
                    elif item["type"] in ("edges", "contours"):
                        row.setVisible(key in feature_fields(item["type"], item["params"]))
                    fields.addWidget(row)
                if item["type"] == "combine":
                    source_row = controls["mask_source"]
                    invert_row = controls["mask_invert"]
                    index = fields.indexOf(source_row)
                    fields.removeWidget(source_row)
                    fields.removeWidget(invert_row)
                    invert_row.layout().itemAt(0).widget().hide()
                    invert_row.input.setText("Invert input")
                    invert_row.input.setAccessibleName("Invert optional mix mask input")
                    invert_row.input.setMinimumWidth(0)
                    invert_row.input.setToolTip(
                        "Reverse the optional mask: dark regions blend and bright regions preserve the current image."
                    )
                    mask_line = QHBoxLayout()
                    mask_line.addWidget(source_row, 1)
                    mask_line.addWidget(invert_row, 0, Qt.AlignmentFlag.AlignBottom)
                    fields.insertLayout(index, mask_line)
                if item["type"] == "preview":
                    thumbnail = PipelinePreview()
                    if item["id"] in self.preview_cache:
                        payload, thumbnail.zoomed, thumbnail.pan, thumbnail.elapsed_ms = (
                            self.preview_cache[item["id"]]
                        )
                        thumbnail.show_image(payload, "Waiting for pipeline image…")
                    self.preview_timing_widgets[item["id"]].setText(
                        f"{float(thumbnail.elapsed_ms):.1f} ms"
                        if thumbnail.elapsed_ms is not None
                        else "— ms"
                    )
                    fields.addWidget(thumbnail)
                    self.preview_widgets[item["id"]] = thumbnail
                body.setVisible(item["expanded"])
                layout.addWidget(body)
                title.clicked.connect(partial(self.expand, item, body, listing, entry))
                listing.setItemWidget(entry, container)
                entry.setSizeHint(container.sizeHint())
                self.widgets[item["id"]] = (stack, controls, bypass, title, badge)
            listing.fit_contents()
        self.update_tab_status()
        self.update_titles()
        self.rebuilding = False

    def update_titles(self) -> None:
        """
        Update titles.
        """
        unit = getattr(self, "last_state", {}).get("temperature_unit", "C")
        for stack in ("hardware", "software"):
            for item in self.nodes(stack):
                title = self.widgets[item["id"]][3]
                prefix = "" if item["type"] == "output" else "▾ " if item["expanded"] else "▸ "
                title.setText(prefix + node_title(stack, item, unit))
                title.setToolTip(node_title(stack, item, unit))

    def update_tab_status(self) -> None:
        """
        Update tab status.
        """
        try:
            active = execution_dependencies(self.document)
        except ValueError:
            active = {}  # A rejected edit is rolled back by publish().
        self.tab_status.setText(
            "Tab A → viewer"
            if self.tab == "A"
            else f"Tab {self.tab} · connected to viewer"
            if self.tab in active
            else f"Tab {self.tab} · preview only"
            if any(
                n["type"] == "preview" and n["expanded"] and not n["bypass"]
                for n in self.nodes("software")
            )
            else f"Tab {self.tab} · idle — expand a Preview node or connect this tab to use its output"
        )
        for i, tab in enumerate("ABCD"):
            self.tab_bar.setTabText(i, tab)
            self.tab_bar.setTabToolTip(
                i, "Viewer output" if tab == "A" else "Connected" if tab in active else "Idle"
            )

    def bypass(self, item: Any, value: Any) -> None:
        """
        Bypass.
        """
        if not self.locked and item["type"] not in ("source", "output"):
            item["bypass"] = value
            self.publish()

    def update_state(self, state: dict[str, Any], locked: bool) -> None:
        """
        Refresh runtime availability and values through the state presenter.
        """
        update_editor_state(self, state, locked)

    def import_file(self) -> None:
        """
        Import file.
        """
        if self.locked:
            return
        directory = load_dialog_directory("pipeline_import")
        path, _ = QFileDialog.getOpenFileName(
            self, "Import pipeline", str(directory or ""), "Pipelines (*.pipeline.json *.json)"
        )
        if not path:
            return
        try:
            candidate = validate_pipeline(json.loads(Path(path).read_text()))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Import rejected", str(exc))
            return
        remember_dialog_directory("pipeline_import", path)
        self.document = candidate
        self.rebuild()
        self.publish()

    def export_file(self) -> None:
        """
        Export file.
        """
        directory = load_dialog_directory("pipeline_export")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export pipeline",
            str((directory or Path.cwd()) / "camera.pipeline.json"),
            "Pipelines (*.pipeline.json)",
        )
        if not path:
            return
        if not path.endswith(".pipeline.json"):
            path += ".pipeline.json"
        destination = Path(path)
        temporary = destination.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(validate_pipeline(self.document), indent=2, allow_nan=False) + "\n"
            )
            temporary.replace(destination)
            remember_dialog_directory("pipeline_export", destination)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
        finally:
            temporary.unlink(missing_ok=True)
