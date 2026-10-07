"""Expandable, draggable pipeline stacks; UI stays in the popup process."""

import json
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .dialog_preferences import load_dialog_directory, remember_dialog_directory
from .pipeline import CATALOG, default_pipeline, node, thermal_source, validate_pipeline
from .pipeline_hardware import desired_hardware


class StackList(QListWidget):
    def __init__(self, editor, stack):
        super().__init__()
        self.editor, self.stack = editor, stack
        self.setObjectName("pipelineStack")
        base = self.palette().color(QPalette.Window)
        lighter = QColor(
            *(
                round(channel * 0.88 + 255 * 0.12)
                for channel in (base.red(), base.green(), base.blue())
            )
        )
        self.setStyleSheet(
            f"QListWidget#pipelineStack {{ background-color: {lighter.name()}; padding: 4px; }}"
        )
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSpacing(6)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._fitting = False
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu)
        self.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self.reordered))

    def fit_contents(self):
        if self._fitting:
            return
        self._fitting = True
        try:
            for i in range(self.count()):
                entry = self.item(i)
                widget = self.itemWidget(entry)
                if widget is None:
                    continue
                widget.layout().activate()
                hint = widget.sizeHint()
                if widget.hasHeightForWidth():
                    hint.setHeight(
                        widget.heightForWidth(max(1, self.viewport().width() - 2 * self.spacing()))
                    )
                entry.setSizeHint(hint)
            chrome = self.height() - self.viewport().height()
            height = sum(
                self.item(i).sizeHint().height() + 2 * self.spacing() for i in range(self.count())
            )
            # Keep a little empty space for the stack's Add/Clear context menu.
            self.setFixedHeight(max(64, height + chrome + 24))
            self.verticalScrollBar().setValue(0)
        finally:
            self._fitting = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.fit_contents()

    def dropEvent(self, event):
        if event.source() is not self or self.editor.locked:
            event.ignore()
            return
        index = self.indexAt(event.position().toPoint()).row()
        current = self.currentItem()
        if self.stack == "hardware" and (
            current is None
            or current.data(Qt.UserRole) == self.editor.document["hardware"][0]["id"]
            or index == 0
        ):
            event.ignore()
            return
        super().dropEvent(event)

    def reordered(self):
        if self.editor.rebuilding:
            return
        lookup = {n["id"]: n for n in self.editor.document[self.stack]}
        ordered = [lookup[self.item(i).data(Qt.UserRole)] for i in range(self.count())]
        if self.stack == "hardware" and ordered[0]["type"] != "source":
            self.editor.rebuild()
            return
        self.editor.document[self.stack] = ordered
        self.editor.publish()

    def context_menu(self, position):
        if self.editor.locked:
            return
        index = self.indexAt(position).row()
        menu = QMenu(self)
        if index < 0:
            self.add_menu(menu, "Add node", len(self.editor.document[self.stack]))
            menu.addAction("Clear current stack", lambda: self.editor.clear(self.stack))
            menu.addAction("Clear all nodes", lambda: self.editor.clear())
        else:
            item = self.editor.document[self.stack][index]
            if item["type"] != "source":
                self.add_menu(menu, "Insert node above", index)
                menu.addAction("Clear node", lambda: self.editor.remove(self.stack, index))
            self.add_menu(menu, "Insert node below", index + 1)
        menu.exec(self.viewport().mapToGlobal(position))

    def add_menu(self, menu, title, position):
        add = menu.addMenu(title)
        existing = {n["type"] for n in self.editor.document[self.stack]}
        for kind, (label, _) in CATALOG[self.stack].items():
            if kind == "source":
                continue
            action = add.addAction(
                label,
                lambda _checked=False, kind=kind: self.editor.insert(self.stack, kind, position),
            )
            action.setEnabled(self.stack != "hardware" or kind not in existing)


class PipelineEditor(QWidget):
    def __init__(self, send, row_class):
        super().__init__()
        self.send, self.row_class = send, row_class
        self.document = default_pipeline()
        self.edit_serial = 0
        self.hardware_state = {}
        self.last_desired = desired_hardware(self.document)
        self.locked = False
        self.rebuilding = False
        self.widgets = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        instruction = QLabel(
            "Click a node header to expand. Drag its grip to rearrange. Right-click to add, insert or clear nodes."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self.stacks = {}
        for stack, title in (
            ("hardware", "Camera hardware · order organizes controls"),
            ("software", "Software processing · runs from top to bottom"),
        ):
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
        layout.addLayout(buttons)
        self.rebuild()

    def publish(self):
        try:
            validate_pipeline(self.document)
        except ValueError as exc:
            QMessageBox.warning(self, "Pipeline rejected", str(exc))
            self.send({"action": "pipeline_refresh"})
            return
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

    def insert(self, stack, kind, position):
        if self.locked:
            return
        if stack == "hardware" and any(n["type"] == kind for n in self.document[stack]):
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
        item["expanded"] = True
        self.document[stack].insert(max(1, position) if stack == "hardware" else position, item)
        self.rebuild()
        self.publish()

    def remove(self, stack, position):
        if self.locked or self.document[stack][position]["type"] == "source":
            return
        del self.document[stack][position]
        self.rebuild()
        self.publish()

    def clear(self, stack=None):
        if self.locked:
            return
        if stack in (None, "hardware"):
            self.document["hardware"] = self.document["hardware"][:1]
        if stack in (None, "software"):
            self.document["software"] = []
        self.rebuild()
        self.publish()

    def defaults(self):
        if not self.locked:
            self.document = default_pipeline()
            self.rebuild()
            self.publish()

    def change(self, item, key, value):
        if self.locked:
            return
        item["params"][key] = value
        if item["type"] == "detail" and key == "enabled" and not value:
            item["params"]["fixed"] = False
        self.publish()

    def expand(self, item, body, listing, entry):
        item["expanded"] = not item["expanded"]
        title = self.widgets[item["id"]][3]
        stack = self.widgets[item["id"]][0]
        title.setText(("▾ " if item["expanded"] else "▸ ") + CATALOG[stack][item["type"]][0])
        body.setVisible(item["expanded"])
        listing.fit_contents()
        self.publish()

    def rebuild(self):
        self.rebuilding = True
        self.widgets = {}
        for stack, listing in self.stacks.items():
            listing.clear()
            for item in self.document[stack]:
                entry = QListWidgetItem(listing)
                entry.setData(Qt.UserRole, item["id"])
                if item["type"] == "source":
                    entry.setFlags(entry.flags() & ~Qt.ItemIsDragEnabled)
                container = QWidget()
                container.setObjectName("pipelineNode")
                container.setAttribute(Qt.WA_StyledBackground, True)
                base = listing.palette().color(QPalette.Window)
                background = QColor(
                    *(
                        round(channel * 0.95 + 255 * 0.05)
                        for channel in (base.red(), base.green(), base.blue())
                    )
                )
                border = listing.palette().color(QPalette.Mid)
                container.setStyleSheet(
                    f"QWidget#pipelineNode {{ background-color: {background.name()}; "
                    f"border: 1px solid {border.name()}; border-radius: 4px; }}"
                )
                container.setContextMenuPolicy(Qt.CustomContextMenu)
                container.customContextMenuRequested.connect(
                    lambda _p, listing=listing, entry=entry: listing.context_menu(
                        listing.visualItemRect(entry).center()
                    )
                )
                layout = QVBoxLayout(container)
                header = QHBoxLayout()
                grip = QLabel("⠿")
                grip.setAttribute(Qt.WA_TransparentForMouseEvents)
                grip.setToolTip("Drag to rearrange")
                header.addWidget(grip)
                title = QPushButton(
                    ("▾ " if item["expanded"] else "▸ ") + CATALOG[stack][item["type"]][0]
                )
                title.setStyleSheet("text-align: left; font-weight: bold")
                title.setContextMenuPolicy(Qt.CustomContextMenu)
                title.customContextMenuRequested.connect(
                    lambda _p, listing=listing, entry=entry: listing.context_menu(
                        listing.visualItemRect(entry).center()
                    )
                )
                header.addWidget(title, 1)
                from .view_window import NoWheelCheckBox

                bypass = NoWheelCheckBox("Bypass")
                bypass.setChecked(item["bypass"])
                bypass.setVisible(item["type"] != "source")
                bypass.toggled.connect(lambda value, item=item: self.bypass(item, value))
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
                        lambda value, item=item, key=key: self.change(item, key, value),
                        minimum=spec.minimum,
                        maximum=spec.maximum,
                        step=spec.step,
                        options=spec.options,
                        unit=spec.unit,
                    )
                    row.update_state(item["params"][key])
                    controls[key] = row
                    fields.addWidget(row)
                body.setVisible(item["expanded"])
                layout.addWidget(body)
                title.clicked.connect(
                    lambda _c=False, item=item, body=body, listing=listing, entry=entry: (
                        self.expand(item, body, listing, entry)
                    )
                )
                listing.setItemWidget(entry, container)
                entry.setSizeHint(container.sizeHint())
                self.widgets[item["id"]] = (stack, controls, bypass, title, badge)
            listing.fit_contents()
        self.rebuilding = False

    def bypass(self, item, value):
        if not self.locked:
            item["bypass"] = value
            self.publish()

    def update_state(self, state, locked):
        self.hardware_state = state.get("hardware", {})
        incoming = state.get("pipeline")
        # Do not overwrite edits with IPC state queued before their command.
        if incoming is not None and state.get("pipeline_serial", 0) >= self.edit_serial:
            incoming = validate_pipeline(incoming)
            self.edit_serial = max(self.edit_serial, state.get("pipeline_serial", 0))
            if incoming != self.document:
                # Reuse controls when possible so edits don't jump the scroll position.
                old_structure = [
                    (n["id"], n["expanded"])
                    for s in ("hardware", "software")
                    for n in self.document[s]
                ]
                new_structure = [
                    (n["id"], n["expanded"]) for s in ("hardware", "software") for n in incoming[s]
                ]
                if old_structure != new_structure:
                    self.document = incoming
                    self.rebuild()
                else:
                    for stack in ("hardware", "software"):
                        for old, new in zip(self.document[stack], incoming[stack]):
                            old.update(new)
        self.last_desired = desired_hardware(self.document)
        self.locked = locked
        thermal = thermal_source(self.document) or state.get("actual_image_source") == "raw"
        for stack in ("hardware", "software"):
            for item in self.document[stack]:
                _, controls, bypass, title, badge = self.widgets[item["id"]]
                inactive = (
                    stack == "hardware" and thermal and item["type"] not in ("source", "humidity")
                )
                unavailable = (
                    stack == "hardware"
                    and item["type"] != "source"
                    and not state.get("processing_preset_available", False)
                )
                badge.setText(
                    "Inactive for thermal rendering; settings retained"
                    if inactive
                    else "Bypassed"
                    if item["bypass"]
                    else ""
                )
                badge.setVisible(bool(badge.text()))
                bypass.blockSignals(True)
                bypass.setChecked(item["bypass"])
                bypass.blockSignals(False)
                bypass.setEnabled(not locked and not unavailable)
                title.setEnabled(not locked)
                for key, row in controls.items():
                    row.set_display_unit(state.get("temperature_unit", "C"))
                    available = (
                        not locked and not inactive and not unavailable and not item["bypass"]
                    )
                    if item["type"] == "enhance":
                        available &= key != "denoise" or item["params"]["model"] == "acnet"
                    row.update_state(item["params"][key], available)
                # Child size changes (badges) need a refreshed item height.
        for listing in self.stacks.values():
            listing.setDragEnabled(not locked)
            listing.fit_contents()
        for button in (self.import_button, self.export_button, self.defaults_button):
            button.setEnabled(not locked)

    def import_file(self):
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

    def export_file(self):
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
