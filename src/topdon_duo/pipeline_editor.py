"""Expandable, draggable pipeline stacks; UI stays in the popup process."""

import base64
import json
import math
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPixmap
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
    QSizePolicy,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from .dialog_preferences import load_dialog_directory, remember_dialog_directory
from .feature_processing import feature_fields
from .image_filters import allowed_kernels, filter_fields
from .pipeline import (
    CATALOG,
    default_pipeline,
    execution_dependencies,
    node,
    preview_required,
    software_tabs,
    validate_pipeline,
)
from .pipeline_hardware import desired_hardware
from .pipeline_titles import node_title


class PipelinePreview(QLabel):
    """Small, aspect-preserving preview; decoding never affects widget geometry."""

    def __init__(self):
        super().__init__("Waiting for pipeline image…")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(200)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base); padding: 4px"
        )
        self.image = QPixmap()
        self.payload = None
        self.zoomed = True
        self.elapsed_ms = None
        self.pan = QPointF(0.5, 0.5)
        self.drag_anchor = None
        self.setToolTip("Right-click to toggle fit-width zoom. Drag to pan while zoomed.")

    def show_image(self, payload, placeholder):
        if payload == self.payload and (payload is not None or self.text() == placeholder):
            return
        self.payload = payload
        self.image = QPixmap()
        if isinstance(payload, str) and len(payload) <= 420_000:
            try:
                decoded = QImage.fromData(base64.b64decode(payload, validate=True))
                if not decoded.isNull() and decoded.width() <= 320 and decoded.height() <= 240:
                    self.image = QPixmap.fromImage(decoded)
            except ValueError:
                pass
        if self.image.isNull():
            self.drag_anchor = None
            self.setCursor(Qt.ArrowCursor)
            self.clear()
            self.setText(placeholder)
        else:
            self.fit_image()

    def fit_image(self):
        if self.image.isNull():
            return
        area = self.contentsRect().size()
        if area.width() < 1 or area.height() < 1:
            return
        if self.zoomed:
            # Paint only the visible source strip; don't allocate the full enlarged image.
            scale = area.width() / self.image.width()
            visible_height = min(area.height(), round(self.image.height() * scale))
            source_height = min(self.image.height(), visible_height / scale)
            source_top = self.pan.y() * (self.image.height() - source_height)
            viewport = QPixmap(area.width(), visible_height)
            viewport.fill(Qt.transparent)
            painter = QPainter(viewport)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(
                QRectF(0, 0, area.width(), visible_height),
                self.image,
                QRectF(0, source_top, self.image.width(), source_height),
            )
            painter.end()
            self.setPixmap(viewport)
        else:
            self.setPixmap(self.image.scaled(area, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setCursor(Qt.OpenHandCursor if self.zoomed else Qt.ArrowCursor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            if not self.image.isNull():
                self.zoomed = not self.zoomed
                self.pan = QPointF(0.5, 0.5)
                self.drag_anchor = None
                self.fit_image()
            event.accept()
        elif event.button() == Qt.LeftButton:
            if self.zoomed and not self.image.isNull():
                self.drag_anchor = event.position()
                self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_anchor is not None and not self.image.isNull():
            delta = event.position() - self.drag_anchor
            self.drag_anchor = event.position()
            area = self.contentsRect().size()
            scaled_height = self.image.height() * area.width() / self.image.width()
            travel_y = max(0, round(scaled_height) - area.height())
            if travel_y:
                self.pan.setY(max(0, min(1, self.pan.y() - delta.y() / travel_y)))
            self.fit_image()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_anchor = None
            if self.zoomed and not self.image.isNull():
                self.setCursor(Qt.OpenHandCursor)
            event.accept()
        elif event.button() == Qt.RightButton:
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        # The image's right-click belongs to zoom; the node header keeps its menu.
        event.accept()


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
        if self.stack == "software" and (
            current is None
            or current.data(Qt.UserRole) == self.editor.nodes("software")[0]["id"]
            or current.data(Qt.UserRole) == self.editor.nodes("software")[-1]["id"]
            and self.editor.nodes("software")[-1]["type"] == "output"
            or index == 0
            or (self.editor.tab == "A" and index < 0)
        ):
            event.ignore()
            return
        super().dropEvent(event)

    def reordered(self):
        if self.editor.rebuilding:
            return
        lookup = {n["id"]: n for n in self.editor.nodes(self.stack)}
        ordered = [lookup[self.item(i).data(Qt.UserRole)] for i in range(self.count())]
        if self.stack == "software" and (
            not ordered
            or ordered[0]["type"] != "source"
            or (self.editor.tab == "A" and ordered[-1]["type"] != "output")
        ):
            self.editor.rebuild()
            return
        self.editor.set_nodes(self.stack, ordered)
        self.editor.publish()

    def context_menu(self, position):
        if self.editor.locked:
            return
        index = self.indexAt(position).row()
        menu = QMenu(self)
        if index < 0:
            self.add_menu(menu, "Add node", len(self.editor.nodes(self.stack)))
            menu.addAction("Clear current stack", lambda: self.editor.clear(self.stack))
            menu.addAction("Clear all nodes", lambda: self.editor.clear())
        else:
            item = self.editor.nodes(self.stack)[index]
            if item["type"] not in ("source", "output"):
                self.add_menu(menu, "Insert node above", index)
                menu.addAction("Clear node", lambda: self.editor.remove(self.stack, index))
            if item["type"] != "output":
                self.add_menu(menu, "Insert node below", index + 1)
            else:
                self.add_menu(menu, "Insert node above", index)
        menu.exec(self.viewport().mapToGlobal(position))

    def add_menu(self, menu, title, position):
        add = menu.addMenu(title)
        existing = {n["type"] for n in self.editor.nodes(self.stack)}
        for kind, (label, _) in CATALOG[self.stack].items():
            if kind in ("source", "output"):
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
        self.accepted_document = deepcopy(self.document)
        self.tab = "A"
        self.edit_serial = 0
        self.hardware_state = {}
        self.last_desired = desired_hardware(self.document)
        self.locked = False
        self.rebuilding = False
        self.widgets = {}
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
        layout.addLayout(buttons)
        self.rebuild()

    def nodes(self, stack, document=None):
        document = self.document if document is None else document
        return (
            document[stack]
            if stack == "hardware" or self.tab == "A"
            else document["branches"][self.tab]
        )

    def set_nodes(self, stack, nodes):
        if stack == "hardware" or self.tab == "A":
            self.document[stack] = nodes
        else:
            self.document["branches"][self.tab] = nodes

    def select_tab(self, index):
        # Commit debounced inputs before their widgets are destroyed by tab switching.
        if not self.locked:
            for _, controls, *_ in list(self.widgets.values()):
                for row in controls.values():
                    if hasattr(row.input, "interpretText"):
                        row.input.interpretText()
                    if row.timer.isActive():
                        row._emit()
        self.tab = "ABCD"[index]
        self.rebuild()
        if hasattr(self, "last_state"):
            self.update_state(self.last_state, self.locked)

    def publish(self):
        try:
            validate_pipeline(self.document)
        except ValueError as exc:
            QMessageBox.warning(self, "Pipeline rejected", str(exc))
            self.document = deepcopy(self.accepted_document)
            self.rebuild()
            self.send({"action": "pipeline_refresh"})
            return
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

    def insert(self, stack, kind, position):
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

    def remove(self, stack, position):
        if self.locked or self.nodes(stack)[position]["type"] in ("source", "output"):
            return
        del self.nodes(stack)[position]
        self.rebuild()
        self.publish()

    def clear(self, stack=None):
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

    def defaults(self):
        if not self.locked:
            self.document = default_pipeline()
            self.rebuild()
            self.publish()

    def change(self, item, key, value):
        if self.locked:
            return
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

    def expand(self, item, body, listing, entry):
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

    def rebuild(self):
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
                entry.setData(Qt.UserRole, item["id"])
                if item["type"] in ("source", "output"):
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
                grip.setVisible(item["type"] not in ("source", "output"))
                header.addWidget(grip)
                title = QPushButton(
                    ("" if item["type"] == "output" else "▾ " if item["expanded"] else "▸ ")
                    + node_title(
                        stack, item, getattr(self, "last_state", {}).get("temperature_unit", "C")
                    )
                )
                title.setStyleSheet("text-align: left; font-weight: bold")
                title.setContextMenuPolicy(Qt.CustomContextMenu)
                title.customContextMenuRequested.connect(
                    lambda _p, listing=listing, entry=entry: listing.context_menu(
                        listing.visualItemRect(entry).center()
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
                from .view_window import NoWheelCheckBox

                bypass = NoWheelCheckBox("Bypass")
                bypass.setChecked(item["bypass"])
                bypass.setVisible(item["type"] not in ("source", "output"))
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
                    if item["type"] == "filter":
                        row.setVisible(key in filter_fields(item["params"]))
                    elif item["type"] in ("edges", "contours"):
                        row.setVisible(key in feature_fields(item["type"], item["params"]))
                    fields.addWidget(row)
                if item["type"] == "preview":
                    thumbnail = PipelinePreview()
                    if item["id"] in self.preview_cache:
                        payload, thumbnail.zoomed, thumbnail.pan, thumbnail.elapsed_ms = (
                            self.preview_cache[item["id"]]
                        )
                        thumbnail.show_image(payload, "Waiting for pipeline image…")
                    self.preview_timing_widgets[item["id"]].setText(
                        f"{thumbnail.elapsed_ms:.1f} ms"
                        if thumbnail.elapsed_ms is not None
                        else "— ms"
                    )
                    fields.addWidget(thumbnail)
                    self.preview_widgets[item["id"]] = thumbnail
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
        self.update_tab_status()
        self.update_titles()
        self.rebuilding = False

    def update_titles(self):
        unit = getattr(self, "last_state", {}).get("temperature_unit", "C")
        for stack in ("hardware", "software"):
            for item in self.nodes(stack):
                title = self.widgets[item["id"]][3]
                prefix = "" if item["type"] == "output" else "▾ " if item["expanded"] else "▸ "
                title.setText(prefix + node_title(stack, item, unit))
                title.setToolTip(node_title(stack, item, unit))

    def update_tab_status(self):
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

    def bypass(self, item, value):
        if not self.locked and item["type"] not in ("source", "output"):
            item["bypass"] = value
            self.publish()

    def update_state(self, state, locked):
        self.last_state = state
        self.hardware_state = state.get("hardware", {})
        incoming = state.get("pipeline")
        if incoming is not None and state.get("pipeline_serial", 0) >= self.edit_serial:
            incoming = validate_pipeline(incoming)
            self.edit_serial = max(self.edit_serial, state.get("pipeline_serial", 0))
            if incoming != self.document:
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
                elapsed = (
                    state.get("pipeline_preview_timings", {}).get(item["id"]) if current else None
                )
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
                if item["type"] == "preview":
                    elapsed = self.preview_widgets[item["id"]].elapsed_ms
                    self.preview_timing_widgets[item["id"]].setText(
                        f"{elapsed:.1f} ms" if elapsed is not None else "— ms"
                    )
                for key, row in controls.items():
                    row.set_display_unit(state.get("temperature_unit", "C"))
                    available = (
                        not locked and not inactive and not unavailable and not item["bypass"]
                    )
                    if item["type"] == "filter":
                        relevant = key in filter_fields(item["params"])
                        row.setVisible(relevant)
                        available &= relevant
                        if key == "kernel":
                            allowed = allowed_kernels(item["params"]["filter"])
                            for index in range(row.input.count()):
                                row.input.model().item(index).setEnabled(
                                    row.input.itemData(index) in allowed
                                )
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
                            key not in ("base_weight", "input_weight", "offset")
                            or mode == "weighted"
                        )
                        available &= key not in ("threshold", "invert") or mode == "mask"
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
