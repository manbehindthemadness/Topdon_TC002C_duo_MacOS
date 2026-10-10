"""
Pipeline thumbnails and draggable stack-list widgets.
"""

import base64
from typing import Any, cast

from PySide6.QtCore import QModelIndex, QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QDrag, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QLabel,
    QListWidget,
    QMenu,
    QSizePolicy,
    QWidget,
)

from ..pipeline import (
    CATALOG,
    LEGACY_SOFTWARE_NODES,
)


def decode_preview(payload: str | None) -> QPixmap:
    """
    Decode a bounded preview payload, returning an empty image for malformed data.
    """
    image = QPixmap()
    if not isinstance(payload, str) or len(payload) > 420_000:
        return image
    try:
        decoded = QImage.fromData(base64.b64decode(payload, validate=True))
        if not decoded.isNull() and decoded.width() <= 320 and decoded.height() <= 240:
            image = QPixmap.fromImage(decoded)
    except ValueError:
        pass
    return image


class PipelinePreview(QLabel):
    """Small, aspect-preserving preview; decoding never affects widget geometry."""

    def __init__(self) -> None:
        """
        Init.
        """
        super().__init__("Waiting for pipeline image…")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(200)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base); padding: 4px"
        )
        self.image = QPixmap()
        self.payload: str | None = None
        self.zoomed = True
        self.elapsed_ms: float | None = None
        self.pan = QPointF(0.5, 0.5)
        self.drag_anchor = None
        self.setToolTip("Right-click to toggle fit-width zoom. Drag to pan while zoomed.")

    def show_image(self, payload: str | None, placeholder: str) -> None:
        """
        Show image.
        """
        if payload == self.payload and (payload is not None or self.text() == placeholder):
            return
        self.payload = payload
        self.image = decode_preview(payload)
        if self.image.isNull():
            self.drag_anchor = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.clear()
            self.setText(placeholder)
        else:
            self.fit_image()

    def fit_image(self) -> None:
        """
        Fit image.
        """
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
            viewport.fill(Qt.GlobalColor.transparent)
            painter = QPainter(viewport)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(
                QRectF(0, 0, area.width(), visible_height),
                self.image,
                QRectF(0, source_top, self.image.width(), source_height),
            )
            painter.end()
            self.setPixmap(viewport)
        else:
            self.setPixmap(
                self.image.scaled(
                    area,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoomed else Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event: Any) -> None:
        """
        Resizeevent.
        """
        super().resizeEvent(event)
        self.fit_image()

    def mousePressEvent(self, event: Any) -> None:
        """
        Mousepressevent.
        """
        if event.button() == Qt.MouseButton.RightButton:
            if not self.image.isNull():
                self.zoomed = not self.zoomed
                self.pan = QPointF(0.5, 0.5)
                self.drag_anchor = None
                self.fit_image()
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.zoomed and not self.image.isNull():
                self.drag_anchor = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        """
        Mousemoveevent.
        """
        if self.drag_anchor is not None and not self.image.isNull():
            delta = event.position() - self.drag_anchor
            self.drag_anchor = event.position()
            area = self.contentsRect().size()
            scaled_height = self.image.height() * area.width() / self.image.width()
            travel_y = max(0, round(scaled_height) - area.height())
            if travel_y:
                self.pan.setY(max(0, min(1, self.pan.y() - delta.y() / travel_y)))
            self.fit_image()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        """
        Mousereleaseevent.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_anchor = None
            if self.zoomed and not self.image.isNull():
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        # The image's right-click belongs to zoom; the node header keeps its menu.
        """
        Contextmenuevent.
        """
        event.accept()


class StackList(QListWidget):
    def __init__(self, editor: Any, stack: Any) -> None:
        """
        Init.
        """
        super().__init__()
        self.editor, self.stack = editor, stack
        self.setObjectName("pipelineStack")
        base = self.palette().color(QPalette.ColorRole.Window)
        lighter = QColor(
            *(
                round(channel * 0.88 + 255 * 0.12)
                for channel in (base.red(), base.green(), base.blue())
            )
        )
        self.setStyleSheet(
            f"QListWidget#pipelineStack {{ background-color: {lighter.name()}; padding: 4px; }}"
        )
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(False)
        self.placement_bar = QFrame(self.viewport())
        self.placement_bar.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.placement_bar.setStyleSheet("background: palette(highlight); border-radius: 2px;")
        self.placement_bar.hide()
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setSpacing(6)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._fitting = False
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu)
        self.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self.reordered))

    def fit_contents(self) -> None:
        """
        Fit contents.
        """
        if self._fitting:
            return
        self._fitting = True
        try:
            for i in range(self.count()):
                entry = self.item(i)
                widget = cast(QWidget | None, self.itemWidget(entry))
                if widget is None:
                    continue
                widget_layout = widget.layout()
                if widget_layout is not None:
                    widget_layout.activate()
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

    def resizeEvent(self, event: Any) -> None:
        """
        Resizeevent.
        """
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.fit_contents()

    def startDrag(self, supported_actions: Qt.DropAction) -> None:
        """
        Drag a node while keeping ownership of model moves in this stack.
        """
        current = self.currentItem()
        if self.editor.locked or current is None:
            return
        if not current.flags() & Qt.ItemFlag.ItemIsDragEnabled:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mimeData([current]))
        drag.setPixmap(self.viewport().grab(self.visualItemRect(current)))
        try:
            drag.exec(supported_actions & Qt.DropAction.MoveAction, Qt.DropAction.MoveAction)
        finally:
            self.placement_bar.hide()

    def drop_row(self, position: QPoint) -> int | None:
        """
        Find the insertion boundary, rejecting fixed endpoints and unchanged order.
        """
        current = self.currentItem()
        if self.editor.locked or current is None:
            return None
        index = next(
            (
                row
                for row in range(self.count())
                if position.y() < self.visualItemRect(self.item(row)).center().y()
            ),
            self.count(),
        )
        if self.stack == "software":
            nodes = self.editor.nodes(self.stack)
            moving = nodes[self.row(current)]
            if moving["type"] in ("source", "output"):
                return None
            maximum = self.count() - (nodes[-1]["type"] == "output")
            if not 1 <= index <= maximum:
                return None
        if index in (self.row(current), self.row(current) + 1):
            return None
        return index

    def dragMoveEvent(self, event: Any) -> None:
        """
        Show a visible placement bar only at valid insertion boundaries.
        """
        destination = self.drop_row(event.position().toPoint()) if event.source() is self else None
        if destination is None:
            self.placement_bar.hide()
            event.ignore()
            return
        if destination < self.count():
            y = self.visualItemRect(self.item(destination)).top() - self.spacing()
        else:
            y = self.visualItemRect(self.item(self.count() - 1)).bottom() + self.spacing()
        self.placement_bar.setGeometry(4, max(0, y - 2), max(1, self.viewport().width() - 8), 4)
        self.placement_bar.show()
        self.placement_bar.raise_()
        event.setDropAction(Qt.DropAction.MoveAction)
        event.accept()

    def dragLeaveEvent(self, event: Any) -> None:
        """
        Clear the placement marker when a drag leaves this stack.
        """
        self.placement_bar.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: Any) -> None:
        """
        Move to the indicated boundary without allowing cross-stack transfers.
        """
        self.placement_bar.hide()
        destination = self.drop_row(event.position().toPoint()) if event.source() is self else None
        if destination is None:
            event.ignore()
            return
        moved = self.model().moveRows(
            QModelIndex(), self.currentRow(), 1, QModelIndex(), destination
        )
        if moved:
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def reordered(self) -> None:
        """
        Reordered.
        """
        if self.editor.rebuilding:
            return
        lookup = {n["id"]: n for n in self.editor.nodes(self.stack)}
        ordered = [lookup[self.item(i).data(Qt.ItemDataRole.UserRole)] for i in range(self.count())]
        if self.stack == "software" and (
            not ordered
            or ordered[0]["type"] != "source"
            or (self.editor.tab == "A" and ordered[-1]["type"] != "output")
        ):
            self.editor.rebuild()
            return
        self.editor.set_nodes(self.stack, ordered)
        self.editor.publish()

    def context_menu(self, position: QPoint) -> None:
        """
        Context menu.
        """
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

    def add_menu(self, menu: Any, title: Any, position: Any) -> None:
        """
        Add menu.
        """
        add = menu.addMenu(title)
        existing = {n["type"] for n in self.editor.nodes(self.stack)}
        for kind, (label, _) in CATALOG[self.stack].items():
            if self.stack == "hardware":
                capabilities = self.editor.last_state.get("camera_capabilities", {})
                if capabilities and kind == "device_control":
                    if capabilities.get("features", {}).get("duo_nodes"):
                        continue
                    if not any(spec.get("supported") and spec.get("effect") == "preview"
                               and spec.get("scope", "device") == "device"
                               for spec in capabilities.get("controls", {}).values()):
                        continue
                elif capabilities and not capabilities.get("nodes", {}).get(kind, {}).get(
                    "supported", False,
                ):
                    continue
            if (
                kind in ("source", "output")
                or self.stack == "software"
                and kind in LEGACY_SOFTWARE_NODES
            ):
                continue
            action = add.addAction(
                label,
                lambda _checked=False, bound_kind=kind: self.editor.insert(
                    self.stack, bound_kind, position
                ),
            )
            action.setEnabled(self.stack != "hardware" or kind == "device_control"
                              or kind not in existing)
