"""
Aspect-preserving Qt image display and image-coordinate pointer events.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    """
    Paint pixels separately from controls and exclude letterboxing from sampling.
    """

    def __init__(self, send: Callable[[dict], None]) -> None:
        """
        Enable tracking and keyboard focus for the live image area.
        """
        super().__init__()
        self.send = send
        self.image = QImage()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(1, 1)
        self._inside = False
        self._dragging = False

    def image_rect(self) -> QRectF:
        """
        Return the centered aspect-preserving image rectangle in widget coordinates.
        """
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def image_position(self, position: QPointF) -> tuple[int, int] | None:
        """
        Map high-DPI logical coordinates to the displayed bitmap, without a toolbar offset.
        """
        rectangle = self.image_rect()
        if rectangle.isEmpty() or not rectangle.contains(position):
            return None
        x = int((position.x() - rectangle.left()) * self.image.width() / rectangle.width())
        y = int((position.y() - rectangle.top()) * self.image.height() / rectangle.height())
        if x >= self.image.width() or y >= self.image.height():
            return None
        return x, y

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Paint a black background and the current owned image.
        """
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.image.isNull():
            painter.drawImage(self.image_rect(), self.image)

    def _pointer(self, event: QMouseEvent, kind: str) -> None:
        """
        Forward image-local positions; preserve releases outside a dragged image.
        """
        if kind != "move" and event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        position = self.image_position(event.position())
        inside = position is not None
        if inside != self._inside:
            self._inside = inside
            self.send({"action": "hover", "inside": inside})
        if position is None:
            if kind == "release" and self._dragging:
                position = (-1, -1)
            else:
                return
        modifiers = event.modifiers()
        flags = (8 if modifiers & Qt.KeyboardModifier.ControlModifier else 0)
        flags |= 16 if modifiers & Qt.KeyboardModifier.ShiftModifier else 0
        flags |= 32 if modifiers & Qt.KeyboardModifier.AltModifier else 0
        flags |= 1 if event.buttons() & Qt.MouseButton.LeftButton else 0
        button = "right" if event.button() == Qt.MouseButton.RightButton else "left"
        if kind == "press" and button == "left":
            self._dragging = True
        elif kind == "release":
            self._dragging = False
        self.send({
            "action": "pointer", "kind": kind, "button": button,
            "x": position[0], "y": position[1], "flags": flags,
        })

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Forward hovering and drag motion.
        """
        self._pointer(event, "move")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Focus the image and begin a pointer gesture.
        """
        self.setFocus()
        self._pointer(event, "press")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Finish a pointer gesture or request a context menu.
        """
        self._pointer(event, "release")

    def leaveEvent(self, event: object) -> None:
        """
        Hide the sampler when the pointer moves into controls or another window.
        """
        del event
        self._inside = False
        self.send({"action": "hover", "inside": False})
