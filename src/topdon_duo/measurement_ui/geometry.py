"""
Native-grid shape preview with draggable polygon vertices and cubic control handles.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from ..measurements.paths import boundary
from ..measurements.schema import default_region


class GeometryEditor(QWidget):
    """
    Edit native coordinates without using display pixels as temperature samples.
    """

    def __init__(self, changed: Callable[[dict, tuple[int, int]], None]) -> None:
        """
        Prepare a sensor-grid preview with independently editable Bezier handles.
        """
        super().__init__()
        self.changed = changed
        self.region = default_region()
        self.center = (128, 96)
        self.sensor_size = (256, 192)
        self.handle: int | None = None
        self.selected = -1
        self.setMinimumSize(300, 220)

    def set_geometry(self, region: dict, center: tuple[int, int], size: tuple[int, int]) -> None:
        """
        Refresh geometry after form edits or authoritative parent updates.
        """
        self.region, self.center, self.sensor_size = deepcopy(region), center, size
        self.update()

    def grid_rect(self) -> QRectF:
        """
        Fit the native aspect ratio inside the available preview space.
        """
        scale = min(
            (self.width() - 40) / self.sensor_size[0], (self.height() - 40) / self.sensor_size[1]
        )
        width, height = self.sensor_size[0] * scale, self.sensor_size[1] * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def display_point(self, point: tuple[float, float] | list) -> QPointF:
        """
        Map a native pixel center into preview coordinates.
        """
        rect = self.grid_rect()
        return QPointF(
            rect.left() + (point[0] + 0.5) * rect.width() / self.sensor_size[0],
            rect.top() + (point[1] + 0.5) * rect.height() / self.sensor_size[1],
        )

    def sensor_point(self, position: QPointF) -> tuple[float, float]:
        """
        Clamp dragged handles to valid native-grid coordinates.
        """
        rect = self.grid_rect()
        return (
            round(
                max(
                    0,
                    min(
                        self.sensor_size[0] - 1,
                        (position.x() - rect.left()) * self.sensor_size[0] / rect.width() - 0.5,
                    ),
                ),
                2,
            ),
            round(
                max(
                    0,
                    min(
                        self.sensor_size[1] - 1,
                        (position.y() - rect.top()) * self.sensor_size[1] / rect.height() - 0.5,
                    ),
                ),
                2,
            ),
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Show the sampled boundary, anchors and distinguishable Bezier handles.
        """
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.grid_rect()
        painter.fillRect(rect, QColor("#20252d"))
        painter.setPen(QPen(QColor("#343c48"), 1))
        for x in range(0, self.sensor_size[0], 16):
            painter.drawLine(
                self.display_point((x, 0)), self.display_point((x, self.sensor_size[1] - 1))
            )
        for y in range(0, self.sensor_size[1], 16):
            painter.drawLine(
                self.display_point((0, y)), self.display_point((self.sensor_size[0] - 1, y))
            )
        points = boundary(self.region, self.center)
        color = QColor("#ed7777") if self.region["exclude"] else QColor("#78c5ff")
        path = QPainterPath(self.display_point(points[0]))
        for point in points[1:]:
            path.lineTo(self.display_point(point))
        path.closeSubpath()
        painter.setPen(
            QPen(
                color, 2, Qt.PenStyle.DashLine if self.region["exclude"] else Qt.PenStyle.SolidLine
            )
        )
        painter.drawPath(path)
        controls = self.region["points"]
        if self.region["kind"] == "bezier":
            painter.setPen(QPen(QColor("#a99d74"), 1, Qt.PenStyle.DashLine))
            for i in range(0, len(controls), 3):
                painter.drawLine(
                    self.display_point(controls[i]), self.display_point(controls[i + 1])
                )
                painter.drawLine(
                    self.display_point(controls[i + 2]),
                    self.display_point(controls[(i + 3) % len(controls)]),
                )
        handles = controls or [self.center]
        for index, point in enumerate(handles):
            anchor = self.region["kind"] != "bezier" or index % 3 == 0
            painter.setBrush(color if anchor else QColor("#e5c774"))
            painter.setPen(
                QPen(Qt.GlobalColor.white if index == self.selected else Qt.GlobalColor.black, 1)
            )
            painter.drawEllipse(self.display_point(point), 5 if anchor else 4, 5 if anchor else 4)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Capture an anchor/control handle without moving the geometry on press.
        """
        if event.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            return
        handles = self.region["points"] or [self.center]
        distances = [
            (
                (self.display_point(point).x() - event.position().x()) ** 2
                + (self.display_point(point).y() - event.position().y()) ** 2,
                index,
            )
            for index, point in enumerate(handles)
        ]
        distance, index = min(distances)
        self.handle = index if distance <= 144 else None
        self.selected = index if self.handle is not None else -1
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Drag the captured vertex/handle and synchronize its numeric coordinates.
        """
        if self.handle is None or not self.isEnabled():
            return
        point = self.sensor_point(event.position())
        if self.region["kind"] == "spot":
            self.center = (round(point[0]), round(point[1]))
        else:
            self.region["points"][self.handle] = list(point)
        self.changed(deepcopy(self.region), self.center)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Finish a geometry gesture.
        """
        self.mouseMoveEvent(event)
        self.handle = None
