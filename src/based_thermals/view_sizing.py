"""
Keep Imaging wide enough for its controls while retaining vertical scrolling.
"""

from PySide6.QtCore import QEvent, QObject, QSize
from PySide6.QtWidgets import QScrollArea


class ImagingScrollArea(QScrollArea):
    """
    Publish content width to the window layout instead of hiding controls behind horizontal scroll.
    """

    def minimumSizeHint(self) -> QSize:
        """
        Reserve the content's minimum width, frame and potential vertical scrollbar.
        """
        hint = super().minimumSizeHint()
        content = self.widget()
        if content is not None:
            width = max(content.minimumWidth(), content.minimumSizeHint().width())
            hint.setWidth(max(hint.width(), width + 2 * self.frameWidth()
                              + self.verticalScrollBar().sizeHint().width()))
        return hint

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Reconsider window width when expanded nodes or refreshed controls change the layout.
        """
        handled = super().eventFilter(watched, event)
        if watched is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return handled
