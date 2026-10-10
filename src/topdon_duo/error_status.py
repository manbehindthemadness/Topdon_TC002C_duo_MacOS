"""
Present status text with a flashing warning marker when an error is active.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class ErrorStatus(QWidget):
    """
    Keep the message readable while flashing a fixed-size trailing red warning.
    """

    def __init__(self) -> None:
        """
        Create a plain-text message and a timer owned by the status widget.
        """
        super().__init__()
        self.message = QLabel()
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        self.icon = QLabel("⚠")
        self.icon.setAccessibleName("Active error")
        self.icon.setToolTip("An error needs attention; see the status message")
        self.icon.setStyleSheet("color: #ff3030; font-size: 22px; font-weight: bold")
        self.icon.hide()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.message, 1)
        layout.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignVCenter)
        self.active_error = False
        self.lit = True
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.flash)

    def update_status(self, message: str, error: bool) -> None:
        """
        Update text without restarting the flash on repeated viewer updates.
        """
        self.message.setText(message)
        self.icon.setVisible(error)
        if error != self.active_error:
            self.active_error = error
            self.lit = True
            self.icon.setStyleSheet("color: #ff3030; font-size: 22px; font-weight: bold")
        if error and self.isVisible():
            if not self.timer.isActive():
                self.timer.start()
        else:
            self.timer.stop()

    def flash(self) -> None:
        """
        Alternate marker color while preserving its position and message layout.
        """
        self.lit = not self.lit
        color = "#ff3030" if self.lit else "transparent"
        self.icon.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold")

    def showEvent(self, event: QShowEvent) -> None:
        """
        Resume the warning when a window containing an active error is reopened.
        """
        super().showEvent(event)
        if self.active_error:
            self.timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        """
        Stop animation while the status is hidden or its window is closed.
        """
        self.timer.stop()
        super().hideEvent(event)
