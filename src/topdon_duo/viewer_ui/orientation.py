"""
Native preview rotation and graph-layout popup, independent of camera processing.
"""

from collections.abc import Callable

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OrientationDialog(QDialog):
    """
    Edit preview orientation while the acquisition parent owns validation and persistence.
    """

    def __init__(self, send: Callable[[dict], None], parent: QWidget) -> None:
        """
        Build nonmodal rotation/layout controls without changing the preview on open.
        """
        super().__init__(parent)
        self.send = send
        self.setWindowTitle("Preview rotation and layout")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.rotation = QComboBox()
        for degrees in (0, 90, 180, 270):
            self.rotation.addItem(f"{degrees}° clockwise", degrees)
        self.layout_choice = QComboBox()
        self.layout_choice.addItem("Landscape — graphs to the right", "landscape")
        self.layout_choice.addItem("Portrait — graphs below video", "portrait")
        form.addRow("Preview rotation", self.rotation)
        form.addRow("UI layout", self.layout_choice)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.apply = QPushButton("Apply")
        self.apply.clicked.connect(self._apply)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.hide)
        buttons.addWidget(self.apply)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def open_state(self, state: dict) -> None:
        """
        Start each editing session from the current authoritative parent settings.
        """
        with QSignalBlocker(self.rotation), QSignalBlocker(self.layout_choice):
            self.rotation.setCurrentIndex(max(0, self.rotation.findData(state.get("rotation", 0))))
            self.layout_choice.setCurrentIndex(
                max(0, self.layout_choice.findData(state.get("viewer_layout", "landscape")))
            )
        self.update_locks(state)
        self.show()
        self.raise_()
        self.activateWindow()

    def update_locks(self, state: dict) -> None:
        """
        Honor newly acquired locks without overwriting edits on every incoming frame.
        """
        enabled = not (state.get("settings_locked") or state.get("log_pending"))
        for control in (self.rotation, self.layout_choice, self.apply):
            control.setEnabled(enabled)

    def _apply(self) -> None:
        """
        Send both choices together and dismiss the popup after a permitted edit.
        """
        if not self.apply.isEnabled():
            return
        self.send({"action": "orientation", "rotation": self.rotation.currentData(),
                   "layout": self.layout_choice.currentData()})
        self.hide()
