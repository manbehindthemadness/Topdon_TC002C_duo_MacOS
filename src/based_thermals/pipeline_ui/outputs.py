"""
Read-only camera output footer, separate from editable and persisted pipeline nodes.
"""

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget


def output_sources(state: dict[str, Any], stack: str) -> str:
    """
    Name supported sources owned by a camera, without treating a missing frame as unsupported.
    """
    profile = state.get("camera_profile")
    if not isinstance(profile, dict) or not any(
        name in profile for name in ("preview", "radiometry", "visible")
    ):
        return "Waiting for camera…"
    sources = []
    if stack == "visible_hardware":
        if profile.get("visible"):
            sources.append("Color camera")
    else:
        if profile.get("preview"):
            sources.append("Thermal upsample")
        if profile.get("radiometry"):
            sources.append("Raw thermal")
    return " · ".join(sources) if sources else "None available"


class HardwareOutputs(QWidget):
    """
    Present a node-shaped footer without selection, editing, dragging or node identity.
    """

    def __init__(self, parent: QWidget) -> None:
        """
        Match pipeline node surfaces and headers while passing mouse events to the stack.
        """
        super().__init__(parent)
        self.setObjectName("hardwareOutputs")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        base = self.palette().color(QPalette.ColorRole.Window)
        background = QColor(*(round(channel * .95 + 255 * .05)
                              for channel in (base.red(), base.green(), base.blue())))
        border = self.palette().color(QPalette.ColorRole.Mid)
        self.setStyleSheet(
            f"QWidget#hardwareOutputs {{ background-color: {background.name()}; "
            f"border: 1px solid {border.name()}; border-radius: 4px; }}")
        layout = QVBoxLayout(self)
        self.title = QPushButton()
        self.title.setStyleSheet("text-align: left; font-weight: bold")
        self.title.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.title.setAccessibleName("Output sources")
        layout.addWidget(self.title)

    def update_sources(self, state: dict[str, Any], stack: str) -> None:
        """
        Refresh descriptive text from cached capabilities without publishing pipeline changes.
        """
        self.title.setText("Output sources → " + output_sources(state, stack))
