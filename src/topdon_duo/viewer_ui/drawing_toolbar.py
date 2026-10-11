"""
Contextual native controls for an active video measurement drawing session.
"""

from collections.abc import Callable

from PySide6.QtWidgets import QLabel, QPushButton, QToolBar, QWidget


class DrawingToolbar(QToolBar):
    """
    Keep finish/undo/cancel controls outside the camera pixels.
    """

    def __init__(self, send: Callable[[dict], None], parent: QWidget) -> None:
        """
        Build a hidden contextual toolbar that appears only during drawing.
        """
        super().__init__("Measurement drawing", parent)
        self.setMovable(False)
        self.active = False
        self.label = QLabel()
        self.addWidget(self.label)
        self.buttons: dict[str, QPushButton] = {}
        for action, text in (
            ("draw_finish", "Finish (Enter)"),
            ("draw_undo", "Undo (Backspace)"),
            ("draw_cancel", "Cancel (Esc)"),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, command=action: send({"action": command}))
            self.addWidget(button)
            self.buttons[action] = button
        self.hide()

    def update_state(self, state: dict) -> None:
        """
        Reflect authoritative parent availability without enabling incomplete commits.
        """
        self.active = bool(state.get("active"))
        self.setVisible(self.active)
        role = "Exclusion" if state.get("exclude") else "Measurement"
        self.label.setText(f"{role} · {state.get('kind', '')}: {state.get('hint', '')}  ")
        self.buttons["draw_finish"].setEnabled(bool(state.get("finish")))
        self.buttons["draw_undo"].setEnabled(bool(state.get("undo")))
