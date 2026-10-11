"""
Native Qt viewer and graph controls, independent of the displayed pixels.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QPushButton, QToolBar, QWidget


class ViewerToolbar(QToolBar):
    """
    Provide accessible controls with Qt's overflow menu at smaller window sizes.
    """

    def __init__(self, send: Callable[[dict], None], parent: QWidget | None = None) -> None:
        """
        Create controls and dispatch the existing session action names.
        """
        super().__init__("Viewer controls", parent)
        self.setMovable(False)
        self.setFloatable(False)
        self.buttons: dict[str, QPushButton] = {}
        self.graph_widgets: list[QAction] = []
        for action, label in (
            ("rotate", "Rotate"), ("unit", "Units: C/cm"), ("view", "Camera"),
            ("spots", "Measurements"), ("capture", "Capture"), ("graph", "Show graph"),
            ("help", "Help"), ("fullscreen", "Fullscreen"), ("quit", "Quit"),
        ):
            button = self._button(action, label, send)
            button.setCheckable(action in ("capture", "graph", "help", "fullscreen"))
        self.graph_widgets.append(self.addSeparator())
        self.history = QLabel("History")
        self.graph_widgets.append(self.addWidget(self.history))
        for action, label in (("graph_reset", "Reset"), ("graph_config", "Config")):
            self._button(action, label, send, graph=True)
        self.graph_widgets.append(self.addWidget(QLabel("Update (s)")))
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.1, 60)
        self.interval.setDecimals(1)
        self.interval.setSingleStep(0.1)
        self.interval.setKeyboardTracking(False)
        self.interval.setAccessibleName("Graph update interval in seconds")
        self.interval.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.graph_widgets.append(self.addWidget(self.interval))
        self.interval.valueChanged.connect(
            lambda value: send({"action": "graph_interval", "value": value})
        )
        self._button("graph_log", "Log to CSV", send, graph=True)

    def _button(
        self, action: str, label: str, send: Callable[[dict], None], *, graph: bool = False
    ) -> QPushButton:
        """
        Add a keyboard-accessible button and retain its toolbar action for visibility.
        """
        button = QPushButton(label)
        button.setAccessibleName(label)
        button.clicked.connect(lambda: send({"action": action}))
        widget_action = self.addWidget(button)
        if graph:
            self.graph_widgets.append(widget_action)
        self.buttons[action] = button
        return button

    def update_state(self, state: dict) -> None:
        """
        Reflect locks and activity without emitting commands or overwriting interval edits.
        """
        unit = state.get("unit", "C")
        self.buttons["unit"].setText(f"Units: {unit}/{'in' if unit == 'F' else 'cm'}")
        graph = bool(state.get("show_graph"))
        self.buttons["graph"].setText("Hide graph" if graph else "Show graph")
        for action, value in (
            ("graph", graph),
            ("capture", state.get("recording", False)),
            ("help", state.get("show_instructions", False)),
        ):
            self.buttons[action].setChecked(bool(value))
        for action in ("rotate", "unit"):
            self.buttons[action].setEnabled(
                not state.get("settings_locked", False)
                and not (action == "rotate" and state.get("log_pending", False))
            )
        locked = bool(state.get("graph_locked"))
        self.buttons["graph"].setEnabled(not locked)
        self.buttons["graph_reset"].setEnabled(not state.get("graph_config_locked", locked))
        self.buttons["graph_config"].setEnabled(True)
        self.interval.setEnabled(not locked)
        if not self.interval.hasFocus() or locked:
            with QSignalBlocker(self.interval):
                self.interval.setValue(state.get("graph_interval", 0.5))
        self.buttons["graph_log"].setText(
            "Cancel logging" if state.get("log_pending")
            else "Stop logging" if state.get("logging") else "Log to CSV"
        )
        self.buttons["graph_log"].setEnabled(state.get("radiometry", True))
        self.history.setText(state.get("history", "History"))
        for action in self.graph_widgets:
            action.setVisible(graph)
