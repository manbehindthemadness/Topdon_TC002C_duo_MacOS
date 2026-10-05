"""Measuring-spot context menu in the separate Qt process."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QWidget,
    QWidgetAction,
)

from .capture_window import run_window


class SpotsMenu(QMenu):
    def __init__(self, send):
        super().__init__()
        self.setWindowTitle("Measuring spots")
        self._send = send
        self._state = None
        self.checkboxes = {}
        self.clear_buttons = {}
        self.name_fields = {}

    def show(self):
        # Wait for the initial state before showing the menu at the pointer.
        if self._state is not None:
            self.popup(QCursor.pos())

    def _widget(self, widget):
        action = QWidgetAction(self)
        action.setDefaultWidget(widget)
        self.addAction(action)

    def _rename(self, number):
        field = self.name_fields[number]
        if field.isEnabled() and field.isModified():
            self._send({"action": "rename", "spot": number, "name": field.text()})
            field.setModified(False)

    def _build(self, spots):
        for field in self.name_fields.values():
            field.blockSignals(True)
        self.clear()
        self.checkboxes = {}
        self.clear_buttons = {}
        self.name_fields = {}
        self.clear_all = QPushButton("Clear all spots")
        self.clear_all.clicked.connect(lambda: self._send({"action": "clear_all"}))
        self._widget(self.clear_all)
        self.placing = QCheckBox("Add spots")
        self.placing.toggled.connect(
            lambda enabled: self._send({"action": "placing", "enabled": enabled})
        )
        self._widget(self.placing)
        self.addSeparator()
        for spot in spots:
            number = spot["number"]
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 3, 8, 3)
            check = QCheckBox(f"{number}:")
            check.setAccessibleName(f"Enable spot {number}")
            check.setToolTip("Show this spot and sample its temperature graph.")
            check.toggled.connect(
                lambda enabled, number=number: self._send(
                    {"action": "enable", "spot": number, "enabled": enabled}
                )
            )
            name = QLineEdit()
            name.setAccessibleName(f"Region name for spot {number}")
            name.setPlaceholderText(f"Spot {number}")
            name.setMaxLength(64)
            name.setMinimumWidth(160)
            name.setToolTip(
                "Region name used in the graph and log. On-image numbers stay unchanged. Press Enter or leave the field to save."
            )
            name.editingFinished.connect(lambda number=number: self._rename(number))
            clear = QPushButton("Clear")
            clear.setAccessibleName(f"Clear spot {number}")
            clear.clicked.connect(
                lambda _checked=False, number=number: self._send(
                    {"action": "clear", "spot": number}
                )
            )
            layout.addWidget(check)
            layout.addWidget(name, 1)
            layout.addWidget(clear)
            self._widget(row)
            self.checkboxes[number] = check
            self.clear_buttons[number] = clear
            self.name_fields[number] = name
        if not spots:
            self._widget(QLabel("No measuring spots. Enable Add spots, then click the image."))
        self.lock_message = QLabel("Stop logging or close calibration to edit spots.")
        self._widget(self.lock_message)

    def update_state(self, state):
        first = self._state is None
        spots = state.get("spots", [])
        numbers = tuple(spot["number"] for spot in spots)
        if numbers != self._state:
            self._state = numbers
            self._build(spots)
        locked = bool(state.get("locked"))
        self.clear_all.setEnabled(bool(spots) and not locked)
        self.lock_message.setVisible(locked)
        with QSignalBlocker(self.placing):
            self.placing.setChecked(bool(state.get("placing")))
            self.placing.setEnabled(not locked)
        for spot in spots:
            number = spot["number"]
            check = self.checkboxes[number]
            with QSignalBlocker(check):
                check.setChecked(spot["enabled"])
                check.setEnabled(not locked)
            self.clear_buttons[number].setEnabled(not locked)
            name = self.name_fields[number]
            with QSignalBlocker(name):
                # Incoming frames must not overwrite a region name being edited.
                if locked or not (name.hasFocus() and name.isModified()):
                    value = spot.get("name", f"Spot {number}")
                    if name.text() != value:
                        name.setText(value)
                name.setEnabled(not locked)
        if first or state.get("raise_window"):
            self.show()

    def hideEvent(self, event):
        # Clicking outside the menu commits edits before the helper exits.
        for number in self.name_fields:
            self._rename(number)
        super().hideEvent(event)
        QApplication.instance().quit()


def main():
    return run_window(SpotsMenu, "Measuring spots")


if __name__ == "__main__":
    raise SystemExit(main())
