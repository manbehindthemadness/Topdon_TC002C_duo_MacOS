"""
One acknowledged preset selector for both Environment and Calibrations.
"""

from collections.abc import Callable
from typing import Any, cast

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .view_controls import NoWheelComboBox


class EnvironmentPresetControls(QWidget):
    """
    Send preset operations to the viewer; camera values and file writes stay in its process.
    """

    def __init__(
        self, send: Callable[[dict[str, Any]], None], prepare: Callable[[bool], None],
    ) -> None:
        """
        Build a combined user-only selector with pipeline-style management actions.
        """
        super().__init__()
        self._send, self._prepare = send, prepare
        self._names: list[str] = []
        self._current: str | None = None
        self._locked = True
        self._ready = False
        self._modified = False
        self._choices_signature: tuple[Any, ...] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Environment + calibration presets")
        title.setStyleSheet("font-weight: bold")
        layout.addWidget(title)
        row = QHBoxLayout()
        self.combo = NoWheelComboBox()
        self.combo.setAccessibleName("Environment + calibration presets")
        self.combo.setSizeAdjustPolicy(
            NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon,
        )
        self.combo.setMinimumContentsLength(24)
        self.combo.setToolTip("Load one saved set of environmental values and calibration references.")
        self.combo.activated.connect(self.select)
        row.addWidget(self.combo, 1)
        self.save = QPushButton("Save preset")
        self.save.clicked.connect(self.save_current)
        row.addWidget(self.save)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.update_state({}, True)

    def update_state(self, state: dict[str, Any], locked: bool) -> None:
        """
        Reflect only viewer-acknowledged selections; repeated updates never send commands.
        """
        names = state.get("names", [])
        self._names = ([name for name in names if isinstance(name, str)]
                       if isinstance(names, list) else [])
        current = state.get("current")
        self._current = current if isinstance(current, str) and current in self._names else None
        self._ready = bool(state.get("ready"))
        self._locked = locked or bool(state.get("locked"))
        self._modified = bool(state.get("modified"))
        signature = (tuple(self._names), self._current, self._modified)
        if signature != self._choices_signature:
            self._refresh_choices()
            self._choices_signature = signature
        model = cast(QStandardItemModel, self.combo.model())
        for index in (2, 3):
            model.item(index).setEnabled(self._current is not None and not self._locked)
        self.combo.setEnabled(not self._locked and self._ready)
        self.save.setEnabled(not self._locked and self._ready)
        self.save.setText("Update preset" if self._current is not None else "Save preset")
        message = (
            "Save your current environment values and calibration references together. "
            "Loading applies the whole preset to the camera and recalibrates."
        )
        if self._current is not None:
            message = (
                f'"{self._current}" has unsaved changes. Update the preset to save them.'
                if self._modified else f'Current settings are saved as "{self._current}".'
            )
        if self._locked:
            message = "Close calibration tools and stop logging before using presets."
        elif not self._ready:
            message = "Presets are available when all environmental settings are ready."
        elif not self._names:
            message = "No saved presets yet. " + message
        self.status.setText(message)

    def _refresh_choices(self) -> None:
        """
        Rebuild only changed choices so live frames cannot reset an open preset menu.
        """
        with QSignalBlocker(self.combo):
            self.combo.clear()
            saved_state = ("modified" if self._modified else "saved") if self._current else "unsaved"
            self.combo.addItem(f"Current settings ({saved_state})", ("current", ""))
            self.combo.addItem("Save current settings as…", ("save", ""))
            self.combo.addItem("Rename current preset…", ("rename", ""))
            self.combo.addItem("Delete current preset…", ("delete", ""))
            if self._names:
                self.combo.insertSeparator(self.combo.count())
            for name in self._names:
                label = f"{name} (modified)" if name == self._current and self._modified else name
                self.combo.addItem(label, ("load", name))
            selected = next((index for index in range(self.combo.count())
                             if self.combo.itemData(index) == ("load", self._current)), 0)
            self.combo.setCurrentIndex(selected)

    def _publish(self, command: dict[str, Any], *, save: bool = False) -> None:
        """
        Commit inputs before saving, discard pending edits before loading, and send one action.
        """
        if save or command.get("operation") == "load":
            self._prepare(save)
        self._send({"action": "environment_preset", **command})

    def _name(self, title: str, initial: str = "") -> str | None:
        """
        Ask for a name without changing selection on cancellation or an empty response.
        """
        name, accepted = QInputDialog.getText(self, title, "Preset name:", text=initial)
        result = name.strip() if accepted and name.strip() else None
        return result

    def _replace(self, name: str) -> bool:
        """
        Confirm collisions before replacing a user-created preset.
        """
        accepted = (name not in self._names or QMessageBox.question(
            self, "Replace preset", f'Replace the saved preset "{name}"?',
        ) == QMessageBox.StandardButton.Yes)
        return accepted

    def save_current(self, _checked: bool = False, *, save_as: bool = False) -> None:
        """
        Update the acknowledged preset or save the applied state under a new name.
        """
        if self._locked or not self._ready:
            return
        name = self._current
        if save_as or name is None:
            name = self._name("Save environment + calibration preset")
            if name is None or not self._replace(name):
                return
        self._publish(
            {"operation": "save", "name": name, "replace": name in self._names}, save=True,
        )

    def select(self, index: int) -> None:
        """
        Load or manage the chosen preset while retaining the last acknowledged selector value.
        """
        choice = self.combo.itemData(index)
        selected = next((row for row in range(self.combo.count())
                         if self.combo.itemData(row) == ("load", self._current)), 0)
        self.combo.setCurrentIndex(selected)
        if self._locked or not self._ready or not isinstance(choice, tuple):
            return
        operation, name = choice
        if operation == "load":
            self._publish({"operation": "load", "name": name})
        elif operation == "save":
            self.save_current(save_as=True)
        elif operation == "rename" and self._current is not None:
            target = self._name("Rename environment + calibration preset", self._current)
            if target is None:
                return
            target_name = target
            if target_name != self._current and self._replace(target_name):
                self._publish({"operation": "rename", "name": self._current,
                               "new_name": target_name, "replace": target_name in self._names})
        elif operation == "delete" and self._current is not None:
            answer = QMessageBox.question(
                self, "Delete environment + calibration preset",
                f'Delete the saved preset "{self._current}"?',
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._publish({"operation": "delete", "name": self._current})
