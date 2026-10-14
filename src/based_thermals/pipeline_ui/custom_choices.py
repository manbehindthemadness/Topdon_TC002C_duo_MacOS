"""
Color pickers and local-model dropdowns for declarative custom controls.
"""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ..custom_nodes.colors import color_value
from ..custom_nodes.model_library import add_model, existing_models


class ColorPicker(QWidget):
    """
    Choose fixed RGB colors or the shared measurement-overlay dynamic coloring mode.
    """

    def __init__(self, value: str, changed: Callable[[str], None]) -> None:
        """
        Provide common colors, a dynamic choice and a native color dialog.
        """
        super().__init__()
        self.changed = changed
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for label, color in (("Dynamic (measurement style)", "dynamic"), ("Green", "#00ff00"),
                             ("White", "#ffffff"), ("Black", "#000000"), ("Red", "#ff0000"),
                             ("Blue", "#0000ff"), ("Cyan", "#00ffff"), ("Yellow", "#ffff00")):
            self.combo.addItem(label, color)
        self.button = QPushButton("Choose…")
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.button)
        self.refresh(value)
        self.combo.currentIndexChanged.connect(self.select)
        self.button.clicked.connect(self.choose)

    def refresh(self, value: str) -> None:
        """
        Display current values silently, including arbitrary saved hex colors.
        """
        value = color_value(value)
        blocker = QSignalBlocker(self.combo)
        index = self.combo.findData(value)
        if index < 0:
            self.combo.addItem(f"Custom ({value})", value)
            index = self.combo.count() - 1
        self.combo.setCurrentIndex(index)
        blocker.unblock()

    def select(self, _index: int) -> None:
        """
        Publish the selected mode or literal RGB color.
        """
        self.changed(self.combo.currentData())

    def choose(self) -> None:
        """
        Retain the current color if the native color chooser is canceled.
        """
        value = self.combo.currentData()
        initial = QColor(value if value != "dynamic" else "#00ff00")
        color = QColorDialog.getColor(initial, self, "Choose box color")
        if color.isValid():
            value = color.name()
            self.refresh(value)
            self.changed(value)


class ModelCombo(QComboBox):
    """
    Rescan the local model list when opening the dropdown after a download.
    """

    def __init__(self, refresh: Callable[[], None]) -> None:
        """
        Retain the refresh callback without opening weights or starting downloads.
        """
        super().__init__()
        self.refresh_models = refresh

    def showPopup(self) -> None:
        """
        Refresh model paths before showing the standard Qt popup.
        """
        self.refresh_models()
        super().showPopup()


class ModelPicker(QWidget):
    """
    Select automatic download or an existing model, with optional Browse registration.
    """

    def __init__(self, value: str, changed: Callable[[str], None]) -> None:
        """
        Bind model paths to configuration without creating inference sessions in the UI.
        """
        super().__init__()
        self.changed = changed
        self.value = value
        self.choices: list[tuple[str, str]] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = ModelCombo(lambda: self.refresh(self.value))
        self.combo.setMinimumWidth(1)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(12)
        self.button = QPushButton("Browse…")
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.button)
        self.refresh(value)
        self.combo.currentIndexChanged.connect(self.select)
        self.button.clicked.connect(self.browse)
        self.setToolTip("Select a compatible ONNX model. The processing node validates its format.")

    def refresh(self, value: str) -> None:
        """
        Preserve open popup choices and rebuild closed lists only when models change.

        Routine frame updates must not discard the highlighted popup row. An explicit
        configuration replacement still applies when its saved value changes.
        """
        if self.combo.view().isVisible() and value == self.value:
            return
        self.value = value
        choices = [("Automatic download", "")]
        for path in existing_models(value):
            file = Path(path)
            label = f"{file.name} ({file.parent.name})"
            if not file.is_file():
                label += " — missing"
            choices.append((label, path))
        selected = str(Path(value).expanduser().absolute()) if value else ""
        blocker = QSignalBlocker(self.combo)
        if choices != self.choices:
            self.combo.clear()
            for label, path in choices:
                self.combo.addItem(label, path)
                if path:
                    self.combo.setItemData(self.combo.count() - 1, path, Qt.ItemDataRole.ToolTipRole)
            self.choices = choices
        self.combo.setCurrentIndex(self.combo.findData(selected))
        blocker.unblock()

    def select(self, _index: int) -> None:
        """
        Publish the chosen local path or empty automatic-download value.
        """
        self.value = self.combo.currentData()
        self.changed(self.value)

    def browse(self) -> None:
        """
        Add an existing ONNX file by path and select it; cancellation changes nothing.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Add local model", "", "ONNX models (*.onnx)")
        if not path:
            return
        try:
            value = add_model(Path(path))
            self.refresh(value)
            self.changed(value)
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, "Custom model", str(exc))
