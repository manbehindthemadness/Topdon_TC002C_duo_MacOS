"""Live display controls in a separate Qt process."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .capture_window import run_window
from .view_settings import COLOR_PALETTES, IMAGE_FILTERS, IMAGE_SOURCES, VIEW_DEFAULTS


class ViewWindow(QWidget):
    def __init__(self, send) -> None:
        super().__init__()
        self._send = send
        self.controls = {}
        self.setWindowTitle("View")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        heading = QLabel("Display settings")
        heading.setStyleSheet("font-size: 18px; font-weight: bold")
        layout.addWidget(heading)
        description = QLabel("Changes appear immediately in the image and saved captures.")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        for name, label, options in (
            ("image_source", "Image source", IMAGE_SOURCES),
            ("image_filter", "Image filter", IMAGE_FILTERS),
            ("color_palette", "Color palette", COLOR_PALETTES),
        ):
            control = QComboBox()
            for value, title in options.items():
                control.addItem(title, value)
            control.setAccessibleName(label)
            control.currentIndexChanged.connect(
                lambda _index, name=name, control=control: self._change(name, control.currentData())
            )
            self.controls[name] = control
            form.addRow(label, control)
        layout.addLayout(form)
        mirror = QGroupBox("Mirror")
        mirror_layout = QHBoxLayout(mirror)
        for name, label in (
            ("mirror_horizontal", "Left / right"),
            ("mirror_vertical", "Top / bottom"),
        ):
            control = QCheckBox(label)
            control.toggled.connect(lambda value, name=name: self._change(name, value))
            self.controls[name] = control
            mirror_layout.addWidget(control)
        layout.addWidget(mirror)
        aa = QCheckBox("Antialiasing — smooth image scaling")
        aa.toggled.connect(lambda value: self._change("antialiasing", value))
        self.controls["antialiasing"] = aa
        layout.addWidget(aa)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        reset = QPushButton("Reset display settings")
        reset.clicked.connect(lambda: self._send({"action": "reset"}))
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(reset)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.update_state(VIEW_DEFAULTS)

    def _change(self, name, value) -> None:
        self._send({"action": "setting", "name": name, "value": value})

    def update_state(self, state: dict) -> None:
        for name, control in self.controls.items():
            value = state.get(name, VIEW_DEFAULTS[name])
            with QSignalBlocker(control):
                if isinstance(control, QComboBox):
                    control.setCurrentIndex(control.findData(value))
                else:
                    control.setChecked(value)
        self.status.setText(state.get("status", ""))
        if state.get("raise_window"):
            self.showNormal()
            self.raise_()
            self.activateWindow()


def main() -> int:
    return run_window(ViewWindow, "View")


if __name__ == "__main__":
    raise SystemExit(main())
