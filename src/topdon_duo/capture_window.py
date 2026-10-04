"""Capture popup UI; runs separately to keep the camera and OpenCV responsive."""

from __future__ import annotations

import json
import os
import sys

from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CaptureWindow(QWidget):
    def __init__(self, send) -> None:
        super().__init__()
        self._send = send
        self.setWindowTitle("Capture")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        heading = QLabel("Image data, video and timelapse")
        heading.setStyleSheet("font-size: 18px; font-weight: bold")
        layout.addWidget(heading)
        self.save_image = QPushButton("Save image data")
        self.save_image.clicked.connect(lambda: self._send({"action": "image"}))
        layout.addWidget(self.save_image)
        instructions = QLabel(
            "Save a PNG with thermal data, or choose a location for an MP4 recording."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        form = QFormLayout()
        self.rate = QSpinBox()
        self.rate.setRange(1, 1500)
        self.rate.setValue(60)
        self.rate.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rate.setKeyboardTracking(False)
        self.rate.setAccessibleName("Frames per minute")
        form.addRow("Timelapse frames per minute", self.rate)
        layout.addLayout(form)
        self.cursor = QCheckBox("Capture cursor")
        layout.addWidget(self.cursor)
        buttons = QHBoxLayout()
        self.video = QPushButton("Record video")
        self.timelapse = QPushButton("Record timelapse")
        buttons.addWidget(self.video)
        buttons.addWidget(self.timelapse)
        layout.addLayout(buttons)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(36)
        layout.addWidget(self.status)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)
        self.video.clicked.connect(lambda: self._record("video"))
        self.timelapse.clicked.connect(lambda: self._record("timelapse"))
        self.rate.valueChanged.connect(lambda value: self._send({"action": "rate", "value": value}))
        self.cursor.toggled.connect(
            lambda checked: self._send({"action": "cursor", "value": checked})
        )

    def _record(self, mode: str) -> None:
        self.rate.interpretText()
        self._send(
            {
                "action": mode,
                "frames_per_minute": self.rate.value(),
                "capture_cursor": self.cursor.isChecked(),
            }
        )

    def closeEvent(self, event) -> None:
        self.rate.interpretText()
        super().closeEvent(event)

    def update_state(self, state: dict) -> None:
        mode = state.get("recording_mode")
        pending = state.get("pending_recording")
        locked = mode or pending
        self.rate.setEnabled(not locked)
        with QSignalBlocker(self.rate), QSignalBlocker(self.cursor):
            self.rate.setMaximum(state.get("max_fpm", 1500))
            if locked or not self.rate.hasFocus():
                self.rate.setValue(state.get("frames_per_minute", 60))
            self.cursor.setChecked(state.get("capture_cursor", False))
        for kind, button in (("video", self.video), ("timelapse", self.timelapse)):
            button.setEnabled(not locked or locked == kind)
            if pending == kind:
                button.setText("Cancel")
            elif mode == kind:
                button.setText("Stop video" if kind == "video" else "Stop timelapse")
            else:
                button.setText("Record video" if kind == "video" else "Record timelapse")
        self.status.setText(state.get("status", "Ready"))
        if state.get("raise_window"):
            self.showNormal()
            self.raise_()
            self.activateWindow()


def run_window(window_type, title: str) -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(title)

    def send(message: dict) -> None:
        try:
            print(json.dumps(message), flush=True)
        except BrokenPipeError:
            app.quit()

    window = window_type(send)
    os.set_blocking(sys.stdin.fileno(), False)
    incoming = b""

    def poll_parent() -> None:
        nonlocal incoming
        while True:
            try:
                chunk = os.read(sys.stdin.fileno(), 4096)
            except BlockingIOError:
                break
            if not chunk:
                app.quit()
                return
            incoming += chunk
        while b"\n" in incoming:
            line, incoming = incoming.split(b"\n", 1)
            try:
                state = json.loads(line)
                if isinstance(state, dict):
                    window.update_state(state)
            except (ValueError, UnicodeDecodeError):
                continue

    timer = QTimer()
    timer.timeout.connect(poll_parent)
    timer.start(30)
    window.show()
    return app.exec()


def main() -> int:
    return run_window(CaptureWindow, "Capture")


if __name__ == "__main__":
    raise SystemExit(main())
