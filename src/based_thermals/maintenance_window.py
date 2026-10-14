"""
Explicit cleanup choices and destructive-data warnings in the separate Qt process.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .capture_window import run_window

CHOICES = {
    "models": "Downloaded model cache",
    "runtimes": "Downloaded inference runtime caches",
    "settings": "Settings, saved pipelines/presets and measurement configuration",
    "sessions": "Debug logs, camera snapshots, session artifacts (:memory:.ses) and test/lint caches",
    "bytecode": "Application Python bytecode (.pyc / __pycache__)",
    "outputs": "Generated captures, recordings, logs and pipeline exports in listed output folders",
}
DATA_WARNING = (
    "Deletes saved custom pipelines/presets, embedded custom-node entries and their configuration, "
    "measurements, calibration preferences, model registrations and window/dialog settings. "
    "Original custom-node folders and user-added camera backend source files are preserved."
)


class MaintenanceWindow(QWidget):
    """
    Confirm a bounded reset plan; the parent alone resolves paths and performs deletion.
    """

    def __init__(self, send: Callable[[dict], None]) -> None:
        """
        Build cleanup choices, path scope and an explicit factory-reset action.
        """
        super().__init__()
        self.send = send
        self.state: dict = {}
        self.scope: dict[str, list[str]] = {}
        self.setWindowTitle("Maintenance")
        self.resize(720, 610)
        layout = QVBoxLayout(self)
        message = QLabel(
            "Maintenance closes the viewer before cleanup so settings and caches cannot be "
            "immediately recreated. Start the app again afterward. Close other viewer instances first."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        self.debug = QCheckBox("Debug logs")
        self.debug.setToolTip(
            "Keep detailed application, USB/timing and stall logs for support. "
            "This preference persists across launches."
        )
        self.debug.toggled.connect(
            lambda enabled: self.send({"action": "debug_logs", "enabled": enabled})
        )
        layout.addWidget(self.debug)
        export_row = QHBoxLayout()
        self.export_settings = QPushButton("Export settings…")
        self.export_all = QPushButton("Export all…")
        self.export_settings.clicked.connect(lambda: self.choose_export("settings"))
        self.export_all.clicked.connect(lambda: self.choose_export("all"))
        export_row.addWidget(self.export_settings)
        export_row.addWidget(self.export_all)
        layout.addLayout(export_row)
        share = QLabel(
            "Exports create a ZIP without closing the viewer. Export all includes settings, "
            "available logs and system/debug details. Settings can contain paths and embedded "
            "custom-node code; review the archive before sharing. Images, videos, model weights "
            "and original node/backend folders are excluded."
        )
        share.setWordWrap(True)
        layout.addWidget(share)
        self.choices: dict[str, QCheckBox] = {}
        for category, title in CHOICES.items():
            control = QCheckBox(title)
            control.setToolTip(DATA_WARNING if category == "settings" else title)
            self.choices[category] = control
            layout.addWidget(control)
        warning = QLabel("Warning: " + DATA_WARNING)
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.paths = QTextEdit()
        self.paths.setReadOnly(True)
        self.paths.setAccessibleName("Maintenance cleanup locations")
        layout.addWidget(self.paths, 1)
        self.clear = QPushButton("Clear selected items and close")
        self.clear.clicked.connect(self.clear_selected)
        layout.addWidget(self.clear)
        self.factory = QPushButton("Factory reset and close")
        self.factory.clicked.connect(self.factory_reset)
        layout.addWidget(self.factory)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.update_state({})

    def choose_export(self, kind: str) -> None:
        """
        Choose an explicit ZIP destination in Qt while the parent continues capture.
        """
        if self.state.get("export_busy"):
            return
        name = (
            f"based-thermals-{kind}-"
            + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            + ".zip"
        )
        dialog = QFileDialog(
            self,
            "Export settings" if kind == "settings" else "Export all support information",
            name,
            "ZIP archives (*.zip)",
        )
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("zip")
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            filename = dialog.selectedFiles()[0]
            self.send({"action": "export", "kind": kind, "path": filename})

    def clear_selected(self) -> None:
        """
        Require confirmation for the selected categories, naming user-data loss explicitly.
        """
        self.confirm([key for key, control in self.choices.items() if control.isChecked()])

    def factory_reset(self) -> None:
        """
        Select all application-owned cleanup categories for first-install testing.
        """
        self.confirm(list(CHOICES))

    def confirm(self, categories: list[str]) -> None:
        """
        Show a concrete deletion scope with Cancel as the default; send only categories.
        """
        if not categories or self.state.get("locked"):
            return
        message = "The viewer will close. The following items will be deleted:\n\n"
        message += "\n".join(CHOICES[key] for key in categories)
        if "settings" in categories:
            message += "\n\nWARNING: " + DATA_WARNING
        if "outputs" in categories:
            message += (
                "\n\nWARNING: Generated captures, recordings, logs and .pipeline.json exports "
            )
            message += "in the listed output folders will be permanently deleted."
        if "models" in categories:
            message += "\n\nMissing models may download again on the next launch."
        if "runtimes" in categories:
            message += "\nInference runtime setup may run/download again on the next launch."
        message += "\n\nLocations:\n" + "\n".join(
            path for key in categories for path in self.scope.get(key, [])
        )
        reply = QMessageBox.warning(
            self,
            "Confirm maintenance",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.send({"action": "cleanup", "categories": categories})
            self.clear.setEnabled(False)
            self.factory.setEnabled(False)
            self.status.setText("Closing the viewer for maintenance…")

    def update_state(self, state: dict) -> None:
        """
        Keep choices while displaying authoritative parent locks and cleanup roots.
        """
        self.state = state
        locked = bool(state.get("locked"))
        busy = bool(state.get("export_busy"))
        blocker = QSignalBlocker(self.debug)
        self.debug.setChecked(state.get("debug_logs") is True)
        del blocker
        self.debug.setEnabled(not busy)
        self.export_settings.setEnabled(not busy)
        self.export_all.setEnabled(not busy)
        self.clear.setEnabled(not locked)
        self.factory.setEnabled(not locked)
        for control in self.choices.values():
            control.setEnabled(not locked)
        scope = state.get("paths")
        self.scope = (
            {
                key: value
                for key, value in scope.items()
                if isinstance(key, str)
                and key in CHOICES
                and isinstance(value, list)
                and all(isinstance(path, str) for path in value)
            }
            if isinstance(scope, dict)
            else {}
        )
        self.paths.setPlainText(
            "\n\n".join(
                CHOICES[key] + ":\n" + "\n".join(value) for key, value in self.scope.items()
            )
        )
        self.status.setText(
            str(state.get("export_status") or "")
            or (
                "Stop recording/logging or finish calibration before maintenance." if locked else ""
            )
        )
        if state.get("raise_window"):
            self.showNormal()
            self.raise_()
            self.activateWindow()


def main() -> int:
    """
    Run Maintenance without loading the capture process's OpenCV Qt binding.
    """
    result = run_window(MaintenanceWindow, "Maintenance")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
