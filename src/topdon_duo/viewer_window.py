"""
Qt main viewer in a separate process, with controls outside the image canvas.
"""

from __future__ import annotations

import mmap
from collections.abc import Callable

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QKeyEvent, QResizeEvent
from PySide6.QtWidgets import QMainWindow

from .capture_window import run_window
from .viewer_panel import FRAME_CAPACITY
from .viewer_ui.canvas import ImageCanvas
from .viewer_ui.drawing_toolbar import DrawingToolbar
from .viewer_ui.orientation import OrientationDialog
from .viewer_ui.toolbar import ViewerToolbar


class ViewerWindow(QMainWindow):
    """
    Own window geometry, fullscreen and input; acquisition stays in the parent.
    """

    def __init__(self, send: Callable[[dict], None]) -> None:
        """
        Build a native toolbar and a resizable black image area.
        """
        super().__init__()
        self.send = send
        self._mapping: mmap.mmap | None = None
        self._path = ""
        self._windowed_size: list[int] | None = None
        self._geometry_pending = False
        self.canvas = ImageCanvas(send)
        self.setCentralWidget(self.canvas)
        self.toolbar = ViewerToolbar(self._command, self)
        self.addToolBar(self.toolbar)
        self.drawing_toolbar = DrawingToolbar(self._command, self)
        self.addToolBarBreak()
        self.addToolBar(self.drawing_toolbar)
        self._viewer_state: dict = {}
        self.orientation_dialog = OrientationDialog(self.send, self)
        self.setWindowTitle("TOPDON TC002C Duo")
        self.resize(1024, 800)
        self.statusBar().showMessage("Waiting for camera…")

    def _command(self, message: dict) -> None:
        """
        Handle window-local fullscreen, forwarding other actions unchanged.
        """
        if message.get("action") == "rotate":
            self.orientation_dialog.open_state(self._viewer_state)
            return
        if message.get("action") == "fullscreen":
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            self.send(message)
        self.canvas.setFocus()

    def _read_frame(self, frame: dict) -> None:
        """
        Copy a validated BGR frame before acknowledging reuse of its shared storage.
        """
        width, height, path = frame.get("width"), frame.get("height"), frame.get("path")
        if (
            not isinstance(width, int) or isinstance(width, bool)
            or not isinstance(height, int) or isinstance(height, bool)
            or not isinstance(path, str)
        ):
            raise TypeError("Invalid viewer frame descriptor")
        if min(width, height) <= 0 or width * height * 3 > FRAME_CAPACITY:
            raise ValueError("Invalid viewer frame dimensions")
        frame_path = path
        if frame_path != self._path:
            self._close_mapping()
            with open(frame_path, "rb") as stream:
                self._mapping = mmap.mmap(stream.fileno(), FRAME_CAPACITY, access=mmap.ACCESS_READ)
            self._path = frame_path
        assert self._mapping is not None
        pixels = self._mapping[:width * height * 3]
        self.canvas.image = QImage(pixels, width, height, width * 3, QImage.Format.Format_BGR888).copy()
        self.canvas.update()
        self.send({"action": "frame_read", "serial": frame.get("serial")})

    def update_state(self, state: dict) -> None:
        """
        Update controls and image independently, without raising the window on each frame.
        """
        frame = state.get("frame")
        if isinstance(frame, dict):
            try:
                self._read_frame(frame)
            except (OSError, ValueError, TypeError) as exc:
                self.send({"action": "error", "message": str(exc)})
                self.send({"action": "frame_read", "serial": frame.get("serial")})
        if "unit" in state:
            self._viewer_state = state
            self.toolbar.update_state(state)
            self.orientation_dialog.update_locks(state)
            drawing = state.get("measurement_drawing", {})
            drawing_was_active = self.drawing_toolbar.active
            self.drawing_toolbar.update_state(drawing if isinstance(drawing, dict) else {})
            if self.drawing_toolbar.active and not drawing_was_active:
                self.raise_()
                self.activateWindow()
                self.canvas.setFocus()
            self.statusBar().showMessage(str(state.get("status", "Ready")))
        size = state.get("resize")
        if (
            isinstance(size, list) and len(size) == 2
            and all(type(n) is int and 0 < n <= 16384 for n in size)
            and not self.isFullScreen()
        ):
            self.resize(
                size[0], size[1] + self.toolbar.height() + self.statusBar().height()
            )
        self._schedule_geometry()

    def _schedule_geometry(self) -> None:
        """
        Coalesce native resize events after Qt has laid out toolbar and canvas.
        """
        if not self._geometry_pending:
            self._geometry_pending = True
            QTimer.singleShot(0, self._send_geometry)

    def _send_geometry(self) -> None:
        """
        Report image-area dimensions and the last windowed size for persistence.
        """
        self._geometry_pending = False
        size = [self.canvas.width(), self.canvas.height()]
        if not self.isFullScreen():
            drawing_height = self.drawing_toolbar.height() if self.drawing_toolbar.isVisible() else 0
            self._windowed_size = [size[0], size[1] + drawing_height]
        self.send({"action": "geometry", "size": size, "windowed_size": self._windowed_size})

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Let Qt lay out the canvas and then report its dimensions.
        """
        super().resizeEvent(event)
        self._schedule_geometry()

    def changeEvent(self, event: QEvent) -> None:
        """
        Track native fullscreen changes, including the macOS green window button.
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and hasattr(self, "toolbar"):
            self.toolbar.buttons["fullscreen"].setChecked(self.isFullScreen())
            self.toolbar.buttons["fullscreen"].setText(
                "Exit fullscreen" if self.isFullScreen() else "Fullscreen"
            )
            self._schedule_geometry()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Keep viewer shortcuts and let Escape leave fullscreen before requesting quit.
        """
        key = event.key()
        if self.drawing_toolbar.active and key in (
            Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Backspace,
        ):
            action = ("draw_cancel" if key == Qt.Key.Key_Escape else "draw_undo"
                      if key == Qt.Key.Key_Backspace else "draw_finish")
            self._command({"action": action})
            event.accept()
            return
        if key == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
            return
        text = event.text().lower()
        if key == Qt.Key.Key_Escape:
            self.send({"action": "quit"})
        elif text in ("q", "o", "p", " ", "g", "l", "f", "v", "s", "c"):
            self.send({"action": "key", "key": ord(text)})
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _close_mapping(self) -> None:
        """
        Release borrowed file storage after the Qt-owned image has been copied.
        """
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self._path = ""

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Notify the parent to restore hardware and finish recordings on window close.
        """
        self._send_geometry()
        self.send({"action": "quit"})
        self._close_mapping()
        super().closeEvent(event)


def main() -> int:
    """
    Run the Qt viewer without the popup windows' always-on-top flag.
    """
    result = run_window(ViewerWindow, "TOPDON TC002C Duo", stays_on_top=False, poll_ms=10)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
