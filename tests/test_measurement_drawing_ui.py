"""
Offscreen Qt drawing tools, keyboard controls and bitmap-coordinate event forwarding.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_popup_starts_drawing_and_video_toolbar_owns_finish_undo_cancel(tmp_path: Path) -> None:
    """
    Uncover the live preview and handle Escape as draft cancellation before fullscreen exit.
    """
    script = r"""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from topdon_duo.measurements_window import MeasurementsWindow
from topdon_duo.viewer_window import ViewerWindow

app = QApplication([])
commands = []
popup = MeasurementsWindow(commands.append)
state = {"sensor_size": [256,192], "orientation": [0,False,False], "spots": [], "locked": False}
popup.update_state(state)
popup.show()
popup.kind.setCurrentIndex(3)
popup.draw_exclude.setChecked(True)
popup.create.click()
assert commands[-1] == {"action": "draw", "kind": "polygon", "exclude": True, "orientation": [0,False,False]}
assert not popup.isVisible()
popup.update_state({**state, "raise_window": True})
popup.auto_hide.setChecked(False)
popup.create.click()
assert commands[-1]["action"] == "draw" and popup.isVisible()
# Live state updates preserve the local visibility choice.
popup.update_state(state)
assert not popup.auto_hide.isChecked()
assert popup.close_button.isEnabled() and not popup.apply.isEnabled()
count = len(commands)
popup.close_button.click()
assert not popup.isVisible() and len(commands) == count
popup.update_state({**state, "raise_window": True, "locked": True})
assert popup.isVisible()
count = len(commands)
assert not popup.create.isEnabled() and not popup.draw_exclude.isEnabled()
popup.create.click()
assert len(commands) == count
assert popup.close_button.isEnabled()
popup.close_button.click()
assert not popup.isVisible()

viewer = ViewerWindow(commands.append)
viewer.canvas.image = QImage(400, 200, QImage.Format.Format_RGB888)
viewer.canvas.image.fill(Qt.GlobalColor.black)
viewer.show()
viewer.update_state({"unit": "C", "measurement_drawing": {
    "active": True, "kind": "polygon", "hint": "Click vertices", "exclude": True,
    "finish": False, "undo": True,
}})
app.processEvents()
assert viewer.drawing_toolbar.isVisible()
viewer._send_geometry()
assert commands[-1]["windowed_size"] == [viewer.canvas.width(), viewer.canvas.height() + viewer.drawing_toolbar.height()]
assert "Exclusion" in viewer.drawing_toolbar.label.text()
assert not viewer.drawing_toolbar.buttons["draw_finish"].isEnabled()
viewer.drawing_toolbar.buttons["draw_undo"].click()
assert commands[-1] == {"action": "draw_undo"}
viewer.canvas.setFocus()
QTest.keyClick(viewer.canvas, Qt.Key.Key_Return)
assert commands[-1] == {"action": "draw_finish"}
QTest.keyClick(viewer.canvas, Qt.Key.Key_Backspace)
assert commands[-1] == {"action": "draw_undo"}

# The actual Qt canvas sends image-local bitmap coordinates despite letterboxing.
rectangle = viewer.canvas.image_rect()
commands.clear()
QTest.mousePress(viewer.canvas, Qt.MouseButton.LeftButton, pos=rectangle.center().toPoint())
QTest.mouseRelease(viewer.canvas, Qt.MouseButton.LeftButton, pos=rectangle.center().toPoint())
pointers = [message for message in commands if message.get("action") == "pointer"]
assert [(p["kind"], p["x"], p["y"]) for p in pointers] == [("press",200,100),("release",200,100)]
assert viewer.canvas.image_position(QPointF(0, 0)) is None
commands.clear()
QTest.mousePress(viewer.canvas, Qt.MouseButton.LeftButton, pos=rectangle.center().toPoint())
QTest.mouseRelease(viewer.canvas, Qt.MouseButton.LeftButton, pos=QPointF(1, 1).toPoint())
assert commands[-1]["action"] == "pointer" and commands[-1]["x"] == commands[-1]["y"] == -1

viewer.toolbar.buttons["fullscreen"].click()
app.processEvents()
commands.clear()
QTest.keyClick(viewer.canvas, Qt.Key.Key_Escape)
assert {"action": "draw_cancel"} in commands and viewer.isFullScreen()
assert {"action": "quit"} not in commands
viewer.update_state({"unit": "C", "measurement_drawing": {"active": False}})
assert not viewer.drawing_toolbar.isVisible()
QTest.keyClick(viewer.canvas, Qt.Key.Key_Escape)
app.processEvents()
assert not viewer.isFullScreen()
viewer.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
