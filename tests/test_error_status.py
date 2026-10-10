"""
Camera error markers flash without changing messages or sending user commands.
"""

from pathlib import Path

import pytest
from support.desktop_recording import viewer_fixture
from support.qt_process import run_popup

from topdon_duo import desktop
from topdon_duo.desktop_app.session import DesktopSession

viewer = viewer_fixture


@pytest.mark.usefixtures("viewer")
@pytest.mark.parametrize("hardware_error,pipeline_error", [("", ""), ("Rejected", ""),
                                                         ("", "Inference failed")])
def test_camera_status_error_uses_explicit_error_state(
    hardware_error: str, pipeline_error: str,
) -> None:
    """
    Mark hardware and image-processing failures without treating slow work as an error.
    """
    session = DesktopSession(desktop, [])
    session.hardware.error = hardware_error
    session.pipeline_error = pipeline_error
    session.pipeline_elapsed_ms = 600
    state = session.view_state()
    assert state["status_error"] is bool(hardware_error or pipeline_error)
    assert "Slow pipeline" in state["status"]


def test_camera_error_status_flashes_and_clears_without_commands(tmp_path: Path) -> None:
    """
    Exercise repeated states, timer ticks, clearing, and window visibility in Qt.
    """
    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
alert = window.status_alert
assert alert.icon.isHidden() and not alert.timer.isActive()
state = {"status": "Camera setting rejected: <Balanced> baseline", "status_error": True}
window.update_state(state)
assert window.status.text() == state["status"]
assert alert.icon.isVisible() and alert.timer.isActive() and alert.lit
icon_width = alert.icon.sizeHint().width()
alert.timer.timeout.emit()
assert not alert.lit and "transparent" in alert.icon.styleSheet()
window.update_state(state)
assert not alert.lit  # Frequent incoming frames must not reset the flash.
assert alert.icon.sizeHint().width() == icon_width
alert.timer.timeout.emit()
assert alert.lit and "#ff3030" in alert.icon.styleSheet()
window.hide()
assert not alert.timer.isActive()
window.show()
app.processEvents()
assert alert.timer.isActive()
window.update_state({"status": "Camera preview", "status_error": False})
assert alert.icon.isHidden() and not alert.timer.isActive()
assert window.status.text() == "Camera preview"
assert messages == []
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
