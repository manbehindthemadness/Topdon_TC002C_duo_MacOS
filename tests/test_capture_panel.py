import os
import subprocess
import sys
import time

from topdon_duo.capture_panel import CapturePanel


def popup_environment():
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    env.pop("QT_QPA_FONTDIR", None)
    return env


def test_popup_number_field_checkbox_and_recording_states(tmp_path):
    # Test Qt 6 in its own process, just as the app does with OpenCV's Qt 5.
    script = """
import sys
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QSlider
from topdon_duo.capture_window import CaptureWindow

app = QApplication([])
messages = []
window = CaptureWindow(messages.append)
window.show()
app.processEvents()
assert window.windowTitle() == "Capture"
assert window.save_image.text() == "Save image data"
window.save_image.click()
assert messages[-1] == {"action": "image"}
assert window.rate.value() == 60
assert window.rate.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
assert not window.findChildren(QSlider)
window.rate.setFocus()
window.rate.lineEdit().selectAll()
QTest.keyClicks(window.rate, "120")
QTest.keyClick(window.rate, Qt.Key.Key_Return)
assert window.rate.value() == 120
assert messages[-1] == {"action": "rate", "value": 120}
window.cursor.click()
assert messages[-1] == {"action": "cursor", "value": True}
window.timelapse.click()
assert messages[-1] == {"action": "timelapse", "frames_per_minute": 120, "capture_cursor": True}
window.update_state({"recording_mode": "timelapse", "frames_per_minute": 120,
                     "capture_cursor": True, "status": "REC timelapse | 12.0s | 25 frames | 120/min"})
assert not window.rate.isEnabled()
assert not window.video.isEnabled()
assert window.timelapse.text() == "Stop timelapse"
assert window.cursor.isEnabled()
assert window.save_image.isEnabled()
app.processEvents()
window.grab().save(sys.argv[1])
window.update_state({"pending_recording": "video", "frames_per_minute": 60})
assert window.video.text() == "Cancel"
assert not window.timelapse.isEnabled()
window.update_state({"frames_per_minute": 60})
assert window.video.text() == "Record video"
assert window.rate.isEnabled()
window.rate.setFocus()
window.rate.lineEdit().selectAll()
QTest.keyClicks(window.rate, "90")
before_close = len(messages)
window.close()
assert messages[before_close:] == [{"action": "rate", "value": 90}]
assert not window.isVisible()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "capture-popup.png")],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_popup_bridge_is_nonblocking_and_preserves_messages(monkeypatch):
    real_popen = subprocess.Popen
    script = """
import json, sys
for line in sys.stdin:
    state = json.loads(line)
    print(json.dumps({"action": "rate", "value": state["frames_per_minute"]}), flush=True)
"""
    calls = []

    def popen(command, **options):
        calls.append((command, options["env"]))
        return real_popen([sys.executable, "-c", script], **options)

    monkeypatch.setattr("topdon_duo.capture_panel.subprocess.Popen", popen)
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/opencv/qt5/plugins")
    monkeypatch.setenv("QT_QPA_FONTDIR", "/opencv/qt5/fonts")
    panel = CapturePanel()
    try:
        panel.open({"frames_per_minute": 60, "status": "Ready"})
        before = time.monotonic()
        events = panel.poll()
        assert time.monotonic() - before < 0.2
        deadline = time.monotonic() + 3
        while not events and time.monotonic() < deadline:
            events.extend(panel.poll())
            time.sleep(0.01)
        assert events == [{"action": "rate", "value": 60}]
        panel.open({"frames_per_minute": 120, "status": "Ready"})
        assert len(calls) == 1  # Reopening raises the existing popup.
        assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in calls[0][1]
        assert "QT_QPA_FONTDIR" not in calls[0][1]
        events = []
        deadline = time.monotonic() + 3
        while not events and time.monotonic() < deadline:
            events.extend(panel.poll())
            time.sleep(0.01)
        assert events == [{"action": "rate", "value": 120}]
    finally:
        panel.close()
    assert not panel.is_open
    assert panel.poll() == []


def test_popup_bridge_reports_startup_failure(monkeypatch):
    real_popen = subprocess.Popen

    def popen(_command, **options):
        return real_popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('cannot open popup'); sys.exit(1)",
            ],
            **options,
        )

    monkeypatch.setattr("topdon_duo.capture_panel.subprocess.Popen", popen)
    panel = CapturePanel()
    try:
        panel.open({"frames_per_minute": 60})
        deadline = time.monotonic() + 3
        events = []
        while not events and time.monotonic() < deadline:
            events.extend(panel.poll())
            time.sleep(0.01)
        assert events == [{"action": "error", "message": "cannot open popup"}]
        assert not panel.is_open
    finally:
        panel.close()
