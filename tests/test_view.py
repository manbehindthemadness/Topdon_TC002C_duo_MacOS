import subprocess
import sys
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from test_camera import make_frame
from test_capture_panel import popup_environment
from test_desktop_recording import viewer as viewer_fixture
from test_render import frame_with_preview

from topdon_duo import desktop
from topdon_duo.camera import HEADER_U16, SENSOR_PIXELS
from topdon_duo.render import ThermalRenderer
from topdon_duo.view_settings import COLOR_PALETTES, IMAGE_FILTERS, VIEW_DEFAULTS

viewer = viewer_fixture


def test_distance_calibration_detects_native_pixels_and_only_applies_on_request(
    viewer, monkeypatch
):
    panel = Mock()
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    viewer.hardware.state.return_value["distance"] = {"value": 1, "available": True}
    step = 0

    def command(operation, **values):
        return [{"action": "distance_calibration", "operation": operation, **values}]

    def detect(thermal):
        assert thermal.shape == (192, 256)
        edge = 48 if step < 10 else 24
        return np.array([(64, 48), (64 + edge, 48), (64 + edge, 48 + edge), (64, 48 + edge)], float)

    monkeypatch.setattr("topdon_duo.distance_calibration.detect_hand_square", detect)

    def poll():
        nonlocal step
        step += 1
        if step == 1:
            return command("select")
        if step == 9:
            assert panel.update.call_args.args[0]["distance_calibration"]["ready"]
            return command("save", side_m=0.076, distance_m=0.5)
        if step == 10:
            assert panel.update.call_args.args[0]["distance_calibration"]["corners"] == 0
            return command("measure")
        if step == 18:
            viewer.hardware.set.assert_not_called()
            assert panel.update.call_args.args[0]["distance_calibration"][
                "estimated_m"
            ] == pytest.approx(1)
            return command("apply")
        return []

    panel.poll.side_effect = poll
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q") if step >= 19 else -1)
    assert desktop.main([]) == 0
    viewer.hardware.set.assert_called_once_with("distance", 1.0, True)
    assert panel.update.call_args.args[0]["distance_calibration"]["corners"] == 0
    assert panel.update.call_args.args[0]["distance_calibration"]["estimated_m"] == pytest.approx(1)
    from topdon_duo.settings_preferences import load_settings

    saved = load_settings()
    assert saved["distance_calibration"]["edge_pixels"] == 48
    assert saved["distance_calibration"]["side_m"] == 0.076
    assert saved["hardware"]["distance"] == 1


def test_distance_calibration_commands_are_rejected_while_logging(viewer, monkeypatch):
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.return_value = [
        {"action": "distance_calibration", "operation": operation}
        for operation in ("select", "save", "measure", "apply", "clear")
    ]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.set.assert_not_called()
    assert panel.update.call_args.args[0]["distance_calibration"]["reference"] is None


def test_view_popup_emits_settings_and_syncs_without_feedback(tmp_path):
    script = """
import sys
from pathlib import Path
from PySide6.QtCore import QPoint, QPointF, QSettings, QSize, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton, QScrollArea
from topdon_duo.view_window import ViewWindow
from topdon_duo.view_settings import VIEW_DEFAULTS
from topdon_duo.hardware_controls import HARDWARE_CONTROLS

app = QApplication([])
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                  str(Path(sys.argv[1]).parent / "config"))
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
assert messages == []
assert window.windowTitle() == "Camera"
# Wheel events cannot edit dropdowns, numeric inputs or sliders, even with focus.
for control_row in (*window.rows.values(), *window.hardware_rows.values()):
    for control in (control_row.input, control_row.slider):
        if control is None:
            continue
        control.setFocus()
        before = control_row.value()
        for delta in (120, -120):
            event = QWheelEvent(QPointF(5, 5), QPointF(control.mapToGlobal(QPoint(5, 5))),
                                QPoint(), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                                Qt.NoScrollPhase, False)
            QApplication.sendEvent(control, event)
            assert not event.isAccepted()
            assert control_row.value() == before
        assert not control_row.timer.isActive()
assert messages == []
# Delivered through the window, ignored input events scroll the containing menu.
scroll = window.findChild(QScrollArea)
scroll.verticalScrollBar().setValue(0)
control = window.controls["image_source"]
control.setFocus()
app.processEvents()
position = control.mapTo(window, control.rect().center())
QTest.wheelEvent(window.windowHandle(), position, QPoint(0, -120))
app.processEvents()
assert scroll.verticalScrollBar().value() > 0
assert messages == []
scroll.verticalScrollBar().setValue(0)

assert window.findChildren(QCheckBox) == [window.advanced_auto]
assert window.advanced_auto.text() == "Advanced / Auto"
for row in (*window.rows.values(), *window.hardware_rows.values()):
    assert row.input.isEnabled()
    assert (row.slider is None) == bool(row.options)
    if row.slider is not None:
        assert row.slider.isEnabled()
window.advanced_auto.click()
assert messages.pop() == {"action": "advanced_auto", "value": False}
assert all(row.input.isEnabled() for row in window.rows.values())
assert all(row.input.isEnabled() for row in window.hardware_rows.values())

for name, value in [("image_source", "raw"), ("temperature_unit", "F"),
                    ("image_filter", "median"), ("upsampling", "acnet-legacy-hdn2"),
                    ("upsampling", "anime4k09"), ("enhancement_input", "preview"),
                    ("color_palette", "white_hot"),
                    ("mirror_horizontal", True), ("mirror_vertical", True),
                    ("antialiasing", False)]:
    control = window.controls[name]
    control.setCurrentIndex(control.findData(value))
    window.rows[name]._emit()
    assert messages[-1] == {"action": "setting", "name": name, "value": value}
for name, value in [("enhancement_amount", 0.5), ("anime4k_passes", 2)]:
    window.controls[name].setValue(value)
    window.rows[name]._emit()
    assert messages[-1] == {"action": "setting", "name": name, "value": value}
count = len(messages)
window.update_state({**VIEW_DEFAULTS, "status": "Camera preview"})
assert len(messages) == count
assert window.controls["image_source"].currentData() == "preview"
assert window.controls["mirror_horizontal"].currentData() is False
assert window.controls["mirror_horizontal"].isEnabled()
assert window.status.text() == "Camera preview"
assert set(window.hardware_rows) == set(HARDWARE_CONTROLS)

row = window.hardware_rows["ambient"]
window.update_state({**VIEW_DEFAULTS, "hardware": {
    "ambient": {"value": 30, "enabled": False, "available": True}}})
assert len(messages) == count
assert row.input.value() == 30
assert row.input.isEnabled() and row.slider.isEnabled()
row.input.setValue(27.5)
assert row.slider.value() == 775
row._emit()
assert messages[-1] == {"action": "hardware", "name": "ambient",
                        "value": 27.5, "enabled": True}
row.slider.setValue(800)
assert row.input.value() == 30
window._restore_hardware()
assert messages[-1] == {"action": "restore_hardware"}
assert not row.timer.isActive()

window.update_state({**VIEW_DEFAULTS, "hardware": {
    "palette": {"value": 11.0, "enabled": True, "available": True},
    "ambient": {"value": 30, "enabled": False, "available": True}}})
assert window.hardware_rows["palette"].input.currentData() == 11
assert window.hardware_rows["palette"].slider is None
assert row.input.isEnabled()

# Display conversion must not change hardware values or emit writes on synchronization.
count = len(messages)
state = {**VIEW_DEFAULTS, "temperature_unit": "F", "hardware": {
    "ambient": {"value": 30, "available": True},
    "reflected": {"value": 20, "available": True}}}
window.update_state(state)
assert len(messages) == count
assert row.input.value() == 86
assert row.input.suffix() == " °F"
assert row.input.minimum() == -58 and row.input.maximum() == 212
assert abs(row.input.singleStep() - 0.18) < 1e-8
assert row.slider.value() == 800
assert window.hardware_rows["reflected"].input.value() == 68
assert window.hardware_rows["distance"].input.suffix() == " in"
row.input.setValue(95)
assert row.value() == 35 and row.slider.value() == 850
row._emit()
assert messages[-1] == {"action": "hardware", "name": "ambient",
                        "value": 35.0, "enabled": True}
row.slider.setValue(775)
assert row.input.value() == 81.5 and row.value() == 27.5
# Preserve a pending edit across a unit change without sending a second write.
count = len(messages)
window.update_state({**state, "temperature_unit": "C"})
assert len(messages) == count
assert row.input.value() == 27.5 and row.input.suffix() == " °C"
row._emit()
assert messages[-1]["value"] == 27.5
window.update_state({**state, "temperature_unit": "C"})
assert row.input.value() == 30

# Distance uses inches in imperial mode while hardware commands retain meters.
count = len(messages)
distance = window.hardware_rows["distance"]
metric = {**VIEW_DEFAULTS, "hardware": {"distance": {"value": 1, "available": True}}}
window.update_state(metric)
assert distance.input.value() == 100 and distance.input.suffix() == " cm"
window.update_state({**metric, "temperature_unit": "F"})
assert len(messages) == count
assert abs(distance.input.value() - 39.37008) < 0.005
assert distance.input.suffix() == " in" and distance.input.decimals() == 2
assert abs(distance.input.minimum() - 0.3/0.0254) < 0.005
assert abs(distance.input.maximum() - 99/0.0254) < 0.005
assert abs(distance.input.singleStep() - 0.01/0.0254) < 0.005
assert distance.slider.value() == 70
# User inputs in inches are quantized to the camera's 1 cm precision.
distance.input.setValue(78.74016)
assert distance.value() == 2 and distance.slider.value() == 170
distance._emit()
assert messages[-1] == {"action": "hardware", "name": "distance", "value": 2.0, "enabled": True}
distance.slider.setValue(120)
assert distance.value() == 1.5 and abs(distance.input.value() - 1.5/0.0254) < 0.005
# A pending distance edit survives a unit change without an unintended write.
count = len(messages)
window.update_state(metric)
assert distance.input.value() == 150 and distance.input.suffix() == " cm"
assert len(messages) == count
distance._emit()
assert messages[-1]["value"] == 1.5
# Repeated round trips at both range endpoints must not drift or write hardware.
for value in (0.3, 1, 27.54, 99):
    expected = {**metric, "hardware": {"distance": {"value": value, "available": True}}}
    count = len(messages)
    for _ in range(5):
        window.update_state(expected)
        window.update_state({**expected, "temperature_unit": "F"})
        assert distance.value() == value
        window.update_state(expected)
        assert distance.input.value() == value * 100
    assert len(messages) == count

# Logging locks every settings input and cancels pending debounce timers.
count = len(messages)
for control_row in (*window.rows.values(), *window.hardware_rows.values()):
    control_row.timer.start()
window.update_state({**state, "temperature_unit": "C", "settings_locked": True})
assert len(messages) == count
for control_row in (*window.rows.values(), *window.hardware_rows.values()):
    assert not control_row.input.isEnabled()
    assert control_row.slider is None or not control_row.slider.isEnabled()
    assert not control_row.timer.isActive()
    control_row._emit()
assert not window.advanced_auto.isEnabled()
assert not window.reset_button.isEnabled() and not window.restore_button.isEnabled()
window._reset_display()
window._restore_hardware()
assert len(messages) == count
window.update_state({**state, "temperature_unit": "C", "settings_locked": False})
assert window.advanced_auto.isEnabled()
assert window.reset_button.isEnabled() and window.restore_button.isEnabled()
assert all(control_row.input.isEnabled()
           for control_row in (*window.rows.values(), *window.hardware_rows.values()))
# Unlocking must still respect controls that are unavailable on this camera.
window.update_state({**state, "hardware": {"ambient": {"value": 30, "available": False}}})
assert not row.input.isEnabled()
window.update_state({**state, "temperature_unit": "C"})

# The Post-it calibration works in native units and obeys the logging lock.
cal = window.distance_calibration
cal.update_state({}, "C", False)
assert cal.side.value() == 7.6 and cal.distance.value() == 0
assert not cal.save.isEnabled() and not cal.measure.isEnabled()
cal.side.lineEdit().setText("8.0000")
cal.update_state({}, "C", False)
assert cal.side.lineEdit().text().startswith("8.0000")
cal.side.interpretText()
assert cal.side.value() == 8
profile = {"side_m": 0.076, "distance_m": 0.5, "edge_pixels": 48,
           "native_size": [256, 192], "version": 1}
cal_state = {"reference": profile, "ready": True, "estimated_m": 1,
             "status": "Estimate ready."}
count = len(messages)
cal.update_state(cal_state, "F", False)
assert len(messages) == count
assert cal.side.suffix() == " in"
assert "39.37 in" in cal.status.text()
assert abs(cal.side.value() * 0.0254 - 0.076) < 0.00002
assert abs(cal.distance.value() * 0.0254 - 0.5) < 0.00002
for _ in range(5):
    cal.update_state(cal_state, "C", False)
    assert cal.side.value() == 7.6 and cal.distance.value() == 50
    cal.update_state(cal_state, "F", False)
    assert cal._meters(cal.side) == 0.076 and cal._meters(cal.distance) == 0.5
assert cal.save.isEnabled() and cal.measure.isEnabled() and cal.apply.isEnabled()
cal.save.click()
assert messages[-1]["operation"] == "save"
assert abs(messages[-1]["side_m"] - 0.076) < 0.00002
assert abs(messages[-1]["distance_m"] - 0.5) < 0.00002
cal.apply.click()
assert messages[-1] == {"action": "distance_calibration", "operation": "apply"}
cal.update_state({**cal_state, "estimated_m": 0.25}, "C", False)
assert not cal.apply.isEnabled() and "Outside" in cal.status.text()
assert "25.00 cm" in cal.status.text()
for control in (cal.side, cal.distance):
    before = control.value()
    event = QWheelEvent(QPointF(5, 5), QPointF(control.mapToGlobal(QPoint(5, 5))),
                        QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    QApplication.sendEvent(control, event)
    assert not event.isAccepted() and control.value() == before
count = len(messages)
cal.update_state(cal_state, "C", True)
for control in (cal.side, cal.distance, cal.select, cal.save, cal.measure, cal.apply, cal.clear):
    assert not control.isEnabled()
cal._save()
assert len(messages) == count

body_layout = row.parentWidget().layout()
headings = {body_layout.itemAt(i).widget().text(): i
            for i in range(body_layout.count())
            if isinstance(body_layout.itemAt(i).widget(), QLabel)}
for name in ("upsampling", "enhancement_input", "enhancement_amount", "anime4k_passes"):
    assert headings["AI enhancement"] < body_layout.indexOf(window.rows[name])
    assert body_layout.indexOf(window.rows[name]) < headings["Camera adjustments"]
assert "On / off settings" not in headings
switches = window.display_switches
assert switches.columnCount() == 2
assert headings["Display controls"] < body_layout.indexOf(switches) < headings["AI enhancement"]
expected = [row for row in (*window.rows.values(), *window.hardware_rows.values())
            if row.is_switch]
assert switches.count() == len(expected)
for index, control in enumerate(expected):
    assert switches.itemAtPosition(index // 2, index % 2).widget() is control
    assert control.slider is None
    assert control.input.isVisible()

window.rows["color_palette"].timer.start()
next(b for b in window.findChildren(QPushButton) if b.text().startswith("Reset")).click()
assert messages[-1] == {"action": "reset"}
assert not window.rows["color_palette"].timer.isActive()
app.processEvents()
window.grab().save(sys.argv[1])
window.resize(760, 880)
app.processEvents()
saved_size = window.size()
window.close()
reopened = ViewWindow(messages.append)
assert reopened.size() == saved_size == QSize(760, 880)
reopened.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "view-popup.png")],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("palette", COLOR_PALETTES)
@pytest.mark.parametrize("image_filter", IMAGE_FILTERS)
def test_filters_and_palettes_leave_measurements_unchanged(palette, image_filter):
    frame, _ = frame_with_preview()
    base = ThermalRenderer(scale=2).render_detailed(frame)
    renderer = ThermalRenderer(scale=2)
    renderer.set_view_setting("color_palette", palette)
    renderer.set_view_setting("image_filter", image_filter)
    rendered = renderer.render_detailed(frame)
    assert rendered.image.shape == base.image.shape
    assert rendered.stats == base.stats
    assert np.array_equal(rendered.raw_counts, base.raw_counts)
    assert np.array_equal(rendered.temperatures_celsius, base.temperatures_celsius)
    if palette in ("white_hot", "black_hot"):
        assert np.array_equal(rendered.image[..., 0], rendered.image[..., 1])
        assert np.array_equal(rendered.image[..., 1], rendered.image[..., 2])


def test_antialiasing_and_filters_change_display():
    frame, _ = frame_with_preview()
    renderer = ThermalRenderer(scale=3)
    smooth = renderer.render_detailed(frame)
    renderer.set_view_setting("antialiasing", False)
    nearest = renderer.render_detailed(frame)
    assert not np.array_equal(smooth.image, nearest.image)
    renderer.set_view_setting("image_filter", "gaussian")
    filtered = renderer.render_detailed(frame)
    assert not np.array_equal(filtered.image, nearest.image)
    assert filtered.stats == nearest.stats == smooth.stats


@pytest.mark.parametrize("horizontal,vertical", [(True, False), (False, True), (True, True)])
def test_mirrored_spots_stay_on_physical_pixels_through_rotations(
    viewer, monkeypatch, horizontal, vertical
):
    words = np.frombuffer(make_frame(), dtype="<u2").copy()
    words[HEADER_U16 : HEADER_U16 + SENSOR_PIXELS] = np.arange(SENSOR_PIXELS) + 10_000
    viewer.camera.frames.return_value = [words.tobytes()] * 30
    panel = Mock()
    panel.events = []

    def poll():
        events, panel.events = panel.events, []
        return events

    panel.poll.side_effect = poll
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    observed = []
    original_draw = desktop.draw_sample_spots

    def draw(image, rendered, spots, scale, unit):
        for x, y in spots.pixels:
            observed.append(int(rendered.raw_counts[y, x]))
        return original_draw(image, rendered, spots, scale, unit)

    monkeypatch.setattr(desktop, "draw_sample_spots", draw)
    step = 0

    def key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            return ord("p")
        if step == 2:
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 124, 202 + desktop.toolbar_layout(768).height, 0, None)
        if step == 3:
            panel.events = [
                {"action": "setting", "name": "mirror_horizontal", "value": horizontal},
                {"action": "setting", "name": "mirror_vertical", "value": vertical},
            ]
            return ord("v")
        if step in (5, 6, 7, 8):
            return ord("o")
        if step == 10:
            panel.events = [{"action": "reset"}]
        if step == 12:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    assert observed and set(observed) == {10_000 + 67 * 256 + 41}
    panel.open.assert_called_once()
    panel.close.assert_called_once()
    assert all(
        panel.update.call_args.args[0][name] == value for name, value in VIEW_DEFAULTS.items()
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("mirror_horizontal", "true"),
        ("color_palette", "bad"),
        ("image_filter", 2),
        ("upsampling", "unknown-model"),
        ("enhancement_input", "invalid"),
        ("enhancement_amount", float("nan")),
        ("enhancement_amount", 1.1),
        ("anime4k_passes", 2.5),
        ("anime4k_passes", 0),
        ("anime4k_passes", True),
        ("unknown", True),
    ],
)
def test_invalid_view_settings_are_rejected(name, value):
    renderer = ThermalRenderer()
    with pytest.raises(ValueError):
        renderer.set_view_setting(name, value)
    assert renderer.view_settings() == VIEW_DEFAULTS
