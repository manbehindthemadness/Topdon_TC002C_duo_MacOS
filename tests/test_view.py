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
assert window.hardware_rows["distance"].input.suffix() == " m"
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
