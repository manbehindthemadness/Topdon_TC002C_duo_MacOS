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
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton
from topdon_duo.view_window import ViewWindow
from topdon_duo.view_settings import VIEW_DEFAULTS
from topdon_duo.hardware_controls import HARDWARE_CONTROLS

app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
app.processEvents()
assert messages == []
assert window.windowTitle() == "Camera"
assert window.findChildren(QCheckBox) == [window.advanced_auto]
assert window.advanced_auto.text() == "Advanced / Auto"
for row in (*window.rows.values(), *window.hardware_rows.values()):
    assert row.input.isEnabled()
    if row.slider is not None:
        assert row.slider.isEnabled()
window.advanced_auto.click()
assert not messages
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
assert window.hardware_rows["palette"].slider.value() == 3
assert row.input.isEnabled()

body_layout = row.parentWidget().layout()
switch_heading = next(body_layout.itemAt(i).widget() for i in range(body_layout.count())
                      if isinstance(body_layout.itemAt(i).widget(), QLabel)
                      and body_layout.itemAt(i).widget().text() == "On / off settings")
for control in (*window.rows.values(), *window.hardware_rows.values()):
    if control.is_switch:
        assert body_layout.indexOf(control) > body_layout.indexOf(switch_heading)
        assert control.slider is None
    else:
        assert body_layout.indexOf(control) < body_layout.indexOf(switch_heading)

window.rows["color_palette"].timer.start()
next(b for b in window.findChildren(QPushButton) if b.text().startswith("Reset")).click()
assert messages[-1] == {"action": "reset"}
assert not window.rows["color_palette"].timer.isActive()
app.processEvents()
window.grab().save(sys.argv[1])
window.close()
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
