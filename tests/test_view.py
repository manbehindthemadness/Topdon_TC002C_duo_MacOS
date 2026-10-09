import subprocess
import sys
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from test_camera import make_frame
from test_capture_panel import popup_environment
from test_render import frame_with_preview

from topdon_duo import desktop
from topdon_duo.camera import HEADER_U16, SENSOR_PIXELS
from topdon_duo.render import ThermalRenderer
from topdon_duo.view_settings import COLOR_PALETTES, IMAGE_FILTERS, VIEW_DEFAULTS


def test_emissivity_point_selection_hides_markers_and_fits_native_camera_reading(
    viewer, monkeypatch
):
    from topdon_duo.emissivity_calibration import EmissivityCalibrator
    from topdon_duo.settings_preferences import load_settings

    monkeypatch.setattr(EmissivityCalibrator, "SETTLE_SECONDS", 0.2)
    panel = Mock()
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    setting = {"value": 0.95, "enabled": False, "available": True}
    viewer.hardware.state.return_value["emissivity"] = setting

    def set_hardware(name, value, enabled):
        assert name == "emissivity"
        setting.update(value=value if enabled else 0.95, enabled=enabled)

    viewer.hardware.set.side_effect = set_hardware
    words = np.frombuffer(make_frame(), dtype="<u2").copy()

    def frames():
        for _ in range(80):
            temperature = min(550, 20 + 10 / setting["value"])
            words[HEADER_U16 : SENSOR_PIXELS + HEADER_U16] = round((temperature + 50) * 64)
            yield words.tobytes()

    viewer.camera.frames.return_value = frames()
    draw_spots = Mock(wraps=desktop.draw_sample_spots)
    monkeypatch.setattr(desktop, "draw_sample_spots", draw_spots)
    draw_reference = Mock(wraps=desktop.draw_emissivity_point)
    monkeypatch.setattr(desktop, "draw_emissivity_point", draw_reference)
    step = 0
    applied = False

    def click(x, y):
        viewer.set_mouse.call_args.args[1](
            cv2.EVENT_LBUTTONUP, x * 3, y * 3 + desktop.toolbar_layout(768).height, 0, None
        )

    def poll():
        nonlocal step, applied
        step += 1
        if step == 4:
            click(40, 50)
            return [{"action": "emissivity_calibration", "operation": "select"}]
        if step == 5:
            assert panel.update.call_args.args[0]["emissivity_calibration"]["point"] == [40, 50]
            assert panel.update.call_args.args[0]["emissivity_calibration"][
                "selected_celsius"
            ] == pytest.approx(30.53125)
            return [{"action": "emissivity_calibration", "operation": "fit", "known_celsius": 40}]
        if step > 5:
            state = panel.update.call_args.args[0]
            if state["emissivity_calibration"]["result"] is not None and not applied:
                assert setting["value"] == 0.95 and not setting["enabled"]
                assert "emissivity" not in load_settings().get("hardware", {})
                assert load_settings()["emissivity_calibration"]["emissivity"] == 0.5
                assert state["emissivity_calibration"]["point"] is None
                applied = True
                return [{"action": "emissivity_calibration", "operation": "apply"}]
        return []

    panel.poll.side_effect = poll
    hidden_count = None

    def key(_delay):
        nonlocal hidden_count
        viewer.clock[0] += 0.25
        if step == 1:
            return ord("p")
        if step == 2:
            click(30, 40)
        state = panel.update.call_args.args[0]["emissivity_calibration"]
        if state["active"]:
            if hidden_count is None:
                hidden_count = draw_spots.call_count
            assert draw_spots.call_count == hidden_count
        if applied:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    assert applied and load_settings()["hardware"]["emissivity"] == 0.5
    assert draw_spots.call_args.args[2].pixels == [(30, 40)]
    assert draw_spots.call_count > hidden_count
    assert draw_reference.call_count > 0


def test_reset_and_restore_preserve_saved_calibrations_and_allow_reapply(viewer, monkeypatch):
    from topdon_duo.distance_calibration import DistanceReference
    from topdon_duo.settings_preferences import load_settings, save_settings

    distance = DistanceReference(0.076, 0.5, 48).as_dict()
    emissivity = {"version": 1, "emissivity": 0.5, "known_celsius": 40}
    reflected = {"version": 1, "celsius": 24.2}
    save_settings(
        {
            "distance_calibration": distance,
            "emissivity_calibration": emissivity,
            "reflected_calibration": reflected,
            "display": {"mirror_horizontal": True, "temperature_unit": "F"},
            "hardware": {"emissivity": 0.5},
        }
    )
    setting = {"value": 0.95, "enabled": False, "available": True}
    viewer.hardware.state.return_value["emissivity"] = setting
    viewer.hardware.set.side_effect = lambda _name, value, enabled: setting.update(
        value=value, enabled=enabled
    )
    viewer.hardware.restore.side_effect = lambda: setting.update(value=0.95, enabled=False)
    panel = Mock()
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    step = 0

    def poll():
        nonlocal step
        step += 1
        if step > 1:
            saved = load_settings()
            assert saved["distance_calibration"] == distance
            assert saved["emissivity_calibration"] == emissivity
            assert saved["reflected_calibration"] == reflected
            state = panel.update.call_args.args[0]
            assert state["distance_calibration"]["reference"] == distance
            assert state["emissivity_calibration"]["reference"] == emissivity
        if step == 1:
            return [{"action": "reset"}]
        if step == 2:
            return [{"action": "restore_hardware"}]
        if step == 3:
            assert load_settings()["hardware"] == {}
            return [{"action": "emissivity_calibration", "operation": "apply"}]
        return []

    panel.poll.side_effect = poll
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q") if step >= 4 else -1)
    assert desktop.main([]) == 0
    saved = load_settings()
    assert saved["distance_calibration"] == distance
    assert saved["emissivity_calibration"] == emissivity
    assert saved["hardware"]["emissivity"] == 0.5


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


def test_emissivity_calibration_commands_are_rejected_while_logging(viewer, monkeypatch):
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.return_value = [
        {"action": "emissivity_calibration", "operation": operation}
        for operation in ("select", "fit", "apply")
    ]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.set.assert_not_called()
    assert not panel.update.call_args.args[0]["emissivity_calibration"]["active"]


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
assert window.operation_status.isHidden()
def check_notification_before_send(command):
    assert not window.operation_status.isHidden()
    assert "Applying detail enhancement mode" in window.operation_status.text()
    assert "may pause briefly" in window.operation_status.text()
window._send_to_viewer = check_notification_before_send
window._send({"action": "hardware", "name": "detail_enabled", "value": 0, "enabled": True})
window._send_to_viewer = messages.append
window.update_state({**VIEW_DEFAULTS, "camera_operation": "Applying detail enhancement mode…",
                     "camera_operation_busy": True})
assert not window.auto_calibrate.isEnabled()
assert window.pipeline_editor.locked
window.update_state({**VIEW_DEFAULTS, "camera_operation": "Detail enhancement mode updated"})
assert window.operation_status.text() == "Detail enhancement mode updated"
window.update_state({**VIEW_DEFAULTS, "camera_operation": "Camera setting rejected: timed out"})
assert "rejected" in window.operation_status.text()
window.update_state(VIEW_DEFAULTS)
assert window.operation_status.isHidden()
assert window.windowTitle() == "Camera"
assert not hasattr(window, "advanced_auto")
assert not hasattr(window, "analyze_widgets")
row = window.hardware_rows["ambient"]
reflected = window.hardware_rows["reflected"]
transmission = window.hardware_rows["transmission"]
body_layout = reflected.parentWidget().layout()
assert body_layout.indexOf(transmission) == body_layout.indexOf(reflected) + 1
window.update_state({**VIEW_DEFAULTS, "image_source": "raw", "actual_image_source": "raw", "hardware": {"transmission": {"value": 95, "available": True}}})
assert transmission.input.isEnabled() and transmission.input.value() == 95
assert transmission.input.suffix() == " %"
transmission.input.setValue(90)
transmission._emit()
assert messages[-1] == {"action": "hardware", "name": "transmission", "value": 90, "enabled": True}
window.update_state({**VIEW_DEFAULTS, "settings_locked": True})
assert not transmission.input.isEnabled()
state = {**VIEW_DEFAULTS, "temperature_unit": "F", "hardware": {
    "ambient": {"value": 30, "available": True},
    "reflected": {"value": 20, "available": True}}}
window.update_state(state)
assert row.input.value() == 86
assert row.input.suffix() == " °F"
row.input.setValue(72)
row._emit()
assert abs(messages[-1]["value"] - 22.2222222222) < 1e-9
window.update_state({**state, "hardware": {"ambient": {"value": messages[-1]["value"], "available": True}}})
assert row.input.value() == 72
for control in window.findChildren(QCheckBox):
    before = control.isChecked()
    event = QWheelEvent(QPointF(5, 5), QPointF(control.mapToGlobal(QPoint(5, 5))), QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(control, event)
    assert control.isChecked() == before
count = len(messages)
window.update_state({**state, "settings_locked": True})
assert window.pipeline_editor.locked
assert not row.input.isEnabled() and not window.restore_button.isEnabled()
window._restore_hardware()
assert len(messages) == count
window.update_state(state)

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

# Known temperature follows the selected units; fit uses canonical Celsius.
em = window.emissivity_calibration
em_state = {"active": True, "point": [40, 50], "measured_celsius": 30}
count = len(messages)
em.update_state(em_state, "F", False)
assert em.known.value() == 68 and em.known.suffix() == " °F"
assert em.fit.isEnabled() and "86.00 °F" in em.status.text()
assert len(messages) == count
em.known.setValue(95)
em.fit.click()
assert messages[-1] == {"action": "emissivity_calibration", "operation": "fit", "known_celsius": 35}
em.update_state(em_state, "C", False)
assert em.known.value() == 35
event = QWheelEvent(QPointF(5, 5), QPointF(em.known.mapToGlobal(QPoint(5, 5))),
                    QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
QApplication.sendEvent(em.known, event)
assert not event.isAccepted() and em.known.value() == 35
em.update_state({**em_state, "running": True}, "C", True)
assert not em.known.isEnabled() and not em.fit.isEnabled() and not em.select.isEnabled()
assert not em.apply.isEnabled() and em.cancel.isEnabled()
em.cancel.click()
assert messages[-1] == {"action": "emissivity_calibration", "operation": "cancel"}
em.update_state({**em_state, "result": 0.5}, "C", False)
assert em.apply.isEnabled()
em.apply.click()
assert messages[-1] == {"action": "emissivity_calibration", "operation": "apply"}
# A point selection imports its temperature once, even if the same pixel is reselected.
selected_em = {**em_state, "selection_id": 1, "selected_celsius": 31.25, "known_celsius": 31.25}
em.update_state(selected_em, "F", False)
assert em.known.value() == 88.25
em.known.setValue(95)
em.update_state({**selected_em, "measured_celsius": 32}, "F", False)
assert em.known.value() == 95
em.update_state({**selected_em, "selection_id": 2, "selected_celsius": 29, "known_celsius": 29}, "C", False)
assert em.known.value() == 29
saved_em = {"version": 1, "emissivity": 0.5, "known_celsius": 40}
em.update_state({"reference": saved_em}, "F", False)
assert em.apply.isEnabled() and em.apply.text() == "Apply saved emissivity"
assert em.known.value() == 104 and "Saved emissivity: 0.50" in em.status.text()

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


def test_reflector_workflow_restores_then_applies_saved_native_reading(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings

    panel = Mock()
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    settings = {
        "emissivity": {"value": 0.8, "enabled": True, "available": True},
        "transmission": {"value": 95, "enabled": False, "available": True},
        "reflected": {"value": 22, "enabled": False, "available": True},
    }
    viewer.hardware.state.return_value.update(settings)
    viewer.hardware.set.side_effect = lambda name, value, enabled: settings[name].update(
        value=value, enabled=enabled
    )
    words = np.frombuffer(make_frame(), dtype="<u2").copy()
    words[HEADER_U16 : SENSOR_PIXELS + HEADER_U16] = round((24.5 + 50) * 64)
    viewer.camera.frames.return_value = iter([words.tobytes()] * 100)
    step = 0
    applied = False
    target = Mock(wraps=desktop.draw_reflector_target)
    monkeypatch.setattr(desktop, "draw_reflector_target", target)

    def poll():
        nonlocal step, applied
        step += 1
        if step == 1:
            return [{"action": "reflected_calibration", "operation": "select"}]
        if step == 2:
            return [{"action": "reflected_calibration", "operation": "measure"}]
        state = panel.update.call_args.args[0]["reflected_calibration"]
        if state["reference"] and not applied:
            assert not state["active"]
            assert settings["emissivity"]["value"] == 0.8
            assert settings["transmission"]["value"] == 95
            assert load_settings()["reflected_calibration"]["celsius"] == 24.5
            assert "reflected" not in load_settings()["hardware"]
            applied = True
            return [{"action": "reflected_calibration", "operation": "apply"}]
        return []

    panel.poll.side_effect = poll

    def key(_delay):
        viewer.clock[0] += 0.25
        return ord("q") if applied else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    assert applied and target.call_count > 0
    assert load_settings()["hardware"]["reflected"] == 24.5


def test_reflector_controls_units_and_measurement_lock():
    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ReflectedCalibrationControls
app = QApplication([])
commands = []
controls = ReflectedCalibrationControls(commands.append)
controls.select.click()
assert commands == [{"action": "reflected_calibration", "operation": "select"}]
state = {"active": True, "running": True, "reference": {"version": 1, "celsius": 20}, "status": "Measuring"}
controls.update_state(state, "F", True)
assert "68.0 °F" in controls.status.text()
assert not controls.select.isEnabled() and not controls.measure.isEnabled() and not controls.apply.isEnabled()
assert controls.cancel.isEnabled()
controls.cancel.click()
assert commands[-1]["operation"] == "cancel"
controls.update_state({"reference": state["reference"]}, "C", False)
assert controls.apply.isEnabled() and not controls.measure.isEnabled() and not controls.cancel.isEnabled()
assert "20.0 °C" in controls.status.text()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_startup_loads_all_calibration_references_and_applied_camera_values(viewer, monkeypatch):
    from topdon_duo.distance_calibration import DistanceReference
    from topdon_duo.settings_preferences import load_settings, save_settings

    references = {
        "distance_calibration": DistanceReference(0.076, 0.5, 48).as_dict(),
        "emissivity_calibration": {"version": 1, "emissivity": 0.5, "known_celsius": 40},
        "reflected_calibration": {"version": 1, "celsius": 24.2},
    }
    applied = {"distance": 1.2, "emissivity": 0.5, "reflected": 24.2}
    save_settings({**references, "hardware": applied})
    settings = viewer.hardware.state.return_value
    for name in applied:
        settings[name] = {"value": 0, "enabled": False, "available": True}
    viewer.hardware.set.side_effect = lambda name, value, enabled: settings[name].update(
        value=value, enabled=enabled
    )
    panel = Mock()
    panel.poll.return_value = []
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    state = panel.update.call_args.args[0]
    for name, reference in references.items():
        assert state[name]["reference"] == reference
        assert load_settings()[name] == reference
    for name, value in applied.items():
        viewer.hardware.set.assert_any_call(name, value, True)
        assert state["hardware"][name]["value"] == value
        assert state["hardware"][name]["enabled"]
    assert load_settings()["hardware"] == applied


def test_processing_preset_is_saved_and_reloaded_after_hardware_settings(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"hardware": {"contrast": 70}, "processing_preset": "shadow"})
    panel = Mock()
    panel.poll.return_value = [{"action": "processing_preset", "value": "soft"}]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    viewer.hardware.set_processing_preset.side_effect = lambda value: setattr(
        viewer.hardware, "processing_preset", value
    )
    assert desktop.main([]) == 0
    assert [call.args for call in viewer.hardware.set_processing_preset.call_args_list] == [
        ("shadow",), ("soft",)
    ]
    calls = viewer.hardware.mock_calls
    assert calls.index(next(c for c in calls if c[0] == "set")) < calls.index(
        next(c for c in calls if c[0] == "set_processing_preset")
    )
    assert load_settings()["processing_preset"] == "soft"


def test_processing_preset_command_is_locked_during_logging(viewer, monkeypatch):
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.return_value = [{"action": "processing_preset", "value": "shadow"}]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.set_processing_preset.assert_not_called()


def test_tone_selection_reloads_after_hardware_and_presets_and_persists(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({'hardware': {'contrast': 70}, 'processing_preset': 'shadow',
                   'camera_gamma': 25, 'camera_boost': 3})
    panel = Mock()
    panel.poll.return_value = [{'action': 'tone', 'gamma': 75}]
    monkeypatch.setattr(desktop, 'ViewPanel', lambda: panel)
    monkeypatch.setattr(desktop.cv2, 'waitKey', lambda _: ord('q'))

    def apply(gamma, boost):
        viewer.hardware.gamma, viewer.hardware.boost = gamma, boost

    viewer.hardware.set_tone.side_effect = apply
    assert desktop.main([]) == 0
    assert [call.args for call in viewer.hardware.set_tone.call_args_list] == [(25, 3), (75, 3)]
    calls = viewer.hardware.mock_calls
    assert calls.index(next(c for c in calls if c[0] == 'set_processing_preset')) < calls.index(
        next(c for c in calls if c[0] == 'set_tone'))
    assert load_settings()['camera_gamma'] == 75 and load_settings()['camera_boost'] == 3


def test_tone_commands_are_locked_during_logging(viewer, monkeypatch):
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.return_value = [{'action': 'tone', 'gamma': 25}, {'action': 'cancel_tone'}]
    monkeypatch.setattr(desktop, 'ViewPanel', lambda: panel)
    monkeypatch.setattr(desktop.cv2, 'waitKey', lambda _: ord('q'))
    assert desktop.main([]) == 0
    viewer.hardware.set_tone.assert_not_called()
    viewer.hardware.restore_tone.assert_not_called()


@pytest.mark.parametrize('fails', [False, True])
def test_camera_operation_notifies_before_usb_work_and_reports_outcome(viewer, monkeypatch, fails):
    from topdon_duo.camera import CameraError

    panel = Mock()
    panel.poll.return_value = [
        {'action': 'hardware', 'name': 'detail_enabled', 'value': 0, 'enabled': True}
    ]
    monkeypatch.setattr(desktop, 'ViewPanel', lambda: panel)
    viewer.hardware.state.return_value = {
        'ambient': {'value': 30, 'available': True},
        'detail_enabled': {'value': 0, 'available': True},
    }

    def apply(*_args):
        state = panel.update.call_args.args[0]
        assert state['camera_operation_busy']
        assert 'Applying detail enhancement mode' in state['camera_operation']
        if fails:
            raise CameraError('test timeout')

    viewer.hardware.set.side_effect = apply
    monkeypatch.setattr(desktop.cv2, 'waitKey', lambda _delay: ord('q'))
    assert desktop.main([]) == 0
    states = [call.args[0] for call in panel.update.call_args_list]
    expected = 'Camera setting rejected: test timeout' if fails else 'Detail enhancement mode updated'
    assert any(state['camera_operation'] == expected and not state['camera_operation_busy']
               for state in states)


__all__ = ["viewer_fixture"]
