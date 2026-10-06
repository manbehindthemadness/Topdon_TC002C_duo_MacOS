import subprocess
import sys
from unittest.mock import Mock

import cv2
import pytest
from test_capture_panel import popup_environment
from test_desktop_recording import viewer as viewer_fixture

from topdon_duo import desktop

viewer = viewer_fixture


def test_spots_disable_and_clear_keep_other_numbers_and_toggle_preserves_positions():
    spots = desktop.SampleSpots(placing=True)
    for point in ((10, 20), (30, 40), (50, 60)):
        spots.add(point)
    spots.toggle()
    assert not spots.placing and len(spots.pixels) == 3
    spots.set_enabled(2, False)
    assert spots.active == [(1, (10, 20)), (3, (50, 60))]
    spots.rotate_clockwise(192)
    spots.mirror(192, 256, True, False)
    assert spots.numbers == [1, 2, 3] and spots.disabled == {2}
    spots.clear(1)
    assert spots.numbers == [2, 3] and spots.generation == 0
    spots.set_enabled(2, True)
    assert [number for number, _point in spots.active] == [2, 3]
    spots.toggle()
    spots.add((100, 100))
    assert spots.numbers == [2, 3, 4]
    spots.clear()
    assert spots.placing and not spots.pixels and not spots.disabled
    assert spots.generation == 1
    spots.add((5, 5))
    assert spots.active == [(1, (5, 5))]


@pytest.mark.parametrize("number", [True, 0, 2, "1"])
def test_missing_spot_cannot_modify_existing_spots(number):
    spots = desktop.SampleSpots(pixels=[(10, 20)])
    with pytest.raises(ValueError):
        spots.set_enabled(number, False)
    with pytest.raises(ValueError):
        spots.clear(number)
    assert spots.active == [(1, (10, 20))]


def test_context_menu_widgets_send_spot_ids_and_respect_locks():
    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.spots_window import SpotsMenu
app = QApplication([])
commands = []
menu = SpotsMenu(commands.append)
state = {"spots": [{"number": 2, "enabled": False}, {"number": 4, "enabled": True}], "placing": False}
menu.update_state(state)
app.processEvents()
assert menu.isVisible()
assert menu.actions()[0].defaultWidget() is menu.clear_all
assert menu.calibrate_now.isEnabled()
menu.calibrate_now.trigger()
assert commands.pop() == {"action": "calibrate_now"}
assert not menu.checkboxes[2].isChecked() and menu.checkboxes[4].isChecked()
menu.checkboxes[2].click()
assert commands[-1] == {"action": "enable", "spot": 2, "enabled": True}
state["spots"][0]["enabled"] = True
menu.update_state({**state, "placing": True})
assert menu.isVisible()
menu.clear_buttons[4].click()
assert commands[-1] == {"action": "clear", "spot": 4}
menu.clear_all.click()
assert commands[-1] == {"action": "clear_all"}
menu.placing.click()
assert commands[-1] == {"action": "placing", "enabled": False}
menu.update_state({**state, "locked": True})
assert not menu.clear_all.isEnabled() and not menu.placing.isEnabled()
assert not menu.calibrate_now.isEnabled()
assert all(not check.isEnabled() for check in menu.checkboxes.values())
assert all(not button.isEnabled() for button in menu.clear_buttons.values())
menu.update_state({"spots": [], "placing": False})
assert not menu.clear_all.isEnabled() and menu.placing.isEnabled()
menu.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_main_context_menu_preserves_spots_and_graph_ids(viewer, monkeypatch):
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    panel = Mock()
    panel.events = []

    def poll():
        events, panel.events = panel.events, []
        return events

    panel.poll.side_effect = poll
    monkeypatch.setattr(desktop, "SpotsPanel", lambda: panel)
    snapshots = []
    real_draw = desktop.draw_sample_spots

    def draw(image, rendered, spots, scale, unit):
        snapshots.append(
            (spots.placing, spots.numbers.copy(), [number for number, _point in spots.active])
        )
        return real_draw(image, rendered, spots, scale, unit)

    monkeypatch.setattr(desktop, "draw_sample_spots", draw)
    step = 0

    def click(x, y, event=cv2.EVENT_LBUTTONUP):
        viewer.set_mouse.call_args.args[1](
            event, x * 3, y * 3 + desktop.toolbar_layout(768).height, 0, None
        )

    def key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            return ord("p")
        if step == 2:
            click(10, 20)
            click(30, 40)
        if step == 3:
            click(10, 20, cv2.EVENT_RBUTTONUP)
            return ord("p")
        if step == 4:
            panel.events = [
                {"action": "rename", "spot": 2, "name": "Motor"},
                {"action": "enable", "spot": 1, "enabled": False},
            ]
        if step == 5:
            panel.events = [{"action": "clear", "spot": 1}]
        if step == 6:
            return ord("p")
        if step == 7:
            panel.events = [{"action": "enable", "spot": 2, "enabled": False}]
        if step == 8:
            panel.events = [{"action": "enable", "spot": 2, "enabled": True}]
        if step == 9:
            panel.events = [{"action": "clear_all"}]
        if step == 10:
            click(50, 60)
        return ord("q") if step == 12 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    panel.open.assert_called_once()
    assert panel.open.call_args.args[0]["spots"] == [
        {"number": 1, "enabled": True, "name": "Spot 1"},
        {"number": 2, "enabled": True, "name": "Spot 2"},
    ]
    assert snapshots[3] == (False, [1, 2], [1, 2])
    assert snapshots[4][2] == [2]
    assert snapshots[5][1:] == ([2], [2])
    assert snapshots[7][1:] == ([2], [])
    assert snapshots[8][1:] == ([2], [2])
    assert snapshots[-1][1:] == ([1], [1])
    graph_keys = [
        [key for key, _value in call.args[0].spots] for call in viewer.graphs.submit.call_args_list
    ]
    assert graph_keys[5] == [(0, 1)]
    assert viewer.graphs.submit.call_args_list[5].args[0].spot_names == (((0, 1), "Motor"),)
    assert graph_keys[7] == []
    assert graph_keys[-1] == [(1, 0)]
    panel.close.assert_called_once()


@pytest.mark.parametrize("action", ["placing", "enable", "rename", "clear", "clear_all"])
def test_context_menu_cannot_change_spots_while_logging(viewer, monkeypatch, action):
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.return_value = [{"action": action, "spot": 1, "enabled": True, "name": "Hand"}]
    monkeypatch.setattr(desktop, "SpotsPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    assert not panel.update.call_args.args[0]["spots"]
    assert not panel.update.call_args.args[0]["placing"]
    assert panel.update.call_args.args[0]["locked"]


def test_disabled_spots_do_not_draw_and_remaining_labels_keep_numbers():
    import numpy as np
    from test_camera import make_frame

    rendered = desktop.ThermalRenderer(scale=3).render_detailed(make_frame())
    spots = desktop.SampleSpots(pixels=[(10, 20), (30, 40), (50, 60)])
    spots.set_enabled(1, False)
    spots.clear(3)
    labels, _occupied = desktop._spot_label_layout(rendered, spots, 3, "C")
    assert len(labels) == 1 and labels[0][0].startswith("2: ")
    spots.set_enabled(2, False)
    assert np.array_equal(
        desktop.draw_sample_spots(rendered.image, rendered, spots, 3), rendered.image
    )


def test_right_click_is_separate_from_placement_clicks():
    picker = desktop.MousePicker()
    picker.callback(cv2.EVENT_RBUTTONUP, 20, 30, 0, None)
    assert not picker.consume_clicks()
    assert picker.consume_context_clicks() == [(20, 30)]
    assert not picker.consume_context_clicks()


def test_region_name_does_not_change_image_labels_or_spot_identity():
    from test_camera import make_frame

    spots = desktop.SampleSpots(pixels=[(10, 20), (30, 40)])
    rendered = desktop.ThermalRenderer(scale=3).render_detailed(make_frame())
    before = desktop._spot_label_layout(rendered, spots, 3, "C")
    spots.rename(1, " Left hand, thumb ")
    assert spots.name(1) == "Left hand, thumb"
    assert desktop._spot_label_layout(rendered, spots, 3, "C") == before
    spots.set_enabled(1, False)
    spots.rotate_clockwise(192)
    spots.mirror(192, 256, True, False)
    spots.set_enabled(1, True)
    spots.clear(2)
    assert spots.numbers == [1] and spots.name(1) == "Left hand, thumb"
    assert spots.generation == 0
    spots.rename(1, " ")
    assert spots.name(1) == "Spot 1"
    spots.rename(1, "Motor")
    spots.clear()
    assert not spots.names


@pytest.mark.parametrize("name", [None, "a" * 65, "Line\nbreak", "Tab\tname"])
def test_region_name_rejects_invalid_values(name):
    spots = desktop.SampleSpots(pixels=[(10, 20)])
    with pytest.raises(ValueError):
        spots.rename(1, name)
    assert spots.name(1) == "Spot 1"


def test_name_field_saves_and_preserves_pending_edits():
    script = """
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from topdon_duo.spots_window import SpotsMenu
app = QApplication([])
commands = []
menu = SpotsMenu(commands.append)
state = {"spots": [{"number": 1, "name": "Spot 1", "enabled": True}]}
menu.update_state(state)
app.processEvents()
field = menu.name_fields[1]
field.setFocus()
field.selectAll()
QTest.keyClicks(field, "Left hand")
menu.update_state(state)
assert menu.name_fields[1] is field and field.text() == "Left hand"
QTest.keyClick(field, Qt.Key.Key_Return)
assert commands[-1] == {"action": "rename", "spot": 1, "name": "Left hand"}
state["spots"][0]["name"] = "Left hand"
menu.update_state(state)
assert field.text() == "Left hand"
menu.update_state({**state, "locked": True})
assert not field.isEnabled()
menu.update_state(state)
assert field.isEnabled()
field.setFocus()
field.selectAll()
QTest.keyClicks(field, "Motor")
menu.hide()
assert commands[-1] == {"action": "rename", "spot": 1, "name": "Motor"}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_calibration_menu_and_setting_persist_and_obey_logging_lock(viewer, monkeypatch):
    from unittest.mock import call

    from topdon_duo.settings_preferences import load_settings

    panel = Mock()
    settings = Mock()
    monkeypatch.setattr(desktop, "SpotsPanel", lambda: panel)
    monkeypatch.setattr(desktop, "ViewPanel", lambda: settings)
    panel.poll.side_effect = [
        [{"action": "calibrate_now"}],
        [{"action": "calibrate_now"}],
        [],
    ]
    settings.poll.side_effect = [
        [
            {"action": "auto_calibrate", "value": True},
            {"action": "fixed_range", "value": True},
        ],
        [
            {"action": "auto_calibrate", "value": False},
            {"action": "fixed_range", "value": False},
        ],
        [],
    ]
    step = 0

    def wait_key(_):
        nonlocal step
        step += 1
        if step == 1:
            viewer.graphs.logging = True
        if step == 2:
            viewer.graphs.logging = False
        return ord("q") if step == 3 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_: 1)
    assert desktop.main([]) == 0
    assert viewer.hardware.set_auto_calibrate.call_args_list == [call(False), call(True)]
    viewer.hardware.set_fixed_range.assert_called_once_with(True)
    assert load_settings()["fixed_range"] is True
    assert viewer.hardware.calibrate_now.call_count == 2  # Startup plus the manual request.
    viewer.hardware.restore_auto_calibrate.assert_called_once()
    assert load_settings()["auto_calibrate"] is True
    assert settings.update.call_args.args[0]["auto_calibrate"] is True
    # Reload uses the saved switch, rather than replacing it with the default.
    viewer.hardware.set_auto_calibrate.reset_mock()
    viewer.hardware.set_fixed_range.reset_mock()
    viewer.hardware.calibrate_now.reset_mock()
    panel.poll.side_effect = None
    panel.poll.return_value = []
    settings.poll.side_effect = None
    settings.poll.return_value = []
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.set_auto_calibrate.assert_called_once_with(True)
    viewer.hardware.set_fixed_range.assert_called_once_with(True)
    viewer.hardware.calibrate_now.assert_called_once()  # Startup runs on each launch.
