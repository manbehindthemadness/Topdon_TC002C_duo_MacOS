from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from test_camera import make_frame
from test_desktop_recording import viewer as viewer_fixture

from topdon_duo import desktop
from topdon_duo.camera import HEADER_U16, SENSOR_HEIGHT, SENSOR_WIDTH

viewer = viewer_fixture


def mouse(picker, event, x, y, flags=0):
    picker.callback(event, x, y, flags, None)


@pytest.mark.parametrize("placing", [False, True])
@pytest.mark.parametrize("viewport", [None, (384, 303)])
def test_drag_keeps_identity_name_and_grab_offset_without_placing_another_spot(placing, viewport):
    picker = desktop.MousePicker()
    drag = desktop.SpotDrag()
    spots = desktop.SampleSpots(placing=placing, pixels=[(10, 20), (50, 60)])
    spots.rename(1, "Motor")
    shape, scale, toolbar = (576, 768, 3), 3, 30

    def at(event, x, y, flags=0):
        if viewport:
            x, y = round(x / 2), round(y / 2)
        mouse(picker, event, x, y, flags)

    # Grab slightly off-center; no jump when the button first goes down.
    at(cv2.EVENT_LBUTTONDOWN, 33, 92)
    drag.update(picker, spots, shape, scale, viewport, toolbar)
    assert spots.pixels[0] == (10, 20)
    at(cv2.EVENT_MOUSEMOVE, 93, 152, cv2.EVENT_FLAG_LBUTTON)
    drag.update(picker, spots, shape, scale, viewport, toolbar)
    at(cv2.EVENT_LBUTTONUP, 93, 152)
    drag.update(picker, spots, shape, scale, viewport, toolbar)
    assert spots.pixels == [(30, 40), (50, 60)]
    assert spots.numbers == [1, 2] and spots.generation == 0
    assert spots.name(1) == "Motor"
    assert not picker.consume_clicks()


def test_disabled_spots_and_locked_spots_cannot_be_dragged():
    for disabled, locked in ((True, False), (False, True)):
        picker = desktop.MousePicker()
        spots = desktop.SampleSpots(pixels=[(10, 20)])
        if disabled:
            spots.set_enabled(1, False)
        mouse(picker, cv2.EVENT_LBUTTONDOWN, 31, 61)
        mouse(picker, cv2.EVENT_MOUSEMOVE, 91, 121, cv2.EVENT_FLAG_LBUTTON)
        mouse(picker, cv2.EVENT_LBUTTONUP, 91, 121)
        desktop.SpotDrag().update(picker, spots, (576, 768, 3), 3, locked=locked)
        assert spots.pixels == [(10, 20)]


def test_drag_release_outside_image_or_cancelled_never_activates_toolbar_or_adds_spot():
    picker = desktop.MousePicker()
    drag = desktop.SpotDrag()
    spots = desktop.SampleSpots(placing=True, pixels=[(10, 20)])
    mouse(picker, cv2.EVENT_LBUTTONDOWN, 31, 91)
    drag.update(picker, spots, (576, 768, 3), 3, toolbar_height=30)
    mouse(picker, cv2.EVENT_MOUSEMOVE, 91, 151, cv2.EVENT_FLAG_LBUTTON)
    drag.update(picker, spots, (576, 768, 3), 3, toolbar_height=30)
    assert spots.pixels == [(30, 40)]
    drag.cancel()
    mouse(picker, cv2.EVENT_LBUTTONUP, 91, 10)
    drag.update(picker, spots, (576, 768, 3), 3, toolbar_height=30)
    assert not picker.consume_clicks()
    assert spots.pixels == [(30, 40)]


def test_lock_mid_drag_stops_movement_but_swallows_release():
    picker = desktop.MousePicker()
    drag = desktop.SpotDrag()
    spots = desktop.SampleSpots(pixels=[(10, 20)])
    mouse(picker, cv2.EVENT_LBUTTONDOWN, 31, 61)
    drag.update(picker, spots, (576, 768, 3), 3)
    mouse(picker, cv2.EVENT_MOUSEMOVE, 91, 121, cv2.EVENT_FLAG_LBUTTON)
    mouse(picker, cv2.EVENT_LBUTTONUP, 91, 121)
    drag.update(picker, spots, (576, 768, 3), 3, locked=True)
    assert spots.pixels == [(10, 20)]
    assert not picker.consume_clicks()


def test_unlock_does_not_resume_drag_started_before_logging():
    picker = desktop.MousePicker()
    drag = desktop.SpotDrag()
    spots = desktop.SampleSpots(pixels=[(10, 20)])
    mouse(picker, cv2.EVENT_LBUTTONDOWN, 31, 61)
    drag.update(picker, spots, (576, 768, 3), 3)
    drag.update(picker, spots, (576, 768, 3), 3, locked=True)
    assert drag.number is None
    mouse(picker, cv2.EVENT_MOUSEMOVE, 91, 121, cv2.EVENT_FLAG_LBUTTON)
    mouse(picker, cv2.EVENT_LBUTTONUP, 91, 121)
    drag.update(picker, spots, (576, 768, 3), 3)
    assert spots.pixels == [(10, 20)]
    assert not picker.consume_clicks()


def test_empty_space_clicks_still_place_and_nearest_marker_is_selected():
    picker = desktop.MousePicker()
    drag = desktop.SpotDrag()
    spots = desktop.SampleSpots(pixels=[(10, 20), (12, 20)])
    mouse(picker, cv2.EVENT_LBUTTONDOWN, 36, 61)
    drag.update(picker, spots, (576, 768, 3), 3)
    assert drag.number == 2
    mouse(picker, cv2.EVENT_LBUTTONUP, 36, 61)
    drag.update(picker, spots, (576, 768, 3), 3)
    mouse(picker, cv2.EVENT_LBUTTONDOWN, 200, 200)
    mouse(picker, cv2.EVENT_LBUTTONUP, 200, 200)
    drag.update(picker, spots, (576, 768, 3), 3)
    assert picker.consume_clicks() == [(200, 200)]


def test_main_drag_updates_temperature_and_keeps_named_graph_identity(viewer, monkeypatch):
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    words = np.frombuffer(make_frame(), dtype="<u2").copy()
    raw = np.tile(20000 + 64 * np.arange(SENSOR_WIDTH), (SENSOR_HEIGHT, 1))
    words[HEADER_U16 : HEADER_U16 + raw.size] = raw.ravel()
    viewer.camera.frames.return_value = [words.tobytes()] * 15
    panel = Mock()
    panel.poll.side_effect = [[], [], [], [{"action": "rename", "spot": 1, "name": "Motor"}]] + [
        []
    ] * 15
    monkeypatch.setattr(desktop, "SpotsPanel", lambda: panel)
    step = 0

    def event(kind, x, y, flags=0):
        viewer.set_mouse.call_args.args[1](
            kind, x, y + desktop.toolbar_layout(768).height, flags, None
        )

    def key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            return ord("p")
        if step == 2:
            event(cv2.EVENT_LBUTTONUP, 31, 61)
        if step == 3:
            return ord("p")
        if step == 4:
            event(cv2.EVENT_LBUTTONDOWN, 31, 61)
        if step == 5:
            event(cv2.EVENT_MOUSEMOVE, 91, 121, cv2.EVENT_FLAG_LBUTTON)
        if step == 6:
            event(cv2.EVENT_LBUTTONUP, 91, 121)
        return ord("q") if step == 7 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    snapshots = [call.args[0] for call in viewer.graphs.submit.call_args_list]
    before, after = snapshots[3], snapshots[-1]
    assert before.spots[0][0] == after.spots[0][0] == (0, 0)
    assert after.spot_names == (((0, 0), "Motor"),)
    assert after.spots[0][1] - before.spots[0][1] == pytest.approx(20)
    assert len(after.spots) == 1
    assert not panel.update.call_args.args[0]["placing"]


def test_logging_locks_spot_controls_from_file_dialog_until_stopped(viewer, monkeypatch):
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    panel = Mock()
    panel.events = []
    states = []

    def poll():
        events, panel.events = panel.events, []
        return events

    panel.poll.side_effect = poll
    panel.update.side_effect = lambda state: states.append(state)
    monkeypatch.setattr(desktop, "SpotsPanel", lambda: panel)
    pixels = []
    real_draw = desktop.draw_sample_spots

    def draw(image, rendered, spots, scale, unit):
        pixels.append(spots.pixels.copy())
        return real_draw(image, rendered, spots, scale, unit)

    monkeypatch.setattr(desktop, "draw_sample_spots", draw)
    step = 0

    def event(kind, x, y, flags=0):
        viewer.set_mouse.call_args.args[1](
            kind, x, y + desktop.toolbar_layout(768).height, flags, None
        )

    def key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            return ord("p")
        if step == 2:
            event(cv2.EVENT_LBUTTONUP, 31, 61)
        if step == 3:
            event(cv2.EVENT_LBUTTONDOWN, 31, 61)
            return ord("l")
        if step == 4:
            event(cv2.EVENT_MOUSEMOVE, 91, 121, cv2.EVENT_FLAG_LBUTTON)
            panel.events = [
                {"action": "rename", "spot": 1, "name": "Changed"},
                {"action": "enable", "spot": 1, "enabled": False},
                {"action": "clear", "spot": 1},
                {"action": "clear_all"},
                {"action": "placing", "enabled": False},
            ]
            return ord("p")
        if step == 5:
            assert viewer.graphs.logging
            event(cv2.EVENT_LBUTTONUP, 91, 121)
            viewer.click_control("spots")
        if step == 6:
            event(cv2.EVENT_LBUTTONUP, 151, 181)  # No new spot while logging.
        if step == 7:
            return ord("l")
        if step == 8:
            assert not viewer.graphs.logging
            event(cv2.EVENT_LBUTTONDOWN, 31, 61)
            event(cv2.EVENT_MOUSEMOVE, 91, 121, cv2.EVENT_FLAG_LBUTTON)
            event(cv2.EVENT_LBUTTONUP, 91, 121)
        return ord("q") if step == 9 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    expected = [{"number": 1, "enabled": True, "name": "Spot 1"}]
    assert all(state["spots"] == expected and state["placing"] for state in states[3:])
    assert [state["locked"] for state in states] == [False] * 3 + [True] * 4 + [False] * 2
    assert [call.kwargs["spots_locked"] for call in viewer.draw_toolbar.call_args_list] == [
        state["locked"] for state in states
    ]
    snapshots = [call.args[0] for call in viewer.graphs.submit.call_args_list]
    assert all(snapshot.spots == snapshots[3].spots for snapshot in snapshots[4:8])
    assert snapshots[-1].spots[0][0] == (0, 0)
    # Movement resumes after stopping logging, with the same marker identity.
    assert all(points == [(10, 20)] for points in pixels[3:8])
    assert pixels[-1] == [(30, 40)]
