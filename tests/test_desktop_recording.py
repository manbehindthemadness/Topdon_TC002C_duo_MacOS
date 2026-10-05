from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from test_camera import make_frame

from topdon_duo import desktop


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    hardware = Mock(original={"loaded": True}, enabled=set(), error="", preview_active=False)
    hardware.state.return_value = {"ambient": {"value": 30.0, "available": True}}
    monkeypatch.setattr(desktop, "HardwareControls", lambda _camera: hardware)
    graphs = Mock(logging=False)
    graphs.take_logging_error.return_value = None

    def start_logging(path):
        graphs.logging = True
        return path.with_suffix(".csv")

    def stop_logging():
        graphs.logging = False
        return dialog.selected.with_suffix(".csv")

    graphs.start_logging.side_effect = start_logging
    graphs.stop_logging.side_effect = stop_logging
    graphs.image.side_effect = lambda size, **_kwargs: np.zeros((size[1], size[0], 3), np.uint8)
    monkeypatch.setattr(desktop, "GraphWorker", lambda: graphs)
    camera = Mock()
    camera.frames.return_value = [make_frame()] * 30
    monkeypatch.setattr(desktop, "TC002CDuoCamera", lambda: camera)
    monkeypatch.setattr(desktop, "mouse_viewport_size", lambda: None)
    monkeypatch.setattr(desktop, "window_resize_size", lambda _name: None)
    pointer = Mock()
    pointer.over_image.return_value = True
    monkeypatch.setattr(desktop, "PointerMonitor", lambda _name: pointer)
    panel = Mock()
    panel.events = []
    panel.state = {"capture_cursor": False, "frames_per_minute": 60}

    def poll_panel():
        events, panel.events = panel.events, []
        return events

    panel.poll.side_effect = poll_panel
    panel.update.side_effect = lambda state: setattr(panel, "state", state.copy())
    monkeypatch.setattr(desktop, "CapturePanel", lambda: panel)
    for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "setTrackbarPos"):
        monkeypatch.setattr(desktop.cv2, name, Mock())
    callbacks = {}
    monkeypatch.setattr(
        desktop.cv2,
        "createTrackbar",
        lambda name, _window, _value, _maximum, callback: callbacks.update({name: callback}),
    )
    set_mouse = Mock()
    monkeypatch.setattr(desktop.cv2, "setMouseCallback", set_mouse)
    displayed = []
    monkeypatch.setattr(desktop.cv2, "imshow", lambda _name, image: displayed.append(image.copy()))
    clock = [0.0]
    monkeypatch.setattr(desktop.time, "monotonic", lambda: clock[0])
    writer = Mock()
    writer.isOpened.return_value = True
    writer.frames = []
    writer.write.side_effect = lambda image: writer.frames.append(image.copy())
    monkeypatch.setattr(desktop.cv2, "VideoWriter", Mock(return_value=writer))

    class Dialog:
        def __init__(self):
            self.is_open = False
            self.polls = 0
            self.selected = tmp_path / "test recording.mp4"
            self.open_calls = []
            self.close_calls = 0

        def open(self, directory, **options):
            self.open_calls.append((directory, options))
            self.is_open = True
            self.polls = 0
            return True

        def poll(self):
            if not self.is_open:
                return False, None
            self.polls += 1
            if self.polls == 1:
                return False, None
            self.is_open = False
            return True, self.selected

        def close(self):
            self.close_calls += 1
            self.is_open = False

    dialog = Dialog()
    monkeypatch.setattr(desktop, "LinuxSaveDialog", lambda: dialog)
    draw_toolbar = Mock(wraps=desktop.draw_toolbar)
    monkeypatch.setattr(desktop, "draw_toolbar", draw_toolbar)

    def click_control(action):
        layout = desktop.toolbar_layout(768)
        callback = set_mouse.call_args.args[1]
        if action in ("video", "timelapse"):
            panel.events.append(
                {
                    "action": action,
                    "frames_per_minute": panel.state["frames_per_minute"],
                    "capture_cursor": panel.state["capture_cursor"],
                }
            )
        elif action == "capture_cursor":
            panel.events.append({"action": "cursor", "value": not panel.state["capture_cursor"]})
        elif action == "image":
            panel.events.append({"action": "image"})
        else:
            x0, y0, x1, y1 = layout.buttons[action]
            callback(cv2.EVENT_LBUTTONUP, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        callback(cv2.EVENT_MOUSEMOVE, 333, 301 + layout.height, 0, None)

    def key_events(actions):
        actions = iter(actions)

        def wait_key(_delay):
            action = next(actions)
            if action in ("video", "timelapse", "image"):
                click_control(action)
                return -1
            return ord(action) if isinstance(action, str) else action

        return wait_key

    return SimpleNamespace(
        graphs=graphs,
        hardware=hardware,
        camera=camera,
        writer=writer,
        dialog=dialog,
        clock=clock,
        callbacks=callbacks,
        set_mouse=set_mouse,
        displayed=displayed,
        click_control=click_control,
        draw_toolbar=draw_toolbar,
        panel=panel,
        key_events=key_events,
    )


def test_main_window_size_is_saved_and_restored_on_next_run(viewer, monkeypatch):
    from topdon_duo.window_preferences import load_main_window_size

    monkeypatch.setattr(desktop, "window_resize_size", lambda _name: (930, 710))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    assert load_main_window_size() == (930, 710)
    desktop.cv2.resizeWindow.reset_mock()
    assert desktop.main([]) == 0
    desktop.cv2.resizeWindow.assert_called_once_with(desktop.WINDOW_NAME, 930, 710)


def test_native_close_keeps_last_size_when_window_has_already_disappeared(viewer, monkeypatch):
    from topdon_duo.window_preferences import load_main_window_size

    calls = []

    def size(_name):
        calls.append(1)
        if len(calls) == 1:
            return (870, 660)
        return None

    monkeypatch.setattr(desktop, "window_resize_size", size)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: int(len(calls) == 1))
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: -1)
    assert desktop.main([]) == 0
    assert len(calls) == 2
    assert load_main_window_size() == (870, 660)


def test_corrupt_main_window_preferences_fall_back_to_image_size(viewer, monkeypatch, tmp_path):
    path = tmp_path / "config/topdon-duo/main-window.json"
    path.parent.mkdir(parents=True)
    path.write_text("invalid json")
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    layout = desktop.toolbar_layout(768)
    desktop.cv2.resizeWindow.assert_called_once_with(desktop.WINDOW_NAME, 768, 576 + layout.height)


@pytest.mark.parametrize("mode", ["video", "timelapse"])
@pytest.mark.parametrize("capture_cursor", [False, True])
def test_recording_controls_save_spots_and_optional_cursor(
    viewer, monkeypatch, mode, capture_cursor
):
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        viewer.clock[0] += 1.0 if mode == "timelapse" else 0.04
        if step == 1:
            viewer.click_control("capture")
            return ord("p")
        if step == 2:
            callback = viewer.set_mouse.call_args.args[1]
            layout = desktop.toolbar_layout(768)
            callback(cv2.EVENT_LBUTTONUP, 123, 201 + layout.height, 0, None)
            callback(cv2.EVENT_MOUSEMOVE, 333, 301 + layout.height, 0, None)
        elif step == 3:
            if capture_cursor:
                viewer.click_control("capture_cursor")
            return ord(" ")  # Help panel must stay out of the recorded video.
        elif step == 4:
            viewer.click_control(mode)
        elif step == 7:
            assert viewer.writer.frames  # Started after two nonblocking dialog polls.
            viewer.click_control("capture_cursor")
        elif step == 9:
            viewer.click_control(mode)
        elif step == 11:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert viewer.dialog.open_calls == [(None, {"suffix": ".mp4", "kind": mode})]
    frames = viewer.writer.frames
    assert len(frames) >= 3
    header = desktop.READOUT_HEIGHT
    assert all(frame.shape == (576 + header, 768, 3) for frame in frames)  # Toolbar excluded.
    thermal = desktop.ThermalRenderer(scale=3).render_detailed(make_frame()).image
    assert all(
        np.array_equal(frame[202 + header, 124], 255 - thermal[202, 124]) for frame in frames
    )
    cursor_color = 255 - thermal[301, 333]
    assert np.array_equal(frames[0][301 + header, 333], cursor_color) == capture_cursor
    assert np.array_equal(frames[-1][301 + header, 333], cursor_color) == (not capture_cursor)
    assert not np.array_equal(
        viewer.displayed[6][-176, 50], frames[0][400 + header, 50]
    )  # Help excluded.
    viewer.writer.release.assert_called_once()
    viewer.camera.close.assert_called_once()
    viewer.panel.open.assert_called_once()
    viewer.panel.close.assert_called_once()
    assert set(viewer.callbacks) == set()
    desktop.cv2.namedWindow.assert_called_once_with(
        desktop.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
    )


@pytest.mark.parametrize("mode", ["video", "timelapse"])
def test_quitting_finalizes_active_recording(viewer, monkeypatch, mode):
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events([mode, -1, -1, "q"]))
    assert desktop.main([]) == 0
    assert len(viewer.writer.frames) == 1
    viewer.writer.release.assert_called_once()


def test_closing_and_reopening_capture_popup_keeps_recording_active(viewer, monkeypatch):
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        viewer.clock[0] += 0.04
        if step == 1:
            viewer.click_control("video")
        elif step == 3:
            viewer.panel.close()  # A popup close is not a recording stop command.
        elif step == 4:
            return ord("c")
        elif step == 5:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert len(viewer.writer.frames) == 3
    assert viewer.panel.open.call_args.args[0]["recording_mode"] == "video"
    viewer.writer.release.assert_called_once()


def test_toolbar_has_single_capture_control_and_no_recording_settings():
    buttons = desktop.toolbar_layout(768).buttons
    assert "capture" in buttons
    assert "save" not in buttons
    assert (
        not {"video", "timelapse", "capture_cursor", "timelapse_down", "timelapse_up"}
        & buttons.keys()
    )


def test_popup_save_image_data_opens_dialog_and_saves_radiometric_files(viewer, monkeypatch):
    viewer.dialog.selected = viewer.dialog.selected.with_suffix(".png")
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["image", -1, -1, "q"]))
    assert desktop.main([]) == 0
    assert viewer.dialog.open_calls == [(None, {})]
    path = viewer.dialog.selected
    assert path.exists()
    assert path.with_suffix(".json").exists()
    with np.load(path.with_suffix(".npz")) as data:
        assert data["raw_counts"].shape == (192, 256)
        assert data["temperatures_celsius"].shape == (192, 256)
    desktop.cv2.VideoWriter.assert_not_called()


def test_cancelled_save_dialog_does_not_start_encoder(viewer, monkeypatch):
    viewer.dialog.selected = None
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, "q"]))
    assert desktop.main([]) == 0
    assert viewer.writer.frames == []
    desktop.cv2.VideoWriter.assert_not_called()


def test_repeat_record_control_cancels_pending_dialog(viewer, monkeypatch):
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", "video", "q"]))
    assert desktop.main([]) == 0
    assert not viewer.dialog.is_open
    assert viewer.dialog.close_calls == 2  # Cancel, then cleanup on quit.
    desktop.cv2.VideoWriter.assert_not_called()


def test_timelapse_number_field_sets_rate_before_recording_and_keeps_it_fixed(viewer, monkeypatch):
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            viewer.panel.events.append({"action": "rate", "value": 120})
            viewer.panel.events.append(
                {"action": "timelapse", "frames_per_minute": 120, "capture_cursor": False}
            )
        if step == 3:
            viewer.panel.events.append({"action": "rate", "value": 600})
        if step == 4:
            viewer.clock[0] = 0.5
        if step == 5:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert len(viewer.writer.frames) == 2  # 120/min means one frame every 0.5 s.
    assert viewer.panel.state["frames_per_minute"] == 120
    assert set(viewer.callbacks) == set()


def test_recording_encoder_failure_is_shown_without_closing_viewer(viewer, monkeypatch):
    viewer.writer.isOpened.return_value = False
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, "q"]))
    assert desktop.main([]) == 0
    assert viewer.writer.frames == []
    viewer.writer.release.assert_called_once()
    assert "Capture failed:" in viewer.draw_toolbar.call_args.kwargs["status"]


def test_recording_write_failure_finalizes_and_keeps_viewer_running(viewer, monkeypatch):
    viewer.writer.write.side_effect = cv2.error("disk write failed")
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, "q"]))
    assert desktop.main([]) == 0
    viewer.writer.release.assert_called_once()
    assert "Recording failed:" in viewer.draw_toolbar.call_args.kwargs["status"]


def test_camera_disconnect_finalizes_active_recording(viewer, monkeypatch):
    def frames():
        yield from [make_frame()] * 4
        raise desktop.CameraError("camera disconnected")

    viewer.camera.frames.side_effect = frames
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, -1]))
    assert desktop.main([]) == 2
    assert len(viewer.writer.frames) == 1
    viewer.writer.release.assert_called_once()
    viewer.camera.close.assert_called_once()


def test_save_dialog_uses_mp4_filename_and_filter(monkeypatch, tmp_path):
    process = Mock()
    popen = Mock(return_value=process)
    monkeypatch.setattr(desktop.subprocess, "Popen", popen)
    dialog = desktop.LinuxSaveDialog()
    assert dialog.open(tmp_path, suffix=".mp4", kind="timelapse")
    command = popen.call_args.args[0]
    assert "--file-filter=MP4 videos | *.mp4" in command
    assert any("TC002C-Duo-timelapse-" in arg and arg.endswith(".mp4") for arg in command)
    mac_command = desktop.MacSaveDialog()._command("video.mp4", str(tmp_path))
    assert "Save thermal recording" in mac_command[2]
    assert mac_command[-2:] == ["video.mp4", str(tmp_path)]


def test_timelapse_cli_default_and_validation():
    assert desktop.parse_args([]).timelapse_fpm is None
    assert desktop.parse_args(["--timelapse-fpm", "120"]).timelapse_fpm == 120
    for value in ("0", "-1", "1501"):
        with pytest.raises(SystemExit):
            desktop.parse_args(["--timelapse-fpm", value])


def test_hardware_ambient_changes_and_restore_drive_measurements(viewer, monkeypatch, tmp_path):
    from test_hardware_controls import CalibrationDevice as Device

    from topdon_duo.hardware_controls import HardwareControls

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    viewer.camera.device = Device()
    hardware = HardwareControls(viewer.camera)
    monkeypatch.setattr(desktop, "HardwareControls", lambda _camera: hardware)
    panel = Mock()
    panel.poll.side_effect = [
        [],
        [{"action": "hardware", "name": "ambient", "value": 35.0, "enabled": True}],
        [{"action": "restore_hardware"}],
    ]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    # Simulate corrected counts delivered by the camera following an ambient write.
    viewer.camera.frames.return_value = [make_frame(4500), make_frame(4600), make_frame(4500)]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events([-1, -1, "q"]))
    assert desktop.main(["--ambient", "99"]) == 0
    calls = viewer.draw_toolbar.call_args_list
    assert [call.args[1] for call in calls] == [30.0, 35.0, 30.0]
    assert [call.kwargs["stats"].average for call in calls] == [
        4500 / 64 - 50,
        4600 / 64 - 50,
        4500 / 64 - 50,
    ]
    assert hardware.state()["ambient"]["value"] == 30.0
    assert viewer.callbacks == {}
    assert "ambient_up" not in desktop.toolbar_layout(768).buttons
    assert "ambient_down" not in desktop.toolbar_layout(768).buttons
    assert panel.update.call_args.args[0]["temperature_conversion"] == "camera"


def test_camera_preferences_survive_restart_and_restore_clears_overrides(
    viewer, monkeypatch, tmp_path
):
    from test_hardware_controls import CalibrationDevice as Device

    from topdon_duo.hardware_controls import HardwareControls
    from topdon_duo.settings_preferences import load_settings

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    device = Device()
    viewer.camera.device = device
    monkeypatch.setattr(desktop, "HardwareControls", HardwareControls)
    panel = Mock()
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    panel.poll.return_value = [
        {"action": "setting", "name": "temperature_unit", "value": "F"},
        {"action": "setting", "name": "mirror_horizontal", "value": True},
        {"action": "setting", "name": "image_filter", "value": "median"},
        {"action": "advanced_auto", "value": False},
        {"action": "hardware", "name": "ambient", "value": 35, "enabled": True},
    ]
    assert desktop.main(["--rotate", "90"]) == 0
    saved = load_settings()
    assert saved["hardware"] == {"ambient": 35}
    assert saved["rotation"] == 90
    # Exit restores the physical camera baseline, without forgetting the override.
    assert HardwareControls(viewer.camera).read(3, 1)[76:80] == (13000).to_bytes(4, "little")
    panel.poll.return_value = []
    assert desktop.main([]) == 0
    state = panel.update.call_args.args[0]
    assert state["temperature_unit"] == "F"
    assert state["mirror_horizontal"] is True
    assert state["image_filter"] == "median"
    assert state["advanced_auto"] is False
    assert state["hardware"]["ambient"]["value"] == 35
    assert viewer.draw_toolbar.call_args.args[1:3] == (35, "F")
    # Explicit CLI options take precedence over remembered display/rotation settings.
    panel.poll.return_value = [{"action": "restore_hardware"}, {"action": "reset"}]
    assert desktop.main(["--rotate", "0", "--image-source", "raw"]) == 0
    saved = load_settings()
    assert saved["hardware"] == {} and saved["display"] == desktop.VIEW_DEFAULTS
    assert saved["rotation"] == 0
    panel.poll.return_value = []
    assert desktop.main([]) == 0
    assert panel.update.call_args.args[0]["hardware"]["ambient"]["value"] == 30


def test_show_graph_doubles_window_width_and_hides_back_to_original(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings
    from topdon_duo.window_preferences import save_main_window_size

    save_main_window_size((930, 710))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step <= 2:
            viewer.click_control("graph")
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert [call.args[1:] for call in desktop.cv2.resizeWindow.call_args_list] == [
        (930, 710),
        (1860, 710),
        (930, 710),
    ]
    assert [image.shape[1] for image in viewer.displayed] == [768, 1536, 768]
    assert np.count_nonzero(viewer.displayed[1][34:, 768:]) == 0
    assert load_settings()["show_graph"] is False


def test_show_graph_survives_restart_without_doubling_again(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings
    from topdon_duo.window_preferences import load_main_window_size

    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("g"))
    viewer.camera.frames.return_value = [make_frame()]
    assert desktop.main([]) == 0
    assert load_settings()["show_graph"] is True
    size = load_main_window_size()
    assert size[0] == 1536
    desktop.cv2.resizeWindow.reset_mock()
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    desktop.cv2.resizeWindow.assert_called_once_with(desktop.WINDOW_NAME, *size)
    assert viewer.displayed[-1].shape[1] == 1536
    assert np.count_nonzero(viewer.displayed[-1][34:, 768:]) == 0


def test_graph_area_does_not_sample_thermal_pixels(viewer, monkeypatch):
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    height = 576 + desktop.toolbar_layout(768).height
    monkeypatch.setattr(desktop, "mouse_viewport_size", lambda: (1536, height))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    samples = []
    draw_picker = desktop.draw_picker

    def sample(*args, **kwargs):
        image, pixel = draw_picker(*args, **kwargs)
        samples.append(pixel)
        return image, pixel

    monkeypatch.setattr(desktop, "draw_picker", sample)
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        callback = viewer.set_mouse.call_args.args[1]
        if step == 1:
            callback(cv2.EVENT_MOUSEMOVE, 1000, 180 + height - 576, 0, None)
        elif step == 2:
            callback(cv2.EVENT_MOUSEMOVE, 240, 180 + height - 576, 0, None)
        else:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert samples == [None, None, (80, 60)]


def test_graph_submission_uses_frame_orientation_and_pauses_when_hidden(viewer, monkeypatch):
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            return ord("p")
        if step == 2:
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 750, 570 + desktop.toolbar_layout(768).height, 0, None)
            viewer.click_control("rotate")
            return -1
        if step == 3:
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    data = viewer.graphs.submit.call_args.args[0]
    assert data.spots == (((0, 0), 20000 / 64 - 50),)
    assert data.stats == (20000 / 64 - 50,) * 4
    viewer.graphs.close.assert_called_once()
    viewer.graphs.submit.reset_mock()
    viewer.graphs.pause.reset_mock()
    save_settings({"show_graph": False})
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    viewer.graphs.submit.assert_not_called()
    viewer.graphs.pause.assert_called_once()


def test_graph_logging_button_locks_hiding_until_logging_stops(viewer, monkeypatch):
    from topdon_duo.graphs import graph_log_button_rect
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def click_log():
        x0, y0, x1, y1 = graph_log_button_rect(768)
        callback = viewer.set_mouse.call_args.args[1]
        callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            click_log()
            viewer.click_control("graph")  # Locked while choosing the logfile.
        elif step == 4:
            assert viewer.graphs.logging
            viewer.click_control("graph")  # Toolbar route must be locked as well as G.
            return ord("g")
        elif step == 5:
            assert viewer.graphs.logging
            click_log()
        elif step == 6:
            assert not viewer.graphs.logging
            viewer.click_control("graph")
        elif step == 7:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert viewer.dialog.open_calls == [(None, {"suffix": ".csv", "kind": "temperatures"})]
    viewer.graphs.start_logging.assert_called_once_with(viewer.dialog.selected)
    viewer.graphs.stop_logging.assert_called_once()
    assert [image.shape[1] for image in viewer.displayed] == [1536] * 6 + [768]
    assert len(desktop.cv2.resizeWindow.call_args_list) == 2
    assert any(call.kwargs["graph_locked"] for call in viewer.draw_toolbar.call_args_list)


def test_cancel_graph_log_dialog_unlocks_graphs(viewer, monkeypatch):
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    # L starts choosing a file; pressing L again cancels before a path is selected.
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["l", "l", "g", "q"]))
    assert desktop.main([]) == 0
    viewer.graphs.start_logging.assert_not_called()
    assert viewer.displayed[-1].shape[1] == 768


def test_graph_logging_button_maps_resized_viewport():
    from topdon_duo.graphs import graph_log_button_rect

    x0, y0, x1, y1 = graph_log_button_rect(768)
    for scale in (0.5, 1, 1.5):
        viewport = (768 * scale, 650 * scale)  # Thermal half of combined viewport.
        assert desktop.graph_logging_button_at(
            round((768 + (x0 + x1) / 2) * scale), round((y0 + y1) / 2 * scale), 768, 650, viewport
        )
        assert not desktop.graph_logging_button_at(20, 20, 768, 650, viewport)


def test_logging_blocks_camera_settings_toolbar_and_shortcuts_then_unlocks(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.side_effect = [
        [
            {"action": "setting", "name": "mirror_horizontal", "value": True},
            {"action": "hardware", "name": "ambient", "value": 35, "enabled": True},
            {"action": "restore_hardware"},
            {"action": "reset"},
            {"action": "advanced_auto", "value": False},
        ],
        [],
        [],
        [],
        [],
    ]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            for action in ("unit", "rotate", "spots"):
                viewer.click_control(action)
            return ord("f")
        if step == 2:
            return ord("o")
        if step == 3:
            return ord("p")
        if step == 4:
            viewer.graphs.logging = False
            return ord("f")  # Works again after logging stops.
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    viewer.hardware.set.assert_not_called()
    viewer.hardware.restore.assert_called_once()  # Normal exit restoration only.
    saved = load_settings()
    assert saved["rotation"] == 0
    assert saved["display"]["mirror_horizontal"] is False
    assert saved["display"]["temperature_unit"] == "F"
    assert saved["advanced_auto"] is True
    assert saved["hardware"] == {}
    assert [call.args[0]["settings_locked"] for call in panel.update.call_args_list] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert [call.kwargs["settings_locked"] for call in viewer.draw_toolbar.call_args_list] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert all(not call.args[0].spots for call in viewer.graphs.submit.call_args_list)


@pytest.mark.parametrize("mode", ["video", "timelapse"])
@pytest.mark.parametrize("include,visible", [(False, True), (True, False), (True, True)])
def test_optional_graph_recording_uses_visible_pane_and_keeps_dimensions(
    viewer, monkeypatch, mode, include, visible
):
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    from topdon_duo.graphs import GraphWorker

    cache = GraphWorker()
    cache.close()
    # Use the real cache lookup: the live pane includes the toolbar's height,
    # while saved images include the temperature readout instead.
    cache._image = np.full(
        (576 + desktop.toolbar_layout(768).height, 768, 3), (12, 34, 56), dtype=np.uint8
    )
    viewer.graphs.image.side_effect = cache.image
    step = 0

    def key(_delay):
        nonlocal step
        step += 1
        viewer.clock[0] += 1 if mode == "timelapse" else 0.04
        if step == 1 and visible:
            return ord("g")
        if step == 2:
            viewer.panel.events.extend(
                [
                    {"action": "graphs", "value": include},
                    {
                        "action": mode,
                        "frames_per_minute": 60,
                        "capture_cursor": False,
                        "capture_graphs": include,
                    },
                ]
            )
        if step == 7:
            viewer.panel.events.append({"action": "graphs", "value": False})
            if visible:
                return ord("g")
        return ord("q") if step == 10 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    frames = viewer.writer.frames
    assert frames
    included = include and visible
    assert all(
        frame.shape == (576 + desktop.READOUT_HEIGHT, 1536 if included else 768, 3)
        for frame in frames
    )
    if included:
        assert np.all(frames[0][:, 768:] == (12, 34, 56))
        assert not frames[-1][:, 768:].any()


@pytest.mark.parametrize("visible", [False, True])
def test_image_graph_capture_preserves_native_radiometric_data(viewer, monkeypatch, visible):
    import json

    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    from topdon_duo.graphs import GraphWorker

    cache = GraphWorker()
    cache.close()
    # Use the real cache lookup: the live pane includes the toolbar's height,
    # while saved images include the temperature readout instead.
    cache._image = np.full(
        (576 + desktop.toolbar_layout(768).height, 768, 3), (12, 34, 56), dtype=np.uint8
    )
    viewer.graphs.image.side_effect = cache.image
    viewer.dialog.selected = viewer.dialog.selected.with_suffix(".png")
    step = 0

    def key(_delay):
        nonlocal step
        step += 1
        if step == 1 and visible:
            return ord("g")
        if step == 2:
            viewer.panel.events.extend([{"action": "graphs", "value": True}, {"action": "image"}])
        return ord("q") if step == 6 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    image = cv2.imread(str(viewer.dialog.selected))
    assert image.shape == (576 + desktop.READOUT_HEIGHT, 1536 if visible else 768, 3)
    if visible:
        assert np.all(image[:, 768:] == (12, 34, 56))
    metadata = json.loads(viewer.dialog.selected.with_suffix(".json").read_text())
    assert metadata["graphs_included"] is visible
    with np.load(viewer.dialog.selected.with_suffix(".npz")) as data:
        assert data["raw_counts"].shape == (192, 256)
        assert data["temperatures_celsius"].shape == (192, 256)


def test_graph_interval_field_applies_and_remembers_value(viewer, monkeypatch):
    from topdon_duo.graphs import graph_interval_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True, "graph_interval": 2})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    keys = iter([-1, ord("0"), ord("."), ord("2"), ord("5"), 13, ord("q")])
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_interval_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        return next(keys)

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert [call.args for call in viewer.graphs.set_interval.call_args_list] == [(2,), (0.25,)]
    assert load_settings()["graph_interval"] == 0.25


def test_graph_interval_field_is_locked_during_logging(viewer, monkeypatch):
    from topdon_duo.graphs import graph_interval_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    viewer.graphs.logging = True
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_interval_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    viewer.graphs.set_interval.assert_called_once_with(0.5)
    assert load_settings()["graph_interval"] == 0.5


def test_graph_interval_field_maps_resized_viewport():
    from topdon_duo.graphs import graph_interval_rect

    x0, y0, x1, y1 = graph_interval_rect(768)
    for scale in (0.5, 1, 1.5):
        assert desktop.graph_interval_at(
            round((768 + (x0 + x1) / 2) * scale),
            round((y0 + y1) / 2 * scale),
            768,
            650,
            (768 * scale, 650 * scale),
        )
        assert not desktop.graph_interval_at(20, 20, 768, 650, (768 * scale, 650 * scale))


def test_startup_calibration_waits_for_valid_frame_and_runs_once(viewer, monkeypatch):
    from test_measurement_validity import frozen_frame

    viewer.camera.frames.return_value = [frozen_frame(), make_frame(0), make_frame(), make_frame()]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    counts = []

    def wait_key(_delay):
        counts.append(viewer.hardware.calibrate_now.call_count)
        return ord("q") if len(counts) == 4 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert counts == [0, 0, 1, 1]
    viewer.hardware.set_auto_calibrate.assert_called_once_with(False)


def test_startup_calibration_failure_keeps_viewer_running_without_retries(
    viewer, monkeypatch, caplog
):
    viewer.hardware.calibrate_now.side_effect = desktop.CameraError("Calibration rejected")
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events([-1, -1, "q"]))
    assert desktop.main([]) == 0
    viewer.hardware.calibrate_now.assert_called_once()
    assert len(viewer.displayed) == 3
    assert "Could not run startup calibration: Calibration rejected" in caplog.text
    viewer.camera.close.assert_called_once()


def test_startup_calibration_is_not_requested_without_valid_camera_data(viewer, monkeypatch):
    from test_measurement_validity import frozen_frame

    viewer.camera.frames.return_value = [frozen_frame()]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.calibrate_now.assert_not_called()


@pytest.mark.parametrize("event_scale", [None, 0.5])
def test_resized_graph_window_keeps_spots_and_controls_aligned(viewer, monkeypatch, event_scale):
    from topdon_duo.graphs import graph_interval_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    native_height = 576 + desktop.toolbar_layout(768).height
    window = [1536, native_height]
    spots = desktop.SampleSpots(pixels=[(80, 60)])
    monkeypatch.setattr(desktop, "SampleSpots", lambda: spots)
    monkeypatch.setattr(desktop, "window_resize_size", lambda _name: tuple(window))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(
        desktop,
        "mouse_viewport_size",
        lambda: tuple(value * event_scale for value in window) if event_scale else None,
    )
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        callback = viewer.set_mouse.call_args.args[1]
        if step == 1:
            window[:] = [2400, 1200]
            layout = desktop.GraphWindowLayout.fit((768, native_height), tuple(window))
            factor = event_scale or 1
            scale_x = layout.camera_size[0] / 768 * factor
            scale_y = layout.camera_size[1] / native_height * factor
            toolbar = desktop.toolbar_layout(768).height
            callback(
                cv2.EVENT_LBUTTONDOWN,
                round(240 * scale_x),
                round((180 + toolbar) * scale_y),
                0,
                None,
            )
            callback(
                cv2.EVENT_MOUSEMOVE,
                round(270 * scale_x),
                round((210 + toolbar) * scale_y),
                cv2.EVENT_FLAG_LBUTTON,
                None,
            )
            callback(
                cv2.EVENT_LBUTTONUP, round(270 * scale_x), round((210 + toolbar) * scale_y), 0, None
            )
        elif step == 2:
            assert spots.pixels == [(90, 70)]
            layout = desktop.GraphWindowLayout.fit((768, native_height), tuple(window))
            x0, y0, x1, y1 = graph_interval_rect(layout.graph_size[0])
            factor = event_scale or 1
            callback(
                cv2.EVENT_LBUTTONUP,
                round((layout.camera_size[0] + (x0 + x1) / 2) * factor),
                round((y0 + y1) / 2 * factor),
                0,
                None,
            )
        elif step == 3:
            return ord("2")
        elif step == 4:
            return 13
        elif step == 5:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert viewer.displayed[0].shape[:2] == (native_height, 1536)
    assert all(image.shape[:2] == (1200, 2400) for image in viewer.displayed[1:])
    assert viewer.graphs.submit.call_args.args[0].size == (1200, 1200)
    assert spots.pixels == [(90, 70)]
    assert load_settings()["graph_interval"] == 2


def test_spots_save_immediately_and_restore_on_restart(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings

    spots = desktop.SampleSpots(pixels=[(80, 60), (120, 90)])
    spots.rename(2, "Motor")
    spots.set_enabled(2, False)
    spots.clear(1)
    spots.placing = True
    monkeypatch.setattr(desktop, "SampleSpots", lambda: spots)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            spots.move(2, (125, 95))
            return -1
        assert load_settings()["spots"]["items"][0]["x"] == 125
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    saved = load_settings()["spots"]
    assert saved["items"] == [{"number": 2, "x": 125, "y": 95, "name": "Motor", "enabled": False}]
    restored = type(spots)()
    monkeypatch.setattr(desktop, "SampleSpots", lambda: restored)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    assert restored.saved_state() == saved
    assert restored.next_number == 3


def test_ambient_input_keeps_requested_value_but_sends_quantized_temperature(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings

    panel = Mock()
    panel.poll.side_effect = [
        [{"action": "hardware", "name": "ambient", "value": 22.2222222222, "enabled": True}]
    ]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    setting = viewer.hardware.state.return_value["ambient"]
    viewer.hardware.set.side_effect = lambda name, value, enabled: setting.update(value=value)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.set.assert_called_once_with("ambient", 22.2, True)
    assert panel.update.call_args.args[0]["hardware"]["ambient"]["value"] == 22.2222222222
    assert viewer.draw_toolbar.call_args.args[1] == 22.2
    assert load_settings()["ambient_input_celsius"] == 22.2222222222
    viewer.hardware.set.reset_mock()
    panel.poll.side_effect = None
    panel.poll.return_value = []
    assert desktop.main([]) == 0
    viewer.hardware.set.assert_called_once_with("ambient", 22.2, True)
    assert panel.update.call_args.args[0]["hardware"]["ambient"]["value"] == 22.2222222222


def test_graph_configuration_button_settings_persist_and_lock(viewer, monkeypatch):
    from topdon_duo.graph_settings import GRAPH_DEFAULTS
    from topdon_duo.graphs import graph_config_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    panel = Mock()
    panel.poll.return_value = []
    monkeypatch.setattr(desktop, "GraphPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    settings = {
        **GRAPH_DEFAULTS,
        "range_mode": "fixed",
        "range_seconds": 600,
        "history_points": 8192,
    }
    step = 0

    def key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_config_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
            panel.poll.return_value = [{"action": "settings", "settings": settings}]
        elif step == 2:
            assert load_settings()["graph_settings"] == settings
            viewer.graphs.logging = True
            panel.poll.return_value = [{"action": "settings", "settings": GRAPH_DEFAULTS}]
        elif step == 3:
            assert load_settings()["graph_settings"] == settings
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    panel.open.assert_called_once()
    assert panel.update.call_args.args[0]["locked"]
    assert [call.args[0] for call in viewer.graphs.configure.call_args_list] == [
        GRAPH_DEFAULTS,
        settings,
    ]
    panel.close.assert_called_once()


@pytest.mark.parametrize("logging", [False, True])
def test_graph_reset_button_only_clears_chart_history_and_obeys_logging_lock(
    viewer, monkeypatch, logging
):
    from topdon_duo.graphs import graph_reset_rect
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    spots = desktop.SampleSpots(pixels=[(80, 60)])
    spots.rename(1, "Motor")
    monkeypatch.setattr(desktop, "SampleSpots", lambda: spots)
    viewer.graphs.logging = logging
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def key(_delay):
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_reset_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    assert viewer.graphs.clear_history.call_count == (0 if logging else 1)
    assert spots.pixels == [(80, 60)] and spots.name(1) == "Motor"
    viewer.graphs.configure.assert_called_once()


@pytest.mark.parametrize(
    "cursor,graphs", [(False, False), (True, False), (False, True), (True, True)]
)
def test_capture_checkboxes_save_immediately_and_restore_on_restart(
    viewer, monkeypatch, cursor, graphs
):
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"capture_cursor": not cursor, "capture_graphs": not graphs})
    viewer.panel.events = [
        {"action": "cursor", "value": cursor},
        {"action": "graphs", "value": graphs},
    ]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)

    def quit_after_check(_delay):
        settings = load_settings()
        assert settings["capture_cursor"] is cursor
        assert settings["capture_graphs"] is graphs
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", quit_after_check)
    assert desktop.main([]) == 0
    assert viewer.panel.state["capture_cursor"] is cursor
    assert viewer.panel.state["capture_graphs"] is graphs
    # A fresh launch must initialize both the UI and capture behavior from preferences.
    assert desktop.main([]) == 0
    assert viewer.panel.state["capture_cursor"] is cursor
    assert viewer.panel.state["capture_graphs"] is graphs
    assert viewer.panel.state["recording_mode"] is None


def test_timelapse_rate_saves_immediately_reloads_and_respects_cli_override(viewer, monkeypatch):
    from topdon_duo.settings_preferences import load_settings

    viewer.panel.events = [{"action": "rate", "value": 120}]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)

    def quit_after_saved_rate(_delay):
        assert load_settings()["timelapse_fpm"] == 120
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", quit_after_saved_rate)
    assert desktop.main([]) == 0
    assert viewer.panel.state["frames_per_minute"] == 120
    assert desktop.main([]) == 0
    assert viewer.panel.state["frames_per_minute"] == 120
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main(["--timelapse-fpm", "240"]) == 0
    assert viewer.panel.state["frames_per_minute"] == 240
    assert load_settings()["timelapse_fpm"] == 240
