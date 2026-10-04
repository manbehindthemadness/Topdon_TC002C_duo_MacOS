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
    camera = Mock()
    camera.frames.return_value = [make_frame()] * 30
    monkeypatch.setattr(desktop, "TC002CDuoCamera", lambda: camera)
    monkeypatch.setattr(desktop, "mouse_viewport_size", lambda: None)
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
    assert all(tuple(frame[202 + header, 124]) == (80, 255, 80) for frame in frames)
    assert bool(np.all(frames[0][301 + header, 333] == (80, 255, 80))) == capture_cursor
    assert bool(np.all(frames[-1][301 + header, 333] == (80, 255, 80))) == (not capture_cursor)
    assert not np.array_equal(
        viewer.displayed[6][-176, 50], frames[0][400 + header, 50]
    )  # Help excluded.
    viewer.writer.release.assert_called_once()
    viewer.camera.close.assert_called_once()
    viewer.panel.open.assert_called_once()
    viewer.panel.close.assert_called_once()
    assert set(viewer.callbacks) == {desktop.AMBIENT_TRACKBAR}
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
    assert set(viewer.callbacks) == {desktop.AMBIENT_TRACKBAR}


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
    assert desktop.parse_args([]).timelapse_fpm == 60
    assert desktop.parse_args(["--timelapse-fpm", "120"]).timelapse_fpm == 120
    for value in ("0", "-1", "1501"):
        with pytest.raises(SystemExit):
            desktop.parse_args(["--timelapse-fpm", value])
