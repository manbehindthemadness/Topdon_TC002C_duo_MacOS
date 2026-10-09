from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from test_camera import make_frame

from topdon_duo import desktop


def test_main_window_size_is_saved_and_restored_on_next_run(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.window_preferences import load_main_window_size

    monkeypatch.setattr(desktop, "window_resize_size", lambda _name: (930, 710))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    assert load_main_window_size() == (930, 710)
    desktop.cv2.resizeWindow.reset_mock()
    assert desktop.main([]) == 0
    desktop.cv2.resizeWindow.assert_called_once_with(desktop.WINDOW_NAME, 930, 710)


def test_native_close_keeps_last_size_when_window_has_already_disappeared(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.window_preferences import load_main_window_size

    calls = []

    def size(_name: Any) -> Any:
        """
        Size.
        """
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


def test_native_1024_canvas_and_size_reset_preserve_aspect(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.window_preferences import save_main_window_size

    save_main_window_size((930, 710))
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main(["--scale", "4", "--rotate", "0", "--reset-window-size"]) == 0
    height = 768 + desktop.toolbar_layout(1024).height
    desktop.cv2.resizeWindow.assert_called_once_with(desktop.WINDOW_NAME, 1024, height)
    assert viewer.displayed[-1].shape == (height, 1024, 3)


def test_corrupt_main_window_preferences_fall_back_to_image_size(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    viewer: Any, monkeypatch: pytest.MonkeyPatch, mode: Any, capture_cursor: Any
) -> None:
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
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
def test_quitting_finalizes_active_recording(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, mode: Any
) -> None:
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events([mode, -1, -1, "q"]))
    assert desktop.main([]) == 0
    assert len(viewer.writer.frames) == 1
    viewer.writer.release.assert_called_once()


def test_closing_and_reopening_capture_popup_keeps_recording_active(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
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


def test_toolbar_has_single_capture_control_and_no_recording_settings() -> None:
    buttons = desktop.toolbar_layout(768).buttons
    assert "capture" in buttons
    assert "save" not in buttons
    assert (
        not {"video", "timelapse", "capture_cursor", "timelapse_down", "timelapse_up"}
        & buttons.keys()
    )


def test_popup_save_image_data_opens_dialog_and_saves_radiometric_files(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_cancelled_save_dialog_does_not_start_encoder(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer.dialog.selected = None
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, "q"]))
    assert desktop.main([]) == 0
    assert viewer.writer.frames == []
    desktop.cv2.VideoWriter.assert_not_called()


def test_repeat_record_control_cancels_pending_dialog(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", "video", "q"]))
    assert desktop.main([]) == 0
    assert not viewer.dialog.is_open
    assert viewer.dialog.close_calls == 2  # Cancel, then cleanup on quit.
    desktop.cv2.VideoWriter.assert_not_called()


def test_timelapse_number_field_sets_rate_before_recording_and_keeps_it_fixed(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
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


def test_recording_encoder_failure_is_shown_without_closing_viewer(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer.writer.isOpened.return_value = False
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, "q"]))
    assert desktop.main([]) == 0
    assert viewer.writer.frames == []
    viewer.writer.release.assert_called_once()
    assert "Capture failed:" in viewer.draw_toolbar.call_args.kwargs["status"]


def test_recording_write_failure_finalizes_and_keeps_viewer_running(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer.writer.write.side_effect = cv2.error("disk write failed")
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, "q"]))
    assert desktop.main([]) == 0
    viewer.writer.release.assert_called_once()
    assert "Recording failed:" in viewer.draw_toolbar.call_args.kwargs["status"]


def test_camera_disconnect_finalizes_active_recording(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def frames() -> Iterator[bytes]:
        """
        Frames.
        """
        yield from [make_frame()] * 4
        raise desktop.CameraError("camera disconnected")

    viewer.camera.frames.side_effect = frames
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["video", -1, -1, -1]))
    assert desktop.main([]) == 2
    assert len(viewer.writer.frames) == 1
    viewer.writer.release.assert_called_once()
    viewer.camera.close.assert_called_once()


def test_save_dialog_uses_mp4_filename_and_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_timelapse_cli_default_and_validation() -> None:
    assert desktop.parse_args([]).timelapse_fpm is None
    assert desktop.parse_args(["--timelapse-fpm", "120"]).timelapse_fpm == 120
    for value in ("0", "-1", "1501"):
        with pytest.raises(SystemExit):
            desktop.parse_args(["--timelapse-fpm", value])


def test_camera_preferences_survive_restart_and_restore_clears_overrides(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_spots_save_immediately_and_restore_on_restart(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import load_settings

    spots = desktop.SampleSpots(pixels=[(80, 60), (120, 90)])
    spots.rename(2, "Motor")
    spots.set_enabled(2, False)
    spots.clear(1)
    spots.placing = True
    monkeypatch.setattr(desktop, "SampleSpots", lambda: spots)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
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


@pytest.mark.parametrize(
    "cursor,graphs", [(False, False), (True, False), (False, True), (True, True)]
)
def test_capture_checkboxes_save_immediately_and_restore_on_restart(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, cursor: Any, graphs: Any
) -> None:
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"capture_cursor": not cursor, "capture_graphs": not graphs})
    viewer.panel.events = [
        {"action": "cursor", "value": cursor},
        {"action": "graphs", "value": graphs},
    ]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)

    def quit_after_check(_delay: Any) -> Any:
        """
        Quit after check.
        """
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


def test_timelapse_rate_saves_immediately_reloads_and_respects_cli_override(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import load_settings

    viewer.panel.events = [{"action": "rate", "value": 120}]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)

    def quit_after_saved_rate(_delay: Any) -> Any:
        """
        Quit after saved rate.
        """
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


__all__ = ["viewer_fixture"]
