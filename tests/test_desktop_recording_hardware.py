from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture
from test_camera import make_frame

from topdon_duo import desktop


def test_hardware_ambient_changes_and_restore_drive_measurements(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_startup_calibration_waits_for_valid_frame_and_runs_once(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_measurement_validity import frozen_frame

    viewer.camera.frames.return_value = [frozen_frame(), make_frame(0), make_frame(), make_frame()]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    counts = []

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        counts.append(viewer.hardware.calibrate_now.call_count)
        return ord("q") if len(counts) == 4 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert counts == [0, 0, 1, 1]
    viewer.hardware.set_auto_calibrate.assert_called_once_with(False)


def test_missing_frames_keep_ui_active_and_exclude_stale_measurements(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    viewer.camera.frames.return_value = [make_frame(), None, None, make_frame(24000)]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        viewer.clock[0] += 1
        return ord("q") if step == 4 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    snapshots = [call.args[0] for call in viewer.graphs.submit.call_args_list]
    assert [snapshot.measurements_valid for snapshot in snapshots] == [True, False, False, True]
    assert snapshots[-1].stats[0] == 325
    assert len(viewer.displayed) == 4


def test_viewer_can_quit_while_waiting_for_first_frame(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer.camera.frames.return_value = [None] * 10
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.calibrate_now.assert_not_called()
    viewer.camera.close.assert_called_once()
    assert len(viewer.displayed) == 1


def test_hardware_settings_retry_after_frames_arrive_before_applying_values(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import save_settings

    save_settings({"hardware": {"ambient": 22.2}})
    viewer.hardware.load.side_effect = [
        desktop.CameraError("Camera not ready"),
        desktop.CameraError("Camera not ready"),
        None,
    ]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    counts = []

    def frames() -> Iterator[bytes]:
        """
        Frames.
        """
        assert viewer.hardware.load.call_count == 1
        viewer.hardware.set_auto_calibrate.assert_not_called()
        viewer.hardware.set.assert_not_called()
        yield make_frame()
        yield make_frame()

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        counts.append(viewer.hardware.calibrate_now.call_count)
        viewer.clock[0] += 1.1
        return ord("q") if len(counts) == 2 else -1

    viewer.camera.frames.side_effect = frames
    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert viewer.hardware.load.call_count == 3
    viewer.hardware.set.assert_called_once_with("ambient", 22.2, True)
    viewer.hardware.set_auto_calibrate.assert_called_once_with(False)
    assert counts == [0, 1]


def test_hardware_startup_retries_are_bounded_and_do_not_send_calibration(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer.hardware.load.side_effect = desktop.CameraError("Unsupported control layout")
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        viewer.clock[0] += 11
        return ord("q") if step == 6 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert viewer.hardware.load.call_count == 5
    viewer.hardware.set_auto_calibrate.assert_not_called()
    viewer.hardware.calibrate_now.assert_not_called()


def test_unsupported_hardware_protocol_does_not_retry_or_block_stream(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer.hardware.load.side_effect = desktop.HardwareProtocolError(
        "Unsupported camera protocol version layout (length reply 0002)"
    )
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events([-1, -1, "q"]))
    assert desktop.main([]) == 0
    viewer.hardware.load.assert_called_once()
    assert len(viewer.displayed) == 3
    viewer.hardware.set_auto_calibrate.assert_not_called()


def test_startup_calibration_failure_keeps_viewer_running_without_retries(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    viewer.hardware.calibrate_now.side_effect = desktop.CameraError("Calibration rejected")
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events([-1, -1, "q"]))
    assert desktop.main([]) == 0
    viewer.hardware.calibrate_now.assert_called_once()
    assert len(viewer.displayed) == 3
    assert "Could not run startup calibration: Calibration rejected" in caplog.text
    viewer.camera.close.assert_called_once()


def test_startup_calibration_is_not_requested_without_valid_camera_data(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_measurement_validity import frozen_frame

    viewer.camera.frames.return_value = [frozen_frame()]
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.calibrate_now.assert_not_called()


def test_ambient_input_keeps_requested_value_but_sends_quantized_temperature(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
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


@pytest.mark.parametrize("camera_error", [False, True])
def test_display_awake_request_matches_camera_lifetime(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, camera_error: Any
) -> None:
    awake = Mock()
    awake.check.return_value = None
    monkeypatch.setattr(desktop, "DisplayAwake", lambda: awake)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    if camera_error:
        viewer.camera.open.side_effect = desktop.CameraError("Connection failed")
    assert desktop.main([]) == (2 if camera_error else 0)
    assert awake.start.call_count == (0 if camera_error else 1)
    awake.close.assert_called_once()
    viewer.camera.close.assert_called_once()


__all__ = ["viewer_fixture"]
