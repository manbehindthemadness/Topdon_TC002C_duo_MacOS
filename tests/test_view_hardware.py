"""
Imaging hardware startup, preset persistence, locking and operation notifications.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture

from based_thermals import desktop


def test_startup_loads_all_calibration_references_and_applied_camera_values(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Restore saved calibration references and hardware corrections during startup.
    """
    from based_thermals.distance_calibration import DistanceReference
    from based_thermals.settings_preferences import load_settings, save_settings

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
    viewer.hardware.set.side_effect = lambda control_name, control_value, enabled: settings[
        control_name
    ].update(value=control_value, enabled=enabled)
    panel = Mock()
    panel.poll.return_value = []
    monkeypatch.setattr("based_thermals.desktop.ViewPanel", lambda: panel)
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


def test_processing_preset_is_saved_and_reloaded_after_hardware_settings(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Persist a processing preset and restore it after loading the hardware baseline.
    """
    from based_thermals.settings_preferences import load_settings, save_settings

    save_settings({"hardware": {"contrast": 70}, "processing_preset": "shadow"})
    panel = Mock()
    panel.poll.return_value = [{"action": "processing_preset", "value": "soft"}]
    monkeypatch.setattr("based_thermals.desktop.ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    viewer.hardware.set_processing_preset.side_effect = lambda value: setattr(
        viewer.hardware, "processing_preset", value
    )
    assert desktop.main([]) == 0
    assert [call.args for call in viewer.hardware.set_processing_preset.call_args_list] == [
        ("shadow",),
        ("soft",),
    ]
    calls = viewer.hardware.mock_calls
    assert calls.index(next(c for c in calls if c[0] == "set")) < calls.index(
        next(c for c in calls if c[0] == "set_processing_preset")
    )
    assert load_settings()["processing_preset"] == "soft"


def test_processing_preset_command_is_locked_during_logging(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reject processing preset changes while temperature logging locks hardware.
    """
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.return_value = [{"action": "processing_preset", "value": "shadow"}]
    monkeypatch.setattr("based_thermals.desktop.ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.set_processing_preset.assert_not_called()


def test_tone_selection_reloads_after_hardware_and_presets_and_persists(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Restore tone selection after hardware and preset initialization and retain edits.
    """
    from based_thermals.settings_preferences import load_settings, save_settings

    save_settings(
        {
            "hardware": {"contrast": 70},
            "processing_preset": "shadow",
            "camera_gamma": 25,
            "camera_boost": 3,
        }
    )
    panel = Mock()
    panel.poll.return_value = [{"action": "tone", "gamma": 75}]
    monkeypatch.setattr("based_thermals.desktop.ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))

    def apply(gamma: int, boost: int) -> None:
        """
        Apply the requested tone values to the fake hardware owner.
        """
        viewer.hardware.gamma, viewer.hardware.boost = gamma, boost

    viewer.hardware.set_tone.side_effect = apply
    assert desktop.main([]) == 0
    assert [call.args for call in viewer.hardware.set_tone.call_args_list] == [(25, 3), (75, 3)]
    calls = viewer.hardware.mock_calls
    assert calls.index(next(c for c in calls if c[0] == "set_processing_preset")) < calls.index(
        next(c for c in calls if c[0] == "set_tone")
    )
    assert load_settings()["camera_gamma"] == 75 and load_settings()["camera_boost"] == 3


def test_tone_commands_are_locked_during_logging(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep tone selection and cancellation locked during temperature logging.
    """
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.return_value = [{"action": "tone", "gamma": 25}, {"action": "cancel_tone"}]
    monkeypatch.setattr("based_thermals.desktop.ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _: ord("q"))
    assert desktop.main([]) == 0
    viewer.hardware.set_tone.assert_not_called()
    viewer.hardware.restore_tone.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_camera_operation_notifies_before_usb_work_and_reports_outcome(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    """
    Publish operation progress before hardware writes and report success or rejection.
    """
    from based_thermals.camera import CameraError

    panel = Mock()
    panel.poll.return_value = [
        {"action": "hardware", "name": "detail_enabled", "value": 0, "enabled": True}
    ]
    monkeypatch.setattr("based_thermals.desktop.ViewPanel", lambda: panel)
    viewer.hardware.state.return_value = {
        "ambient": {"value": 30, "available": True},
        "detail_enabled": {"value": 0, "available": True},
    }

    def apply(*_args: Any) -> None:
        """
        Verify the progress notice before simulating a completed or failed write.
        """
        state = panel.update.call_args.args[0]
        assert state["camera_operation_busy"]
        assert "Applying detail enhancement mode" in state["camera_operation"]
        if fails:
            raise CameraError("test timeout")

    viewer.hardware.set.side_effect = apply
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    states = [call.args[0] for call in panel.update.call_args_list]
    expected = (
        "Camera setting rejected: test timeout" if fails else "Detail enhancement mode updated"
    )
    assert any(
        state["camera_operation"] == expected and not state["camera_operation_busy"]
        for state in states
    )


__all__ = ["viewer_fixture"]
