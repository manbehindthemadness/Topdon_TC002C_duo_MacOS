"""
Committed environmental corrections recalibrate without disrupting fitting probes.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, call

import pytest
from support.desktop_recording import viewer_fixture

from based_thermals import desktop
from based_thermals.desktop_app.session import DesktopSession
from based_thermals.settings_preferences import load_settings


@pytest.fixture
def session(viewer: Any) -> Iterator[Any]:
    """
    Create a real command dispatcher with fake camera boundaries and renderer spies.
    """
    current = DesktopSession(desktop, [])
    current.view_panel = Mock()
    current.hardware.capabilities.return_value = {"features": {"calibrate_now": True}}
    current.renderer.request_measurement_restart = Mock()
    current.renderer.reset_measurement_average = Mock()
    current.notify = Mock()
    assert current.hardware is viewer.hardware
    try:
        yield current
    finally:
        current.renderer.pipeline_processor.close()
        current.support_export.close()
        current.support_logging.close()


def dispatch(session: Any, command: dict[str, Any]) -> None:
    """
    Send one Imaging command through normal validation and persistence.
    """
    session.view_panel.poll.return_value = [command]
    session.poll_camera_commands()


@pytest.mark.parametrize("name,value", [
    ("ambient", 22.2), ("distance", 0.5), ("emissivity", 0.95),
    ("reflected", 24.0), ("transmission", 100), ("humidity", 40.0),
])
@pytest.mark.parametrize("enabled", [True, False])
def test_environment_apply_and_disable_recalibrate_after_write(
    session: Any, name: str, value: float, enabled: bool,
) -> None:
    """
    Every environmental field writes before recalibration, including override removal.
    """
    dispatch(session, {"action": "hardware", "name": name, "value": value, "enabled": enabled})
    writes = session.hardware.method_calls
    assert writes.index(call.set(name, value, enabled)) < writes.index(call.calibrate_now())
    session.hardware.calibrate_now.assert_called_once_with()
    session.renderer.request_measurement_restart.assert_called_once_with()
    session.renderer.reset_measurement_average.assert_not_called()
    saved = load_settings()["hardware"]
    assert (name in saved) == enabled


@pytest.mark.parametrize("kind,name,value", [
    ("distance", "distance", 0.5),
    ("emissivity", "emissivity", 0.95),
    ("reflected", "reflected", 24.0),
])
def test_calibration_apply_recalibrates(
    session: Any, kind: str, name: str, value: float,
) -> None:
    """
    Applying a measured or saved calibration uses the same recalibration path.
    """
    session.distance_calibration.estimated_m = 0.5
    session.emissivity_calibration.reference = {
        "version": 1, "emissivity": 0.95, "known_celsius": 40.0,
    }
    session.reflected_calibration.reference = {"version": 1, "celsius": 24.0}
    dispatch(session, {"action": f"{kind}_calibration", "operation": "apply"})
    session.hardware.set.assert_called_with(name, value, True)
    session.hardware.calibrate_now.assert_called_once_with()
    session.renderer.request_measurement_restart.assert_called_once_with()
    session.renderer.reset_measurement_average.assert_not_called()
    assert load_settings()["hardware"][name] == value


def test_fit_probes_and_cancellation_do_not_trigger_recalibration(session: Any) -> None:
    """
    Temporary fitting writes and restoration retain their existing settling behavior.
    """
    session.emissivity_calibration.begin()
    session.emissivity_calibration.select_point((3, 4), (256, 192), 25.0)
    dispatch(session, {"action": "emissivity_calibration", "operation": "fit",
                       "known_celsius": 30.0})
    assert session.emissivity_calibration.running
    dispatch(session, {"action": "emissivity_calibration", "operation": "cancel"})
    assert not session.emissivity_calibration.running
    session.hardware.calibrate_now.assert_not_called()


def test_display_change_and_rejected_write_do_not_recalibrate(session: Any) -> None:
    """
    Display controls and unsuccessful correction writes never request recalibration.
    """
    dispatch(session, {"action": "hardware", "name": "brightness", "value": 50, "enabled": True})
    session.renderer.reset_measurement_average.assert_called_once_with()
    session.hardware.set.side_effect = desktop.CameraError("Write rejected")
    dispatch(session, {"action": "hardware", "name": "ambient", "value": 22.0, "enabled": True})
    session.hardware.calibrate_now.assert_not_called()
    session.renderer.request_measurement_restart.assert_not_called()
    assert "ambient" not in load_settings()["hardware"]


def test_unsupported_recalibration_keeps_environment_change(session: Any) -> None:
    """
    A backend without a recalibration command still accepts its supported correction.
    """
    session.hardware.capabilities.return_value = {"features": {}}
    dispatch(session, {"action": "hardware", "name": "ambient", "value": 22.0, "enabled": True})
    session.hardware.calibrate_now.assert_not_called()
    session.renderer.request_measurement_restart.assert_called_once_with()
    assert load_settings()["hardware"]["ambient"] == 22.0


def test_recalibration_failure_preserves_applied_value_and_reports_error(session: Any) -> None:
    """
    A failed shutter request must not misreport a successful settings write as rejected.
    """
    session.hardware.calibrate_now.side_effect = desktop.CameraError("Shutter unavailable")
    dispatch(session, {"action": "hardware", "name": "ambient", "value": 22.0, "enabled": True})
    assert session.hardware.state()["ambient"]["value"] == 22.0
    assert load_settings()["hardware"]["ambient"] == 22.0
    assert session.hardware.error == "Setting applied, but camera recalibration failed: Shutter unavailable"
    session.notify.assert_called_with(session.hardware.error)


def test_restore_settings_recalibrates_once(session: Any) -> None:
    """
    Restoring the original environmental settings requests one final recalibration.
    """
    dispatch(session, {"action": "restore_hardware"})
    session.hardware.restore.assert_called_once_with()
    session.hardware.calibrate_now.assert_called_once_with()


__all__ = ["viewer_fixture"]
