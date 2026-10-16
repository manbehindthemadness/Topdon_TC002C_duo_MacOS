"""
Combined user presets preserve environmental values and calibration references as one unit.
"""

import json
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import Mock, call

import pytest
from support.desktop_recording import viewer_fixture

from based_thermals import desktop
from based_thermals.camera_backends import ControlSpec
from based_thermals.camera_backends.duo import PROFILE, control_specs
from based_thermals.desktop_app.session import DesktopSession
from based_thermals.distance_calibration import DistanceReference
from based_thermals.environment_presets import (
    load_presets,
    measurement_specs,
    presets_path,
    save_presets,
    validate_name,
    validate_preset,
)
from based_thermals.settings_preferences import load_settings


@pytest.fixture
def session(viewer: Any) -> Iterator[Any]:
    """
    Provide real session coordination with stable fake hardware and saved references.
    """
    current = DesktopSession(desktop, [])
    current.view_panel = Mock()
    current.notify = Mock()
    current.renderer.request_measurement_restart = Mock()
    current.hardware.capabilities.return_value = {"features": {
        "calibrate_now": True, "host_distance_calibration": True,
        "host_emissivity_calibration": True, "host_reflected_calibration": True,
    }}
    for name, value in {"ambient": 22.2, "distance": 0.5, "emissivity": 0.95,
                        "reflected": 24.0, "transmission": 100, "humidity": 45.5}.items():
        viewer.hardware.state.return_value[name].update(value=value)
    current.distance_calibration.reference = DistanceReference(0.076, 0.5, 40.0)
    current.emissivity_calibration.reference = {
        "version": 1, "emissivity": 0.95, "known_celsius": 40.0,
    }
    current.reflected_calibration.reference = {"version": 1, "celsius": 24.0}
    assert current.hardware is viewer.hardware
    try:
        yield current
    finally:
        current.renderer.pipeline_processor.close()
        current.support_export.close()
        current.support_logging.close()


def dispatch(session: Any, operation: str, name: str = "Desk", **kwargs: Any) -> None:
    """
    Run preset operations through the public Imaging command boundary.
    """
    session.view_panel.poll.return_value = [
        {"action": "environment_preset", "operation": operation, "name": name, **kwargs},
    ]
    session.poll_camera_commands()


def test_user_collection_starts_empty_and_save_round_trips_everything(session: Any) -> None:
    """
    Save actual camera corrections and all references without applying or shipping anything.
    """
    assert load_presets(PROFILE, control_specs()) == {}
    assert not presets_path(PROFILE).exists()
    expected = session.environment_presets.snapshot()
    session.renderer.temperature_unit = "F"
    dispatch(session, "save")
    assert load_presets(PROFILE, control_specs()) == {"Desk": expected}
    assert set(expected["hardware"]) == {
        "ambient", "distance", "emissivity", "reflected", "transmission", "humidity",
    }
    assert expected["hardware"]["ambient"] == 22.2
    assert all(expected["calibrations"].values())
    session.hardware.set.assert_not_called()
    session.hardware.calibrate_now.assert_not_called()
    assert session.view_state()["environment_presets"] == {
        "names": ["Desk"], "current": "Desk", "ready": True, "modified": False, "locked": False,
    }
    assert load_settings()["environment_preset"] == "Desk"


def test_load_restores_both_sections_then_recalibrates_once(session: Any) -> None:
    """
    Restore all values as explicit corrections and clear unfinished calibration state together.
    """
    dispatch(session, "save")
    saved = deepcopy(session.environment_presets.presets["Desk"])
    session.hardware.state.return_value["ambient"].update(value=35.0)
    session.distance_calibration.reference = None
    session.distance_calibration.estimated_m = 8.0
    session.distance_calibration.corners = [[1, 2]]
    session.emissivity_calibration.reference = None
    session.reflected_calibration.reference = None
    previous_pipeline = deepcopy(session.pipeline)
    session.hardware.reset_mock()
    assert session.environment_presets.state()["modified"]
    dispatch(session, "load")
    assert session.hardware.set.call_args_list == [
        call(name, value, True) for name, value in saved["hardware"].items()
    ]
    calls = session.hardware.method_calls
    assert calls.index(call.calibrate_now()) > max(
        calls.index(call.set(name, value, True)) for name, value in saved["hardware"].items()
    )
    session.hardware.calibrate_now.assert_called_once_with()
    session.renderer.request_measurement_restart.assert_called_once_with()
    assert session.environment_presets.snapshot() == saved
    calibration_state = session.distance_calibration.state()
    assert calibration_state["estimated_m"] is None
    assert calibration_state["corners"] == 0
    assert session.pipeline == previous_pipeline
    assert load_settings()["hardware"] == saved["hardware"]
    assert session.ambient_input_celsius == 22.2
    assert not session.environment_presets.state()["modified"]


def test_matching_saved_values_recover_name_without_writes(session: Any) -> None:
    """
    Recognize saved values after losing selection metadata or independently restoring them.
    """
    dispatch(session, "save")
    session.environment_presets.current_name = None
    state = session.environment_presets.state()
    assert state["current"] == "Desk" and not state["modified"]
    session.hardware.set.assert_not_called()
    session.hardware.calibrate_now.assert_not_called()


def test_missing_selection_is_recognized_after_restart(session: Any) -> None:
    """
    An older preferences file without association metadata still recognizes the full snapshot.
    """
    dispatch(session, "save")
    saved = load_settings()
    saved.pop("environment_preset")
    desktop.save_settings(saved)
    restarted = DesktopSession(desktop, [])
    try:
        state = restarted.environment_presets.state()
        assert state["current"] == "Desk" and not state["modified"]
    finally:
        restarted.renderer.pipeline_processor.close()
        restarted.support_export.close()
        restarted.support_logging.close()


@pytest.mark.parametrize("difference", ["hardware", "reference", "unavailable", "locked"])
def test_recognition_requires_ready_matching_values_and_references(
    session: Any, difference: str,
) -> None:
    """
    Never label a different or temporary setup as a saved preset.
    """
    dispatch(session, "save")
    session.environment_presets.current_name = None
    if difference == "hardware":
        session.hardware.state.return_value["ambient"]["value"] = 25.0
    elif difference == "reference":
        session.reflected_calibration.reference = {"version": 1, "celsius": 25.0}
    elif difference == "unavailable":
        session.hardware.state.return_value["ambient"]["available"] = False
    else:
        session.emissivity_calibration.active = True
    assert session.environment_presets.state()["current"] is None


def test_recognition_preserves_explicit_selection_and_modified_state(session: Any) -> None:
    """
    Identical presets keep the selected name, and edits do not silently switch to another name.
    """
    dispatch(session, "save", name="Other")
    dispatch(session, "save")
    assert session.environment_presets.state()["current"] == "Desk"
    session.hardware.state.return_value["ambient"]["value"] = 25.0
    dispatch(session, "save", replace=True)
    assert session.environment_presets.state()["current"] == "Desk"
    session.hardware.state.return_value["ambient"]["value"] = 22.2
    state = session.environment_presets.state()
    assert state["current"] == "Desk" and state["modified"]


def test_presets_without_references_replace_previous_references_with_none(session: Any) -> None:
    """
    An environment-only snapshot is still one preset and cannot inherit stale references.
    """
    session.distance_calibration.reference = None
    session.emissivity_calibration.reference = None
    session.reflected_calibration.reference = None
    dispatch(session, "save")
    session.reflected_calibration.reference = {"version": 1, "celsius": 30.0}
    dispatch(session, "load")
    assert session.environment_presets.snapshot()["calibrations"] == {
        "distance": None, "emissivity": None, "reflected": None,
    }


def test_update_rename_delete_preserve_working_values_and_survive_restart(session: Any) -> None:
    """
    Manage user names and changed snapshots without affecting applied settings or references.
    """
    dispatch(session, "save")
    session.hardware.state.return_value["ambient"].update(value=25.0)
    dispatch(session, "save", replace=True)
    assert session.environment_presets.presets["Desk"]["hardware"]["ambient"] == 25.0
    session.hardware.state.return_value["ambient"].update(value=30.0)
    dispatch(session, "rename", new_name="Studio")
    assert set(session.environment_presets.presets) == {"Studio"}
    assert session.environment_presets.presets["Studio"]["hardware"]["ambient"] == 25.0
    assert session.environment_presets.current_name == "Studio"
    restarted = DesktopSession(desktop, [])
    try:
        assert restarted.environment_presets.current_name == "Studio"
        assert set(restarted.environment_presets.presets) == {"Studio"}
    finally:
        restarted.renderer.pipeline_processor.close()
        restarted.support_export.close()
        restarted.support_logging.close()
    dispatch(session, "delete", name="Studio")
    assert load_presets(PROFILE, control_specs()) == {}
    assert session.environment_presets.current_name is None
    assert session.hardware.state()["ambient"]["value"] == 30.0
    assert session.reflected_calibration.reference is not None
    session.hardware.set.assert_not_called()
    session.hardware.calibrate_now.assert_not_called()


@pytest.mark.parametrize("operation", ["save", "rename"])
def test_name_collisions_require_explicit_replacement(session: Any, operation: str) -> None:
    """
    Failed collision attempts retain both saved entries and the acknowledged selection.
    """
    dispatch(session, "save")
    dispatch(session, "save", name="Other")
    before = deepcopy(session.environment_presets.presets)
    dispatch(session, operation, new_name="Other")
    assert session.environment_presets.presets == before
    assert "Confirm replacement" in session.hardware.error
    dispatch(session, operation, new_name="Other", replace=True)
    if operation == "rename":
        assert "Desk" not in session.environment_presets.presets


@pytest.mark.parametrize("lock", ["logging", "pending", "emissivity", "reflected", "distance", "tone"])
def test_save_and_load_respect_all_locks(session: Any, lock: str) -> None:
    """
    Never capture temporary calibration settings or mutate a logging session.
    """
    dispatch(session, "save")
    if lock == "logging":
        session.graphs.logging = True
    elif lock == "pending":
        session.pending_save_kind = "graph_log"
    elif lock == "emissivity":
        session.emissivity_calibration.active = True
    elif lock == "reflected":
        session.reflected_calibration.active = True
    elif lock == "distance":
        session.distance_calibration.selecting = "reference"
    else:
        session.hardware.tone_busy = True
    before = deepcopy(session.environment_presets.presets)
    dispatch(session, "save", replace=True)
    dispatch(session, "load")
    assert session.environment_presets.presets == before
    assert session.environment_presets.state()["locked"]
    session.hardware.set.assert_not_called()
    session.hardware.calibrate_now.assert_not_called()


def test_failed_load_rolls_back_without_replacing_preferences_or_references(session: Any) -> None:
    """
    A later hardware failure restores earlier writes and retains the original working context.
    """
    dispatch(session, "save")
    session.hardware.state.return_value["ambient"].update(value=35.0, enabled=True)
    session.reflected_calibration.reference = {"version": 1, "celsius": 33.0}
    before = session.hardware.state()
    preferences = load_settings()
    setter = session.hardware.set.side_effect
    failed = False

    def write(name: str, value: float, enabled: bool) -> None:
        """
        Fail once after a preceding setting was successfully written.
        """
        nonlocal failed
        if name == "distance" and not failed:
            failed = True
            raise desktop.CameraError("Injected write failure")
        setter(name, value, enabled)

    session.hardware.set.side_effect = write
    dispatch(session, "load")
    assert session.hardware.state() == before
    assert session.reflected_calibration.reference["celsius"] == 33.0
    assert load_settings() == preferences
    assert "Injected write failure" in session.hardware.error
    session.hardware.calibrate_now.assert_not_called()


def test_recalibration_failure_retains_successful_preset_application(session: Any) -> None:
    """
    Shutter failure reports its own error after values, references and selection are saved.
    """
    dispatch(session, "save")
    session.hardware.state.return_value["ambient"].update(value=30.0)
    session.hardware.calibrate_now.side_effect = desktop.CameraError("Shutter failed")
    dispatch(session, "load")
    assert session.hardware.state()["ambient"]["value"] == 22.2
    assert load_settings()["hardware"]["ambient"] == 22.2
    assert session.environment_presets.current_name == "Desk"
    assert "recalibration failed: Shutter failed" in session.hardware.error


def test_failed_rollback_reports_both_failures_and_retains_references(session: Any) -> None:
    """
    A disconnected camera must report incomplete restoration without publishing new references.
    """
    dispatch(session, "save")
    session.reflected_calibration.reference = {"version": 1, "celsius": 33.0}
    session.hardware.set.side_effect = desktop.CameraError("Disconnected")
    dispatch(session, "load")
    assert "Preset rejected: Disconnected; restoration failed: ambient: Disconnected" in session.hardware.error
    assert session.reflected_calibration.reference["celsius"] == 33.0
    session.hardware.calibrate_now.assert_not_called()


def test_presets_and_selection_survive_display_reset_and_hardware_restore(session: Any) -> None:
    """
    Restore actions change the working setup without deleting named user presets.
    """
    dispatch(session, "save")
    before = deepcopy(session.environment_presets.presets)
    for action in ("reset", "restore_hardware"):
        session.view_panel.poll.return_value = [{"action": action}]
        session.poll_camera_commands()
        assert session.environment_presets.presets == before
        assert session.environment_presets.current_name == "Desk"
    assert load_presets(PROFILE, control_specs()) == before


def test_disk_failure_preserves_collection_and_selection(
    session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A failed rename must leave the saved and in-memory collections untouched.
    """
    dispatch(session, "save")
    monkeypatch.setattr("based_thermals.desktop_app.environment.save_presets",
                        Mock(side_effect=OSError("Disk full")))
    dispatch(session, "rename", new_name="Studio")
    assert session.environment_presets.current_name == "Desk"
    assert set(session.environment_presets.presets) == {"Desk"}
    assert set(load_presets(PROFILE, control_specs())) == {"Desk"}
    assert "Disk full" in session.hardware.error


@pytest.mark.parametrize("failure", ["unavailable", "missing", "incompatible", "invalid", "unsupported"])
def test_invalid_or_unavailable_load_is_rejected_before_writes(session: Any, failure: str) -> None:
    """
    All document fields and capabilities are checked before the first hardware write.
    """
    dispatch(session, "save")
    document = session.environment_presets.presets["Desk"]
    if failure == "unavailable":
        session.hardware.state.return_value["humidity"]["available"] = False
    elif failure == "missing":
        del document["hardware"]["humidity"]
    elif failure == "incompatible":
        document["camera"] = "other-profile"
    elif failure == "invalid":
        document["hardware"]["humidity"] = float("nan")
    else:
        session.hardware.capabilities.return_value["features"]["host_reflected_calibration"] = False
    dispatch(session, "load")
    session.hardware.set.assert_not_called()
    session.hardware.calibrate_now.assert_not_called()
    assert session.hardware.error


def test_store_ignores_malformed_entries_and_keeps_profile_collections_isolated(session: Any) -> None:
    """
    User storage has no fallback defaults and never mixes incompatible camera identities.
    """
    document = session.environment_presets.snapshot()
    other = replace(PROFILE, id="other", calibration_key="different")
    assert presets_path(other) != presets_path(PROFILE)
    with pytest.raises(ValueError):
        validate_preset(document, other, control_specs())
    malformed = deepcopy(document)
    malformed["calibrations"]["emissivity"] = {"version": 1, "emissivity": None}
    path = presets_path(PROFILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"Desk": document, "Broken": malformed, "Empty": None}))
    assert load_presets(PROFILE, control_specs()) == {"Desk": document}
    assert load_presets(other, control_specs()) == {}
    for content in ("null", "[]", "{broken"):
        path.write_text(content)
        assert load_presets(PROFILE, control_specs()) == {}


def test_atomic_store_validation_preserves_existing_file(session: Any) -> None:
    """
    Invalid saves cannot overwrite the last complete collection.
    """
    dispatch(session, "save")
    before = presets_path(PROFILE).read_bytes()
    malformed = session.environment_presets.snapshot()
    malformed["hardware"]["emissivity"] = True
    with pytest.raises(ValueError):
        save_presets({"Invalid": malformed}, PROFILE, control_specs())
    assert presets_path(PROFILE).read_bytes() == before


@pytest.mark.parametrize("name", [None, "", "  ", 42, "a\nb", "x" * 101])
def test_invalid_names_are_rejected(name: object) -> None:
    """
    Reject malformed names before persistence or camera work.
    """
    with pytest.raises((ValueError, TypeError)):
        validate_name(name)


def test_only_supported_device_measurement_controls_belong_to_presets() -> None:
    """
    Display effects, spot controls and unsupported fields stay outside this preset scope.
    """
    specs = {"correction": ControlSpec("Correction", effect="measurement"),
             "display": ControlSpec("Display"),
             "spot": ControlSpec("Spot", effect="measurement", scope="spot"),
             "unsupported": ControlSpec("Unsupported", effect="measurement", supported=False)}
    assert set(measurement_specs(specs)) == {"correction"}


__all__ = ["viewer_fixture"]
