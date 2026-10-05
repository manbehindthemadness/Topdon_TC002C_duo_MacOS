import numpy as np
import pytest

from topdon_duo.camera import CameraError
from topdon_duo.reflected_calibration import ReflectedCalibrator, validate_reference
from topdon_duo.settings_preferences import load_settings, save_settings


class Hardware:
    def __init__(self):
        self.settings = {
            "emissivity": {"value": 0.8, "enabled": True, "available": True},
            "transmission": {"value": 95, "enabled": False, "available": True},
            "reflected": {"value": 22, "enabled": False, "available": True},
        }
        self.fail_name = None

    def state(self):
        return self.settings

    def set(self, name, value, enabled):
        if name == self.fail_name:
            self.fail_name = None
            raise CameraError("USB failed")
        self.settings[name].update(value=value, enabled=enabled)


def prepared():
    hardware = Hardware()
    calibration = ReflectedCalibrator(hardware)
    calibration.begin()
    calibration.start(0)
    return calibration, hardware


def test_stable_native_circle_saves_restores_and_apply_is_explicit():
    cal, hw = prepared()
    assert hw.state()["emissivity"]["value"] == 1
    assert hw.state()["transmission"]["value"] == 100
    temperatures = np.full((192, 256), 60.0)
    y, x = np.ogrid[:192, :256]
    temperatures[(x - 128) ** 2 + (y - 96) ** 2 <= 16] = 23.25
    for step in range(1, 20):
        cal.update(temperatures, step * 0.2)
    assert cal.reference == {"version": 1, "celsius": 23.2}
    assert not cal.active and not cal.running
    assert hw.state()["emissivity"]["value"] == 0.8
    assert hw.state()["emissivity"]["enabled"]
    assert hw.state()["transmission"]["value"] == 95
    assert not hw.state()["transmission"]["enabled"]
    assert hw.state()["reflected"]["value"] == 22
    assert cal.apply() == 23.2
    assert hw.state()["reflected"]["enabled"]


@pytest.mark.parametrize("bad", [float("nan"), 150])
def test_invalid_reading_never_saves_and_timeout_restores(bad):
    cal, hw = prepared()
    for step in range(1, 160):
        cal.update(np.full((192, 256), bad), step * 0.2)
    assert cal.reference is None
    assert not cal.running
    assert hw.state()["emissivity"]["value"] == 0.8


def test_changing_target_times_out_without_overwriting_saved_reference():
    cal, hw = prepared()
    cal.reference = {"version": 1, "celsius": 22}
    for step in range(1, 160):
        cal.update(np.full((192, 256), 20 + step % 2), step * 0.2)
    assert cal.reference["celsius"] == 22
    assert not cal.running
    assert hw.state()["transmission"]["value"] == 95


def test_cancel_restore_failure_retries_remaining_fields():
    cal, hw = prepared()
    hw.fail_name = "transmission"
    with pytest.raises(CameraError):
        cal.cancel()
    assert hw.state()["emissivity"]["value"] == 0.8
    assert cal.running
    cal.update(np.full((192, 256), 20), 1)
    assert not cal.running and not cal.active
    assert hw.state()["transmission"]["value"] == 95


def test_partial_start_failure_restores_camera():
    hw = Hardware()
    cal = ReflectedCalibrator(hw)
    cal.begin()
    hw.fail_name = "transmission"
    with pytest.raises(CameraError):
        cal.start(0)
    assert hw.state()["emissivity"]["value"] == 0.8
    assert not cal.running


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"version": True, "celsius": 20},
        {"version": 1, "celsius": True},
        {"version": 1, "celsius": float("inf")},
        {"version": 1, "celsius": 101},
    ],
)
def test_invalid_reference(value):
    with pytest.raises(ValueError):
        validate_reference(value)


def test_saved_reference_survives_reload_and_cancel(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    reference = {"version": 1, "celsius": 24.2}
    save_settings({"reflected_calibration": reference})
    cal = ReflectedCalibrator(Hardware(), load_settings()["reflected_calibration"])
    cal.begin()
    cal.cancel()
    assert cal.reference == reference
    assert cal.apply() == 24.2
