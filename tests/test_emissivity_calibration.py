import numpy as np
import pytest

from topdon_duo.camera import CameraError
from topdon_duo.emissivity_calibration import EmissivityCalibrator


class Hardware:
    def __init__(self, enabled=False):
        self.value = 0.8 if enabled else 0.95
        self.baseline = 0.95
        self.enabled = enabled
        self.writes = []
        self.fail = False

    def state(self):
        return {"emissivity": {"value": self.value, "enabled": self.enabled, "available": True}}

    def set(self, name, value, enabled):
        if self.fail:
            self.fail = False
            raise CameraError("USB error")
        self.writes.append((name, value, enabled))
        self.enabled = enabled
        self.value = value if enabled else self.baseline


def prepared(enabled=False):
    hardware = Hardware(enabled)
    cal = EmissivityCalibrator(hardware)
    cal.begin()
    cal.select_point((3, 4), (10, 10))
    return cal, hardware


def run_fit(cal, hardware, target=40, response=lambda e: 20 + 10 / e):
    cal.start(target, 0)
    for step in range(1, 200):
        cal.update(np.full((10, 10), response(hardware.value)), step * 0.2)
        if not cal.running:
            return
    pytest.fail("Fit did not finish")


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "response,target,expected", [(lambda e: 20 + 10 / e, 40, 0.5), (lambda e: 20 + 10 * e, 25, 0.5)]
)
def test_fit_uses_camera_response_restores_previous_and_only_apply_keeps_result(
    enabled, response, target, expected
):
    cal, hardware = prepared(enabled)
    initial = hardware.value
    run_fit(cal, hardware, target, response)
    assert cal.result == pytest.approx(expected, abs=0.02)
    assert hardware.value == initial and hardware.enabled == enabled
    assert cal.active and cal.point is None
    assert cal.apply() == cal.result
    assert not cal.active and cal.point is None
    assert hardware.enabled and hardware.value == cal.result


@pytest.mark.parametrize("target,response", [(20, lambda e: 20 + 10 / e), (30, lambda _e: 30)])
def test_unreachable_or_uninformative_response_is_not_applied(target, response):
    cal, hardware = prepared()
    run_fit(cal, hardware, target, response)
    assert cal.result is None
    assert hardware.value == 0.95 and not hardware.enabled


def test_cancel_restores_active_override_and_retains_no_selection():
    cal, hardware = prepared(True)
    cal.start(40, 0)
    assert hardware.value == 1
    cal.cancel()
    assert hardware.value == 0.8 and hardware.enabled
    assert not cal.running and not cal.active and cal.point is None


def test_settling_unstable_readings_and_timeout_restore():
    cal, hardware = prepared()
    cal.start(40, 0)
    for step in range(20):
        cal.update(np.full((10, 10), 25 if step % 2 else 27), step * 0.2)
    assert len(hardware.writes) == 1
    cal.update(np.full((10, 10), np.nan), 31)
    assert not cal.running and cal.result is None
    assert hardware.value == 0.95


def test_failed_restore_is_retried_without_exposing_a_result():
    cal, hardware = prepared()
    cal.start(40, 0)
    hardware.fail = True
    with pytest.raises(CameraError):
        cal.cancel()
    assert cal.running and cal.result is None
    cal.update(np.full((10, 10), 30), 1)
    assert not cal.running and hardware.value == 0.95
    assert not cal.active and cal.point is None


def test_initial_probe_error_restores_previous_override():
    cal, hardware = prepared(True)
    hardware.fail = True
    with pytest.raises(CameraError):
        cal.start(40, 0)
    assert not cal.running and cal.result is None
    assert hardware.value == 0.8 and hardware.enabled


def test_saved_fit_survives_close_and_can_be_reapplied_after_reload():
    cal, hardware = prepared()
    run_fit(cal, hardware)
    reference = cal.reference.copy()
    cal.cancel()
    assert cal.reference == reference and cal.result is None
    reloaded = EmissivityCalibrator(hardware, reference)
    assert not reloaded.active
    assert reloaded.apply() == 0.5
    assert hardware.value == 0.5


def test_selection_copies_pixel_temperature_once_and_each_reselection_has_an_id():
    cal = EmissivityCalibrator(Hardware())
    cal.begin()
    cal.select_point((3, 4), (10, 10), 31.25)
    assert cal.state()["selected_celsius"] == 31.25
    assert cal.known_celsius == 31.25
    first_id = cal.selection_id
    cal.update(np.full((10, 10), 32), 0)
    assert cal.known_celsius == 31.25 and cal.measured_celsius == 32
    cal.begin()
    cal.select_point((3, 4), (10, 10), 29)
    assert cal.selection_id > first_id and cal.known_celsius == 29


def test_reference_marker_is_visible_and_clears_when_fit_is_saved():
    from topdon_duo.desktop import draw_emissivity_point

    cal, hardware = prepared()
    image = np.full((30, 30, 3), 100, np.uint8)
    marked = draw_emissivity_point(image, cal, 3, "C")
    assert not np.array_equal(marked, image)
    assert np.all(image == 100)
    run_fit(cal, hardware)
    assert cal.point is None
    assert np.array_equal(draw_emissivity_point(image, cal, 3, "C"), image)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -51, 551, "40"])
def test_invalid_known_temperature_never_writes_hardware(value):
    cal, hardware = prepared()
    with pytest.raises(ValueError):
        cal.start(value, 0)
    assert hardware.writes == []
