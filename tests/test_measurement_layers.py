"""Persistent display layers, overlap selection and isolated Qt layer controls."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from support.qt_process import run_popup

from based_thermals.desktop import SampleSpots, draw_sample_spots, spot_hit
from based_thermals.measurements.schema import default_region, validate_region
from based_thermals.render import RenderedThermalFrame, TemperatureStats


def square(**settings: object) -> dict:
    """
    Return a label-free filled square for independent overlap pixel references.
    """
    region = {
        **default_region("rectangle"),
        "points": [[1, 1], [8, 8]],
        "labels": [],
        "fill": True,
        **settings,
    }
    return region


def display_frame() -> RenderedThermalFrame:
    """
    Build a neutral display and independent native readings for layering references.
    """
    stats = TemperatureStats(0, 49.5, 99, 44)
    rendered = RenderedThermalFrame(
        np.full((10, 10, 3), 100, np.uint8),
        stats,
        np.arange(100, dtype=float).reshape(10, 10),
        None,
    )
    return rendered


def hit_number(spots: SampleSpots, position: tuple[int, int]) -> int:
    """
    Require a hit before inspecting its stable identity.
    """
    hit = spot_hit(position, spots, (10, 10, 3), 1)
    assert hit is not None
    return hit[1]


@pytest.mark.parametrize("value", [True, None, 1.5, "1", -1001, 1001])
def test_invalid_z_index_is_rejected(value: object) -> None:
    """
    Reject non-integer or out-of-range layers at the document boundary.
    """
    with pytest.raises(ValueError, match="Z index"):
        validate_region(square(z_index=value), (10, 10))


def test_saved_layers_preserve_identity_readings_and_mask_cache() -> None:
    """
    Layer changes persist through restore and transforms without changing radiometry.
    """
    spots = SampleSpots(native_size=(10, 10))
    first = spots.create_region((4, 4), square(z_index=7), (10, 10))
    second = spots.create_region((5, 5), square(z_index=-2), (10, 10))
    temperatures = np.arange(100, dtype=float).reshape(10, 10)
    original = spots.graph_readings(temperatures)
    mask = spots.measurement_mask(first, (4, 4), (10, 10))
    assert [number for number, _ in spots.display_active] == [second, first]
    spots.configure_region(first, square(z_index=-3), (10, 10))
    assert spots.measurement_mask(first, (4, 4), (10, 10)) is mask
    assert spots.graph_readings(temperatures) == original
    assert [number for number, _ in spots.active] == [first, second]
    restored = SampleSpots(native_size=(10, 10))
    restored.restore_saved_state(deepcopy(spots.saved_state()))
    restored.transform_regions(lambda x, y: (9 - x, y))
    assert restored.region(first)["z_index"] == -3
    assert restored.region(second)["z_index"] == -2
    old = square()
    old.pop("z_index")
    assert validate_region(old, (10, 10))["z_index"] == 0


def test_higher_layer_wins_color_overlap_and_selection() -> None:
    """
    Apply higher fills last and select their layer even when its anchor is farther away.
    """
    rendered = display_frame()
    image = rendered.image
    spots = SampleSpots(native_size=(10, 10))
    first = spots.create_region((4, 4), square(line_color="#ff0000", z_index=2), (10, 10))
    second = spots.create_region((7, 7), square(line_color="#0000ff", z_index=1), (10, 10))
    np.testing.assert_array_equal(draw_sample_spots(image, rendered, spots, 1)[4, 4], [0, 0, 255])
    assert hit_number(spots, (7, 7)) == first
    spots.configure_region(second, square(line_color="#0000ff", z_index=3), (10, 10))
    np.testing.assert_array_equal(draw_sample_spots(image, rendered, spots, 1)[4, 4], [255, 0, 0])
    assert hit_number(spots, (4, 4)) == second
    spots.set_enabled(second, False)
    assert hit_number(spots, (4, 4)) == first
    np.testing.assert_array_equal(image, np.full_like(image, 100))


@pytest.mark.parametrize("exclude", [False, True])
def test_lower_dynamic_border_does_not_draw_above_higher_fill(exclude: bool) -> None:
    """
    Flush lower contrast outlines before compositing higher colored or exclusion fills.
    """
    rendered = display_frame()
    image = rendered.image
    spots = SampleSpots(native_size=(10, 10))
    spots.create_region((4, 4), square(points=[[2, 2], [6, 6]], fill=False, z_index=-1), (10, 10))
    spots.create_region((4, 4), square(line_color="#0000ff", z_index=1, exclude=exclude), (10, 10))
    np.testing.assert_array_equal(draw_sample_spots(image, rendered, spots, 1)[4, 6], [255, 0, 0])


def test_equal_layers_keep_creation_order_and_nearest_hit() -> None:
    """
    Equal Z values preserve established drawing and selection tie behavior.
    """
    spots = SampleSpots(native_size=(10, 10))
    first = spots.create_region((4, 4), square(), (10, 10))
    second = spots.create_region((7, 7), square(), (10, 10))
    assert [number for number, _ in spots.display_active] == [first, second]
    assert hit_number(spots, (4, 4)) == first
    assert hit_number(spots, (7, 7)) == second


def test_layer_controls_commit_reorder_and_obey_locks(tmp_path: Path) -> None:
    """
    Keep pending layer edits, honor front/back actions and persist through the JSON boundary.
    """
    script = r"""
from copy import deepcopy
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.measurements_window import MeasurementsWindow
from based_thermals.measurements.paths import initial_region
app = QApplication([])
messages = []
window = MeasurementsWindow(messages.append)
center, region = initial_region("rectangle", (32, 24))
region["z_index"] = 0
other = {**region, "z_index": 8, "exclude": True}
state = {"sensor_size": [32, 24], "spots": [
    {"number": 1, "name": "Area", "enabled": True, "pixel": list(center), "region": region},
    {"number": 2, "name": "Exclude", "enabled": False, "pixel": list(center), "region": other}]}
window.update_state(deepcopy(state))
window.bring_front.click()
assert window.z_index.value() == 9 and window._dirty and messages == []
window.update_state(deepcopy(state))
assert window.z_index.value() == 9
window.send_back.click()
assert window.z_index.value() == 7
editor = window.z_index.lineEdit()
editor.selectAll()
QTest.keyClicks(editor, "-4")
window.update_state(deepcopy(state))
assert editor.text() == "-4"
window.apply.click()
assert messages[-2]["region"]["z_index"] == -4
state["spots"][0]["region"] = messages[-2]["region"]
window.update_state(deepcopy(state))
assert window.z_index.value() == -4
window.items.setCurrentRow(1)
assert window.z_index.value() == 8
window.send_back.click()
assert window.z_index.value() == -5
window.apply.click()
assert messages[-2]["region"]["exclude"] and messages[-2]["region"]["z_index"] == -5
state["spots"][1]["region"] = messages[-2]["region"]
window.update_state({**deepcopy(state), "locked": True})
assert all(not control.isEnabled() for control in (window.z_index, window.bring_front, window.send_back))
count = len(messages)
window.bring_front.click()
window.send_back.click()
window.apply.click()
assert len(messages) == count
window.update_state(deepcopy(state))
window.items.setCurrentRow(0)
state["spots"][1]["region"]["z_index"] = 1000
window.update_state(deepcopy(state))
window.bring_front.click()
assert window.z_index.value() == -4
assert "limit" in window.status.text()
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
