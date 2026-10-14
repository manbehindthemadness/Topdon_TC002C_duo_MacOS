"""
Measurement styling persistence, independent display references and isolated Qt controls.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from support.qt_process import run_popup

from based_thermals.desktop import SampleSpots, draw_sample_spots
from based_thermals.measurements.schema import default_region, validate_region
from based_thermals.render import RenderedThermalFrame, TemperatureStats


def rectangle(**values: object) -> dict:
    """
    Define a label-free square with hand-checkable display and native sample bounds.
    """
    return {**default_region("rectangle"), "points": [[2, 2], [6, 6]], "labels": [], **values}


@pytest.mark.parametrize(
    "values",
    [
        {"line_color": None},
        {"line_color": "red"},
        {"line_color": "#12345g"},
        {"fill": 1},
        {"opacity": True},
        {"opacity": None},
        {"opacity": -0.1},
        {"opacity": 1.1},
        {"opacity": float("nan")},
        {"opacity": float("inf")},
    ],
)
def test_invalid_appearance_is_rejected(values: dict) -> None:
    """
    Reject malformed styling before it reaches persistence or either renderer.
    """
    with pytest.raises((ValueError, TypeError)):
        validate_region(rectangle(**values), (10, 10))


def test_legacy_styles_and_saved_color_survive_restore() -> None:
    """
    Supply defaults to old regions and preserve appearance through native coordinate transforms.
    """
    old = rectangle()
    for key in ("line_color", "fill", "opacity", "z_index"):
        old.pop(key)
    assert validate_region(old, (10, 10)) == rectangle()
    spots = SampleSpots(native_size=(10, 10))
    number = spots.create_region(
        (4, 4), rectangle(line_color="#Ab1234", fill=True, opacity=0.4), (10, 10)
    )
    assert spots.region(number)["line_color"] == "#ab1234"
    restored = SampleSpots(native_size=(10, 10))
    restored.restore_saved_state(deepcopy(spots.saved_state()))
    assert restored.region(number) == spots.region(number)
    restored.transform_regions(lambda x, y: (9 - x, y))
    assert {key: restored.region(number)[key] for key in ("line_color", "fill", "opacity")} == {
        "line_color": "#ab1234",
        "fill": True,
        "opacity": 0.4,
    }


def test_fill_opacity_and_rgb_preserve_measurement_samples() -> None:
    """
    Compare interior and boundary pixels against independent alpha references, including zero opacity.
    """
    image = np.full((10, 10, 3), 100, np.uint8)
    temperatures = np.arange(100, dtype=float).reshape(10, 10)
    rendered = RenderedThermalFrame(image, TemperatureStats(0, 49.5, 99, 44), temperatures, None)
    spots = SampleSpots(native_size=(10, 10))
    number = spots.create_region(
        (4, 4), rectangle(line_color="#ff0000", fill=True, opacity=0.5), (10, 10)
    )
    readings = spots.readings(temperatures)
    saved_image, saved_temperatures = image.copy(), temperatures.copy()
    styled = draw_sample_spots(image, rendered, spots, 1)
    np.testing.assert_array_equal(styled[4, 4], [50, 50, 178])
    np.testing.assert_array_equal(styled[2, 4], [50, 50, 178])
    np.testing.assert_array_equal(styled[0, 0], [100, 100, 100])
    spots.configure_region(number, {**spots.region(number), "fill": False}, (10, 10))
    outlined = draw_sample_spots(image, rendered, spots, 1)
    np.testing.assert_array_equal(outlined[4, 4], [100, 100, 100])
    spots.configure_region(number, {**spots.region(number), "opacity": 0}, (10, 10))
    np.testing.assert_array_equal(draw_sample_spots(image, rendered, spots, 1), image)
    assert spots.readings(temperatures) == readings
    np.testing.assert_array_equal(image, saved_image)
    np.testing.assert_array_equal(temperatures, saved_temperatures)


def test_dynamic_fill_and_filled_exclusion_dashes() -> None:
    """
    Keep Dynamic alpha predictable and exclusion dash gaps visible above filled interiors.
    """
    image = np.full((100, 100, 3), 100, np.uint8)
    temperatures = np.arange(100, dtype=float).reshape(10, 10)
    rendered = RenderedThermalFrame(image, TemperatureStats(0, 49.5, 99, 44), temperatures, None)
    spots = SampleSpots(native_size=(10, 10))
    number = spots.create_region((4, 4), rectangle(fill=True, opacity=0.5), (10, 10))
    dynamic = draw_sample_spots(image, rendered, spots, 10)
    np.testing.assert_array_equal(dynamic[40, 40], [128, 128, 128])
    spots.configure_region(
        number, rectangle(fill=True, line_color="#ff0000", exclude=True), (10, 10)
    )
    exclusion = draw_sample_spots(image, rendered, spots, 10)
    np.testing.assert_array_equal(exclusion[40, 40], [0, 0, 255])
    assert exclusion[25, 28, 2] > 200
    assert np.all(exclusion[25, 28, :2] < 20)
    np.testing.assert_array_equal(exclusion[25, 34], [100, 100, 100])


def test_appearance_controls_commit_preserve_edits_and_obey_locks(tmp_path: Path) -> None:
    """
    Exercise the real color dialog boundary and parent JSON payload without opening native UI.
    """
    script = """
from copy import deepcopy
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QColorDialog
from based_thermals.measurements_window import MeasurementsWindow
from based_thermals.measurements.paths import initial_region
app = QApplication([])
messages = []
window = MeasurementsWindow(messages.append)
center, region = initial_region("rectangle", (32, 24))
state = {"sensor_size": [32,24], "orientation": [0,False,False], "locked": False,
         "spots": [{"number": 1, "name": "Area", "enabled": True,
                    "pixel": list(center), "region": region}]}
window.update_state(deepcopy(state))
controls = window.appearance
assert controls.dynamic.isChecked() and controls.opacity.value() == 100
QColorDialog.getColor = lambda *_: QColor("#12ab34")
controls.dynamic.setChecked(False)
controls.line_color.click()
controls.fill.setChecked(True)
controls.opacity.setValue(37)
assert messages == []
window.update_state(deepcopy(state))
assert controls.color == "#12ab34" and controls.opacity.value() == 37
assert window.preview.region["fill"] and window.preview.region["opacity"] == .37
window.apply.click()
configured = messages[-2]["region"]
assert configured["line_color"] == "#12ab34" and configured["fill"]
assert configured["opacity"] == .37
state["spots"][0]["region"] = configured
window.update_state(deepcopy(state))
assert controls.color == "#12ab34" and controls.opacity.value() == 37
editor = controls.opacity.lineEdit()
editor.selectAll()
QTest.keyClicks(editor, "42")
assert window._dirty and controls.opacity.value() == 37
window.update_state(deepcopy(state))
assert editor.text().startswith("42")
window.apply.click()
assert messages[-2]["region"]["opacity"] == .42
state["spots"][0]["region"] = messages[-2]["region"]
state["locked"] = True
window.update_state(deepcopy(state))
before = len(messages)
assert not controls.isEnabled() and not controls.opacity.isEnabled()
controls.line_color.click()
window.apply.click()
assert len(messages) == before
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr
