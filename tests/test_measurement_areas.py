"""
Independent native-grid measurement references, geometry persistence and graph statistics.
"""

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from based_thermals.desktop import SampleSpots, draw_sample_spots, spot_label_layout
from based_thermals.desktop_app.overlays import place_temperature_label
from based_thermals.measurements.overlays import label_layout
from based_thermals.measurements.paths import initial_region
from based_thermals.measurements.schema import default_region, validate_region
from based_thermals.render import RenderedThermalFrame, TemperatureStats
from based_thermals.settings_preferences import load_settings, save_settings


def region(kind: str, points: list, **values: object) -> dict:
    """
    Build explicit test geometry without deriving expected masks from application code.
    """
    return {**default_region(kind), "points": points, **values}


def test_native_kernels_and_exclusions_have_independent_references() -> None:
    """
    Sample known sensor pixels, subtract an extreme, and retain unchanged input arrays.
    """
    temperatures = np.arange(25, dtype=float).reshape(5, 5)
    original = temperatures.copy()
    spots = SampleSpots(native_size=(5, 5))
    point = spots.create_region((2, 2), region("spot", [], radius=1), (5, 5))
    rectangle = spots.create_region((2, 2), region("rectangle", [[1, 1], [3, 3]]), (5, 5))
    oval = spots.create_region((2, 2), region("oval", [[1, 1], [3, 3]]), (5, 5))
    triangle = spots.create_region((0, 0), region("polygon", [[0, 0], [2, 0], [0, 2]]), (5, 5))
    exclusion = spots.create_region(
        (2, 1), region("rectangle", [[2, 1], [2, 1]], exclude=True), (5, 5)
    )
    readings = spots.readings(temperatures)
    # Radius 1 is the center and four neighbors; exclusion removes value 7.
    assert (
        readings[point].minimum,
        readings[point].maximum,
        readings[point].average,
        readings[point].count,
    ) == (11, 17, 13.25, 4)
    assert readings[oval] == readings[point]
    assert readings[rectangle].count == 8
    assert readings[rectangle].minimum == 6 and readings[rectangle].maximum == 18
    assert readings[rectangle].average == pytest.approx((6 + 8 + 11 + 12 + 13 + 16 + 17 + 18) / 8)
    assert readings[triangle].count == 6
    assert readings[triangle].average == pytest.approx((0 + 1 + 2 + 5 + 6 + 10) / 6)
    assert exclusion not in readings
    assert exclusion - 1 not in [key[1] for key, _value in spots.graph_readings(temperatures)]
    spots.set_enabled(exclusion, False)
    assert spots.readings(temperatures)[point].average == 12
    np.testing.assert_array_equal(temperatures, original)


def test_nonconvex_polygon_and_closed_cubic_rectangle() -> None:
    """
    Cover a concave L and four straight cubic segments with hand-counted pixel sets.
    """
    temperatures = np.arange(16, dtype=float).reshape(4, 4)
    spots = SampleSpots(native_size=(4, 4))
    polygon = spots.create_region(
        (1, 1), region("polygon", [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]), (4, 4)
    )
    bezier = spots.create_region(
        (1, 1),
        region(
            "bezier",
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [2, 3],
                [1, 3],
                [0, 3],
                [0, 2],
                [0, 1],
            ],
        ),
        (4, 4),
    )
    samples = spots.readings(temperatures)
    assert samples[polygon].count == 12
    assert samples[polygon].average == pytest.approx(70 / 12)
    assert samples[bezier].count == 16
    assert samples[bezier].graph_values() == (0, 7.5, 15)


def test_fully_excluded_nonfinite_and_clipped_spot_samples_are_unavailable() -> None:
    """
    Produce gaps rather than zeros for empty kernels; radius clips at the sensor edge.
    """
    temperatures = np.arange(25, dtype=float).reshape(5, 5)
    spots = SampleSpots(native_size=(5, 5))
    circle = spots.create_region((0, 0), region("spot", [], radius=1), (5, 5))
    point = spots.create_region((4, 4), default_region(), (5, 5))
    assert spots.readings(temperatures)[circle].average == 2
    assert spots.readings(temperatures)[circle].count == 3
    temperatures[4, 4] = np.inf
    assert spots.readings(temperatures)[point].count == 0
    zone = spots.create_region((0, 0), region("rectangle", [[0, 0], [4, 4]], exclude=True), (5, 5))
    assert spots.readings(temperatures)[circle].count == 0
    assert np.isnan(spots.readings(temperatures)[circle].average)
    assert all(np.isnan(v) for v in dict(spots.graph_readings(temperatures))[0, circle - 1])
    assert zone not in spots.readings(temperatures)


@pytest.mark.parametrize("kind", ["spot", "rectangle", "oval", "polygon", "bezier"])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(
    "horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)]
)
def test_area_geometry_and_kernels_round_trip_orientation(
    kind: str,
    rotation: int,
    horizontal: bool,
    vertical: bool,
) -> None:
    """
    Preserve shapes, radii, handles, exclusions and samples through rotation/mirroring.
    """
    spots = SampleSpots(native_size=(32, 24))
    center, geometry = initial_region(kind, (32, 24))
    geometry.update(radius=2.5, labels=["name", "min"], exclude=False)
    number = spots.create_region(center, geometry, (32, 24))
    spots.rename(number, "Bearing")
    saved = spots.saved_state()
    restored = SampleSpots(native_size=(32, 24))
    restored.restore_saved_state(saved, rotation, horizontal, vertical)
    assert restored.saved_state(rotation, horizontal, vertical) == saved
    plane = np.arange(32 * 24, dtype=float).reshape(24, 32)
    oriented = np.rot90(plane, -rotation // 90)
    if horizontal:
        oriented = oriented[:, ::-1]
    if vertical:
        oriented = oriented[::-1]
    original = spots.readings(plane)[number]
    current = restored.readings(oriented)[number]
    assert current.count == original.count
    assert current.graph_values() == pytest.approx(original.graph_values())


def test_legacy_migration_persistence_and_atomic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preserve legacy point files and independently reject malformed area configuration.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    spots = SampleSpots(pixels=[(10, 10)])
    legacy = spots.saved_state()
    assert legacy["version"] == 1
    spots.restore_saved_state(legacy)
    center, rectangle = initial_region("rectangle", (256, 192))
    number = spots.create_region(center, rectangle, (256, 192))
    save_settings({"spots": spots.saved_state()})
    assert load_settings()["spots"] == spots.saved_state()
    before = deepcopy(spots.saved_state())
    for bad in (
        None,
        {},
        {**rectangle, "points": [[0, 0], [256, 3]]},
        {**rectangle, "labels": ["min", "min"]},
    ):
        with pytest.raises((ValueError, TypeError)):
            spots.configure_region(number, bad, (256, 192))
        assert spots.saved_state() == before
    with pytest.raises(ValueError):
        spots.configure_region(number, rectangle, (256, 192), pixel=[999, 0])
    assert spots.saved_state() == before
    invalid = deepcopy(before)
    invalid["items"][1]["region"]["points"] = None
    save_settings({"spots": invalid, "rotation": 90})
    assert "spots" not in load_settings()
    assert load_settings()["rotation"] == 90


def test_edge_labels_visibility_and_shape_collision_reservations() -> None:
    """
    Attach leaders outside areas, reserve shape bounds and show configured Celsius/Fahrenheit.
    """
    plane = np.arange(48, dtype=float).reshape(6, 8)
    rendered = RenderedThermalFrame(
        np.zeros((120, 160, 3), np.uint8), TemperatureStats(0, 23.5, 47, 28), plane, None
    )
    spots = SampleSpots(native_size=(8, 6))
    number = spots.create_region(
        (3, 2), region("rectangle", [[2, 1], [4, 3]], labels=["number", "average"]), (8, 6)
    )
    labels, occupied = label_layout(rendered, spots, 20, "F", place_temperature_label)
    assert "66.20 F" in labels[0].text
    assert labels[0].anchor != (70, 50)
    assert labels[0].anchor[0] in (50, 90) or labels[0].anchor[1] in (30, 70)
    assert occupied[0] == (47, 27, 94, 74)
    image = draw_sample_spots(rendered.image, rendered, spots, 20)
    assert np.any(image)
    hidden = {**spots.region(number), "labels": []}
    spots.configure_region(number, hidden, (8, 6))
    assert spot_label_layout(rendered, spots, 20, "C")[0] == []
    assert np.any(draw_sample_spots(rendered.image, rendered, spots, 20))
    np.testing.assert_array_equal(rendered.temperatures_celsius, plane)


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (["number", "name"], "Exclusion 1: Keepout"),
        (["number"], "Exclusion 1"),
        (["name"], "Exclusion: Keepout"),
        (["name", "average"], "Exclusion: Keepout"),
        (["min", "max", "average"], None),
        ([], None),
    ],
)
def test_exclusion_labels_honor_number_and_name(fields: list[str], expected: str | None) -> None:
    """
    Hidden temperature selections never force an exclusion identifier to remain visible.
    """
    spots = SampleSpots(native_size=(8, 6))
    number = spots.create_region(
        (3, 2), region("rectangle", [[2, 1], [4, 3]], exclude=True, labels=["number", "name"]), (8, 6)
    )
    spots.rename(number, "Keepout")
    rendered = RenderedThermalFrame(
        np.zeros((120, 160, 3), np.uint8), TemperatureStats(0, 0, 0, 0), np.zeros((6, 8)), None,
    )
    # Warm the layout cache before changing the visible label fields.
    label_layout(rendered, spots, 20, "C", place_temperature_label)
    spots.configure_region(number, {**spots.region(number), "labels": fields}, (8, 6))
    labels, occupied = label_layout(rendered, spots, 20, "C", place_temperature_label)
    assert [label.text for label in labels] == ([expected] if expected is not None else [])
    assert len(occupied) == 1 + len(labels)


def test_area_move_clamps_whole_geometry_and_keeps_identity() -> None:
    """
    Retain vertices/handles inside the grid when an area is dragged to an edge.
    """
    spots = SampleSpots(native_size=(10, 8))
    number = spots.create_region((5, 4), region("rectangle", [[3, 2], [7, 6]]), (10, 8))
    spots.rename(number, "Motor")
    spots.move(number, (9, 7), size=(10, 8))
    assert spots.pixels == [(7, 5)]
    assert spots.region(number)["points"] == [[5, 3], [9, 7]]
    assert spots.name(number) == "Motor"
    assert validate_region(spots.region(number), (10, 8)) == spots.region(number)


def test_dragging_area_body_consumes_release_without_placing_a_point() -> None:
    """
    Grab away from the center and translate every vertex using the existing drag path.
    """
    from based_thermals.desktop import MousePicker, SpotDrag

    spots = SampleSpots(placing=True, native_size=(40, 30))
    number = spots.create_region((20, 15), region("rectangle", [[5, 5], [25, 20]]), (40, 30))
    picker, drag = MousePicker(), SpotDrag()
    picker.callback(cv2.EVENT_LBUTTONDOWN, 17, 17, 0, None)
    drag.update(picker, spots, (60, 80, 3), 2)
    assert drag.number == number
    picker.callback(cv2.EVENT_MOUSEMOVE, 27, 21, cv2.EVENT_FLAG_LBUTTON, None)
    drag.update(picker, spots, (60, 80, 3), 2)
    picker.callback(cv2.EVENT_LBUTTONUP, 27, 21, 0, None)
    drag.update(picker, spots, (60, 80, 3), 2)
    assert spots.pixels == [(25, 17)]
    assert spots.region(number)["points"] == [[10, 7], [30, 22]]
    assert not picker.consume_clicks()


def test_area_editor_changes_leave_no_invisible_center_hit_target() -> None:
    """
    Select an area's visible body after vertex edits, keeping empty space clickable.
    """
    from based_thermals.desktop import spot_hit

    spots = SampleSpots(native_size=(40, 30))
    number = spots.create_region((20, 15), region("rectangle", [[1, 1], [3, 3]]), (40, 30))
    assert spot_hit((41, 31), spots, (60, 80, 3), 2) is None
    hit = spot_hit((5, 5), spots, (60, 80, 3), 2)
    assert hit is not None and hit[1] == number
