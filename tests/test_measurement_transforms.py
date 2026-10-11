"""
Display geometry never controls the native thermal canvas used by measurement regions.
"""

from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.frames import make_frame
from support.pipeline import raw_pipeline

from topdon_duo import desktop
from topdon_duo.camera import HEADER_U16, SENSOR_PIXELS
from topdon_duo.measurements.schema import default_region
from topdon_duo.pipeline import node
from topdon_duo.render import ThermalRenderer


def gradient_frame() -> bytes:
    """
    Encode an independent Celsius reference: 10 + x/4 + y/8 on the native plane.
    """
    words = np.frombuffer(make_frame(), dtype="<u2").copy()
    y, x = np.indices((192, 256))
    words[HEADER_U16:HEADER_U16 + SENSOR_PIXELS] = (60 * 64 + x * 16 + y * 8).ravel()
    return words.tobytes()


@pytest.mark.parametrize("settings", [
    {"width": 50, "height": 50},
    {"resize_mode": "resolution", "width": 50, "height": 50},
    {"position_x": 50, "position_y": -20},
    {"skew_x": 30},
    {"top_right_x": -20, "bottom_right_y": -20},
    {"horizontal": True, "vertical": True},
])
@pytest.mark.parametrize("connected", [False, True])
def test_regions_and_exclusions_sample_thermal_plane_through_transforms(
    settings: dict[str, Any], connected: bool,
) -> None:
    """
    Verify exact native samples and graph values despite transformed, white display pixels.
    """
    renderer = ThermalRenderer(scale=2)
    renderer.native_temperatures = True
    transform = node("software", "transform", **settings)
    document = raw_pipeline(node("software", "brightness", amount=100), transform)
    if connected:
        document["branches"]["B"] = document["software"][:-1]
        document["software"] = raw_pipeline(node("software", "combine", tab="B", mode="replace"))[
            "software"
        ]
    spots = desktop.SampleSpots()
    area = spots.create_region((2, 2), {
        **default_region("rectangle"), "points": [[1, 1], [3, 3]],
    }, spots.native_size)
    exclusion = spots.create_region((3, 3), {
        **default_region("rectangle"), "points": [[3, 3], [3, 3]], "exclude": True,
    }, spots.native_size)
    renderer.set_pipeline(document)
    try:
        rendered = renderer.render_detailed(gradient_frame())
        assert rendered.measurements_valid and rendered.measurement_mapping_valid
        y, x = np.indices((192, 256))
        np.testing.assert_array_equal(rendered.temperatures_celsius, 10 + x / 4 + y / 8)
        readings = spots.readings(rendered.temperatures_celsius)
        assert readings[area].count == 8
        assert readings[area].graph_values() == (10.375, 10.703125, 11)
        assert exclusion not in readings
        assert dict(spots.graph_readings(rendered.temperatures_celsius))[0, area - 1] == (
            10.375, 10.703125, 11,
        )
    finally:
        renderer.pipeline_processor.close()


@pytest.mark.usefixtures("viewer")
def test_pipeline_edits_keep_drawing_and_saved_regions_in_native_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Draw through warps, keep geometry on mirror edits, and rotate only with the viewer control.
    """
    monkeypatch.setattr("topdon_duo.desktop.SpotsPanel", Mock)
    session = desktop.DesktopSession(desktop, [])
    session.viewer = Mock()
    document = raw_pipeline(node("software", "transform", width=50, position_x=40, skew_x=20))
    session.set_pipeline(document)
    session.rendered = session.renderer.render_detailed(gradient_frame(), image_processing=False)
    assert session.spots_state()["mapping_available"]
    session.begin_measurement_drawing({"kind": "rectangle", "orientation": [0, False, False]})
    session.measurement_drawing.pointer("press", "left", (1, 1))
    session.measurement_drawing.pointer("release", "left", (3, 3))
    mirrored = deepcopy(document)
    mirrored["software"][2]["params"]["horizontal"] = True
    session.set_pipeline(mirrored)
    session.reconcile_measurement_drawing()
    assert session.measurement_drawing.kind == "rectangle"
    session.measurement_drawing_action("draw_finish")
    before = session.spots.saved_state()
    session.set_view_setting("mirror_vertical", True)
    assert session.spots.saved_state() == before
    session.set_pipeline(document)
    assert session.spots.saved_state() == before
    session.rotate_view()
    assert session.spots.saved_state(session.renderer.rotation) == before
    rotated = session.renderer.render_detailed(gradient_frame(), image_processing=False)
    assert session.spots.readings(rotated.temperatures_celsius)[1].graph_values() == (
        10.375, 10.75, 11.125,
    )
