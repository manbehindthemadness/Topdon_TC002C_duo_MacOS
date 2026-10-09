"""
Public measurement-cache operations retain sensor data across calibration pauses.
"""

import numpy as np
from test_camera import make_frame

from topdon_duo.render import ThermalRenderer


def test_restart_keeps_held_counts_and_replaces_average_on_next_frame() -> None:
    """
    Requesting calibration preserves the held average until a new valid frame arrives.
    """
    renderer = ThermalRenderer(scale=1, ambient_celsius=None)
    renderer.native_temperatures = True
    first = make_frame(20_000)
    renderer.render_detailed(first)
    average = renderer.averaged_raw_counts
    assert average is not None
    renderer.request_measurement_restart()
    assert renderer.averaged_raw_counts is average
    assert renderer.last_valid_frame == first
    renderer.render_detailed(make_frame(21_000))
    assert np.all(renderer.averaged_raw_counts == 21_000)


def test_reset_discards_average_and_preserves_last_valid_frame() -> None:
    """
    A changed measurement setting starts a fresh average without losing the display fallback.
    """
    renderer = ThermalRenderer(scale=1, ambient_celsius=None)
    renderer.native_temperatures = True
    frame = make_frame()
    renderer.render_detailed(frame)
    renderer.reset_measurement_average()
    assert renderer.averaged_raw_counts is None
    assert renderer.last_valid_frame == frame
