"""Regression coverage for display-only sample smoothing."""

from unittest.mock import Mock

import numpy as np
import pytest

from based_thermals import graphs
from based_thermals.graph_plotting import visible_history
from based_thermals.graph_smoothing import interpolate_samples, rolling_average


def test_average_restarts_independently_at_gaps_and_preserves_input() -> None:
    """
    Each series warms up independently without averaging across unavailable data.
    """
    values = np.array([[2, 10], [4, np.nan], [8, 30], [np.nan, 50], [20, 70]])
    original = values.copy()
    result = rolling_average(values, 2)
    np.testing.assert_allclose(result, [[2, 10], [3, np.nan], [6, 30], [np.nan, 40], [20, 60]])
    np.testing.assert_array_equal(values, original)
    assert rolling_average(values, 1) is values


@pytest.mark.parametrize("window", [2, 5, 256])
def test_average_matches_independent_segment_reference(window: int) -> None:
    """
    Compare vectorized results with a direct trailing-list definition across many gaps.
    """
    values = np.random.default_rng(28).uniform(-40, 100, (512, 3))
    values[1::2, 0] = np.nan
    values[::7, 1] = np.inf
    values[200:300, 2] = np.nan
    expected = np.full_like(values, np.nan)
    for column in range(3):
        trailing = []
        for index, value in enumerate(values[:, column]):
            if not np.isfinite(value):
                trailing.clear()
                continue
            trailing.append(float(value))
            expected[index, column] = np.mean(trailing[-window:])
    np.testing.assert_allclose(rolling_average(values, window), expected, atol=1e-10)


def test_visible_average_uses_samples_before_the_time_window() -> None:
    """
    Preserve preceding context so a moving time window does not reset averaging.
    """
    history = tuple((float(index), (float(index),)) for index in range(10))
    samples = visible_history(history, 9, 2, 2)
    assert [stamp for stamp, _ in samples] == [5, 6, 7, 8, 9]
    averaged = rolling_average(np.array([values for _, values in samples]), 3)
    np.testing.assert_allclose(averaged[-3:, 0], [6, 7, 8])


def test_interpolation_preserves_knots_and_extrema_with_uneven_timestamps() -> None:
    """
    Intermediate samples remain within each measured interval and preserve knots.
    """
    stamps = np.array([0.0, 0.2, 3.0, 4.0, 10.0])
    values = np.array([10.0, 30.0, 12.0, 12.0, 50.0])
    times, curve = interpolate_samples(stamps, values, 7)
    assert len(times) == (len(stamps) - 1) * 8 + 1
    np.testing.assert_array_equal(times[::8], stamps)
    np.testing.assert_array_equal(curve[::8], values)
    assert np.all(np.diff(times) > 0)
    for index in range(len(stamps) - 1):
        segment = curve[index * 8:(index + 1) * 8 + 1]
        assert segment.min() >= min(values[index:index + 2])
        assert segment.max() <= max(values[index:index + 2])


def test_interpolation_handles_disabled_single_and_duplicate_samples() -> None:
    """
    Degenerate geometry and disabled interpolation retain their original data.
    """
    for stamps, values, count in (
        (np.array([0, 1]), np.array([10, 20]), 0),
        (np.array([0]), np.array([10]), 3),
        (np.array([0, 0]), np.array([10, 20]), 3),
    ):
        times, curve = interpolate_samples(stamps, values, count)
        np.testing.assert_array_equal(times, stamps)
        np.testing.assert_array_equal(curve, values)


def test_smoothed_chart_preserves_gap_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Interpolated chart geometry never joins runs separated by missing readings.
    """
    polylines = Mock()
    monkeypatch.setattr(graphs.cv2, "polylines", polylines)
    graphs._draw_history_curves(
        np.zeros((200, 400, 3), dtype=np.uint8), np.arange(5, dtype=float),
        np.array([[10], [20], [np.nan], [30], [40]]),
        (0, 399, 0, 199), 0, 50, 4, 4, 8,
    )
    curves = polylines.call_args.args[1]
    assert len(curves) == 2
    assert curves[0][:, 0].max() <= 100
    assert curves[1][:, 0].min() >= 299


def test_chart_averages_before_fahrenheit_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Display conversion follows averaging without modifying the Celsius history.
    """
    draw = Mock()
    monkeypatch.setattr(graphs, "_draw_history_curves", draw)
    history = ((0.0, (10.0,)), (1.0, (20.0,)), (2.0, (30.0,)))
    graphs._draw_chart(
        np.zeros((200, 400, 3), dtype=np.uint8), "Spot", history, ("Temp",),
        "F", 2, 1, averaging_samples=2, interpolation_samples=4,
    )
    np.testing.assert_allclose(draw.call_args.args[2][:, 0], [59, 77])
    assert draw.call_args.args[-1] == 4
    assert history[-1][1] == (30.0,)
