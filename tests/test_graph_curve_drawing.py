"""Regression checks for bounded chart drawing and measurement gaps."""

from unittest.mock import Mock

import numpy as np
import pytest

from topdon_duo import graphs


def test_curves_keep_gaps_and_isolated_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Separate valid runs and keep lone samples visible without bridging gaps.
    """
    lines = Mock()
    circles = Mock()
    monkeypatch.setattr(graphs.cv2, "polylines", lines)
    monkeypatch.setattr(graphs.cv2, "circle", circles)
    canvas = np.zeros((101, 81, 3), np.uint8)
    values = np.array([[0], [10], [np.nan], [5], [np.inf], [2], [3], [-np.inf], [10]])

    graphs._draw_history_curves(
        canvas, np.arange(9), values, (0, 80, 0, 100), 0, 10, 8, 8,
    )

    assert lines.call_count == 1
    curves = lines.call_args.args[1]
    assert len(curves) == 2
    np.testing.assert_array_equal(curves[0], [[0, 100], [10, 0]])
    np.testing.assert_array_equal(curves[1], [[50, 80], [60, 70]])
    assert [call.args[1] for call in circles.call_args_list] == [(30, 50), (80, 0)]


def test_dense_history_uses_one_native_curve_call_per_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Thousands of samples must not produce thousands of OpenCV drawing calls.
    """
    lines = Mock()
    circles = Mock()
    monkeypatch.setattr(graphs.cv2, "polylines", lines)
    monkeypatch.setattr(graphs.cv2, "circle", circles)
    canvas = np.zeros((101, 601, 3), np.uint8)
    values = np.tile(np.array([1., 2., 3., 4.]), (4096, 1))

    graphs._draw_history_curves(
        canvas, np.linspace(0, 600, 4096), values, (0, 600, 0, 100), 0, 5, 600, 600,
    )

    assert lines.call_count == 4
    circles.assert_not_called()
    for call in lines.call_args_list:
        assert len(call.args[1]) == 1
        assert call.args[1][0].shape[1] == 2
        assert len(call.args[1][0]) <= 4 * 601


def test_invalid_series_is_not_drawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A series containing only unavailable measurements contributes no pixels.
    """
    lines = Mock()
    circles = Mock()
    monkeypatch.setattr(graphs.cv2, "polylines", lines)
    monkeypatch.setattr(graphs.cv2, "circle", circles)

    graphs._draw_history_curves(
        np.zeros((101, 61, 3), np.uint8), np.arange(3),
        np.array([[np.nan], [np.inf], [-np.inf]]), (0, 60, 0, 100), 0, 10, 2, 2,
    )

    lines.assert_not_called()
    circles.assert_not_called()
