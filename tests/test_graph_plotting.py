"""Visible-range and pixel-reduction guarantees for chart rendering."""

import time
from collections import deque
from collections.abc import Iterator
from itertools import pairwise
from unittest.mock import Mock

import numpy as np
import pytest

from topdon_duo import graphs
from topdon_duo.graph_plotting import HistorySample, pixel_curve_samples, visible_history


class WatchedHistory(deque):
    """
    Record reverse traversal and reject a full scan of retained history.
    """

    def __init__(self) -> None:
        """
        Build ordered history with a short visible tail.
        """
        super().__init__((float(index), (float(index),)) for index in range(10_000))
        self.visited: list[float] = []

    def __iter__(self) -> Iterator[HistorySample]:
        """
        Fail if visible selection traverses the entire history forwards.
        """
        raise AssertionError("Older history must not be scanned")

    def __reversed__(self) -> Iterator[HistorySample]:
        """
        Track exactly the samples visited from the newest end.
        """
        for sample in super().__reversed__():
            self.visited.append(sample[0])
            yield sample


def test_visible_selection_stops_at_first_older_sample() -> None:
    """
    Select the inclusive visible boundary without scanning older retained data.
    """
    history = WatchedHistory()

    result = visible_history(history, 9998., 3.)

    assert [stamp for stamp, _ in result] == [9995., 9996., 9997., 9998.]
    assert history.visited == [9999., 9998., 9997., 9996., 9995., 9994.]
    assert len(history) == 10_000


def test_visible_selection_reuses_tuple_and_session_keeps_all_samples() -> None:
    """
    Reuse worker snapshots and keep full retained history for session plots.
    """
    history = ((0., (1.,)), (10., (2.,)), (20., (3.,)))

    assert visible_history(history, 20., 20.) is history
    assert visible_history(history, 20., None) is history
    assert visible_history(history, 30., 1.) == ()
    assert visible_history((), 30., None) == ()


def test_single_pixel_retains_endpoints_extrema_and_original_gap_splits() -> None:
    """
    Preserve spikes and gaps even when many samples collapse into one column.
    """
    values = np.array([5., 6., np.nan, 7., 1., 8., 10., np.nan, 4., 5.])

    indices, boundaries = pixel_curve_samples(np.zeros(len(values), dtype=int), values)

    np.testing.assert_array_equal(indices, [0, 4, 6, 9])
    np.testing.assert_array_equal(boundaries, [1, 3])
    np.testing.assert_array_equal(values[indices], [5., 1., 10., 5.])


def test_dense_series_keeps_every_pixel_extremum_with_bounded_output() -> None:
    """
    Bound output while preserving independently measured extrema and endpoints.
    """
    rng = np.random.default_rng(42)
    x = np.repeat(np.arange(21), 100)
    values = rng.normal(size=len(x))
    values[::13] = np.nan

    indices, boundaries = pixel_curve_samples(x, values)

    assert len(indices) <= 4 * 21
    assert np.all(np.diff(indices) > 0)
    for column in range(21):
        original = np.flatnonzero((x == column) & np.isfinite(values))
        kept = indices[x[indices] == column]
        assert original[0] in kept and original[-1] in kept
        assert values[kept].min() == values[original].min()
        assert values[kept].max() == values[original].max()
    for run in np.split(indices, boundaries):
        for left, right in pairwise(run):
            assert np.isfinite(values[left:right + 1]).all()


def test_invalid_series_has_no_pixel_geometry() -> None:
    """
    Leave unavailable histories empty rather than creating invalid coordinates.
    """
    indices, boundaries = pixel_curve_samples(np.arange(3), np.array([np.nan, np.inf, -np.inf]))

    assert not indices.size and not boundaries.size


def test_worker_filters_render_snapshot_without_trimming_retained_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Render the visible tail while keeping older measurements for range changes.
    """
    render = Mock(wraps=graphs.render_graphs)
    monkeypatch.setattr(graphs, "render_graphs", render)
    worker = graphs.GraphWorker()
    data = graphs.GraphSnapshot((1., 2., 3., 4.), (((0, 0), 5.),), (400, 400))
    now = time.monotonic()
    try:
        worker.configure({"range_seconds": 6.})
        worker._sample(data, now - 1000)
        worker._sample(data, now - 1)
        worker.submit(data)
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            assert len(worker._master) == 3 and worker._master[0][0] == now - 1000
            assert len(worker._spots[0, 0]) == 3
            first_image = worker._image
            worker.set_interval(60)
        assert len(render.call_args.args[1]) == 2
        assert len(render.call_args.args[2][0, 0]) == 2

        worker.configure({"range_mode": "session"})
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not first_image, timeout=2)
            assert len(worker._master) == 3
        assert len(render.call_args.args[1]) == 3
        assert len(render.call_args.args[2][0, 0]) == 3
    finally:
        worker.close()
