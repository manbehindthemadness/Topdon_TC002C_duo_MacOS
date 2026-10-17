"""
Graph geometry refreshes independently of sampling and rejects obsolete worker output.
"""

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from based_thermals import graphs


def test_resize_redraws_without_sampling_or_logging(tmp_path: Path) -> None:
    """
    Resize before a long sampling deadline while preserving histories and CSV cadence.
    """
    worker = graphs.GraphWorker()
    snapshot = graphs.GraphSnapshot((18, 20, 24, 21), (), (384, 650))
    try:
        worker.set_interval(60)
        worker.submit(snapshot)
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            history = list(worker._master)
        path = worker.start_logging(tmp_path / "temperatures.csv")
        for size in ((1000, 650), (600, 900), (384, 650)):
            worker.submit(replace(snapshot, size=size))
            with worker._condition:
                assert worker._condition.wait_for(
                    lambda expected=size: worker._image.shape[:2] == expected[::-1], timeout=2,
                )
                assert list(worker._master) == history
            assert len(path.read_text().splitlines()) == 1
    finally:
        worker.close()


@pytest.mark.parametrize("size", [(384, 650), (1000, 650)])
def test_reopen_refreshes_immediately_and_preserves_pause_gap(size: tuple[int, int]) -> None:
    """
    Reopening before the next sample refreshes both unchanged and changed pane sizes.
    """
    worker = graphs.GraphWorker()
    snapshot = graphs.GraphSnapshot((18, 20, 24, 21), (), (384, 650))
    try:
        worker.set_interval(60)
        worker.submit(snapshot)
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            previous = worker._image
            original_sample = worker._master[0]
        worker.pause()
        worker.submit(replace(snapshot, size=size))
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not previous, timeout=2)
            assert worker._image.shape[:2] == size[::-1]
            assert len(worker._master) == 3
            assert worker._master[0] == original_sample
            assert np.isnan(worker._master[1][1]).all()
            assert worker._master[2][1] == snapshot.stats
    finally:
        worker.close()


@pytest.mark.parametrize("reopen", [False, True])
def test_obsolete_inflight_render_cannot_publish(
    monkeypatch: pytest.MonkeyPatch, reopen: bool,
) -> None:
    """
    Reject an obsolete render even when resizing returns to its size or reopens the pane.
    """
    obsolete_started, release_obsolete = threading.Event(), threading.Event()
    current_started, release_current = threading.Event(), threading.Event()
    calls = 0

    def render(
        pane: graphs.GraphSnapshot, _master: graphs.History,
        _spots: dict[tuple[int, int], graphs.History], _now: float,
    ) -> np.ndarray:
        """
        Hold renders at publication boundaries and identify each result by its pixels.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            obsolete_started.set()
            assert release_obsolete.wait(3)
        elif calls == 3:
            current_started.set()
            assert release_current.wait(3)
        width, height = pane.size
        return np.full((height, width, 3), calls, np.uint8)

    monkeypatch.setattr(graphs, "render_graphs", render)
    worker = graphs.GraphWorker()
    snapshot = graphs.GraphSnapshot((18, 20, 24, 21), (), (384, 650))
    try:
        worker.set_interval(60)
        worker.submit(snapshot)
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            original = worker._image
        worker.submit(replace(snapshot, size=(600, 650)))
        assert obsolete_started.wait(2)
        if reopen:
            worker.pause()
        else:
            worker.submit(snapshot)
        worker.submit(replace(snapshot, size=(600, 650)))
        release_obsolete.set()
        assert current_started.wait(2)
        with worker._condition:
            assert worker._image is original
        release_current.set()
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not original, timeout=2)
            assert np.all(worker._image == 3)
            assert worker._image.shape[:2] == (650, 600)
    finally:
        release_obsolete.set()
        release_current.set()
        worker.close()
