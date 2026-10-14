"""Deterministic compaction races without blocking the UI-shared condition lock."""

import threading
from collections import deque

import numpy as np
import pytest

from based_thermals import graphs


@pytest.mark.parametrize("mutation", ["append", "reset", "settings"])
def test_compaction_releases_ui_lock_and_rejects_stale_history(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """
    Permit image reads during slow compression and preserve concurrent changes.
    """
    worker = graphs.GraphWorker()
    worker.configure({"history_points": 256})
    snapshot = graphs.GraphSnapshot((1, 2, 3, 4), (), (20, 20))
    for index in range(257):
        worker._sample(snapshot, float(index), compact=False)
    worker._image = np.zeros((20, 20, 3), np.uint8)
    entered, release, read_done = threading.Event(), threading.Event(), threading.Event()
    real_compress = graphs.compress_history

    def slow_compress(history: deque, limit: int) -> None:
        """
        Hold expensive work outside the condition until the test mutates state.
        """
        entered.set()
        assert release.wait(3)
        real_compress(history, limit)

    def read_image() -> None:
        """
        Demonstrate that cached images remain accessible while compression is paused.
        """
        worker.image((20, 20))
        read_done.set()

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(graphs, "compress_history", slow_compress)
    compact = threading.Thread(target=worker._compact_histories)
    reader = threading.Thread(target=read_image)
    try:
        compact.start()
        assert entered.wait(2)
        reader.start()
        assert read_done.wait(1)
        if mutation == "append":
            with worker._condition:
                worker._sample(snapshot, 1000.0, compact=False)
        elif mutation == "reset":
            worker.clear_history()
        else:
            worker.configure({"history_points": 256, "compression": False})
        release.set()
        compact.join(2)
        assert not compact.is_alive()
        if mutation == "append":
            assert worker._master[-1][0] == 1000
            assert len(worker._master) <= 256
        elif mutation == "reset":
            assert not worker._master
        else:
            assert len(worker._master) == 256 and worker._master[0][0] == 1
    finally:
        release.set()
        compact.join(3)
        if reader.ident is not None:
            reader.join(3)
        worker.close()
