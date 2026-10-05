import csv
import threading
import time
from collections import deque
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from topdon_duo import graphs


def snapshot(spots=(), unit="C", size=(600, 650)):
    return graphs.GraphSnapshot((18, 20, 24, 21), spots, size, unit)


def test_worker_samples_on_own_thread_at_half_second_intervals_and_pauses(monkeypatch):
    real_render = graphs.render_graphs
    render = Mock(wraps=real_render)
    monkeypatch.setattr(graphs, "render_graphs", render)
    worker = graphs.GraphWorker()
    try:
        with worker._condition:
            assert worker._image is None and not worker._master
        worker.submit(snapshot())
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            first_image = worker._image
        assert worker._thread.ident != threading.get_ident()
        # Repeated submitted camera frames must not drive graph refreshes at video rate.
        for _ in range(20):
            worker.submit(snapshot())
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not first_image, timeout=2)
            times = [stamp for stamp, _ in worker._master]
        assert len(times) == 2 and times[1] - times[0] >= 0.45
        worker.pause()
        count = render.call_count
        with worker._condition:
            paused_image = worker._image
        time.sleep(0.6)
        assert render.call_count == count
        assert len(worker._master) == 2
        assert worker._image is paused_image
        worker.submit(snapshot())
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not paused_image, timeout=2)
    finally:
        worker.close()
    assert not worker._thread.is_alive()


def test_spot_histories_survive_reorientation_but_not_clear_and_replacement():
    worker = graphs.GraphWorker()
    try:
        # Stable generation/index identities are independent of displayed pixel coordinates.
        worker._sample(snapshot((((0, 0), 22),)), 0)
        worker._sample(snapshot((((0, 0), 23), ((0, 1), 24))), 0.5)
        assert list(worker._spots[0, 0]) == [(0, (22,)), (0.5, (23,))]
        worker._sample(snapshot((((1, 0), 25),)), 1)
        assert set(worker._spots) == {(1, 0)}
        assert list(worker._spots[1, 0]) == [(1, (25,))]
        worker._sample(snapshot(), 1.5)
        assert worker._spots == {}
        for index in range(300):
            worker._sample(snapshot(), index + 2)
        assert len(worker._master) == graphs.HISTORY_SAMPLES
    finally:
        worker.close()


def test_graphs_share_height_and_convert_entire_history_to_selected_unit(monkeypatch):
    charts = []
    real_chart = graphs._draw_chart
    monkeypatch.setattr(
        graphs,
        "_draw_chart",
        lambda canvas, *args: (charts.append(canvas.shape), real_chart(canvas, *args)),
    )
    text = Mock(wraps=graphs.cv2.putText)
    monkeypatch.setattr(graphs.cv2, "putText", text)
    data = snapshot((((0, 0), 20), ((0, 1), 25)), "F")
    master = deque([(0, data.stats), (0.5, (19, 21, 25, 22))])
    spots = {(0, 0): deque([(0, (20,)), (0.5, (21,))]), (0, 1): deque([(0.5, (25,))])}
    image = graphs.render_graphs(data, master, spots, 0.5)
    assert image.shape == (650, 600, 3)
    assert max(shape[0] for shape in charts) - min(shape[0] for shape in charts) <= 1
    assert sum(shape[0] for shape in charts) == 650 - 34
    labels = [call.args[1] for call in text.call_args_list]
    assert "Master stats (F)" in labels and "Spot 1 (F)" in labels and "Spot 2 (F)" in labels
    assert "Temp 69.8" in labels and "Temp 77.0" in labels
    # Very short panels and invalid samples must remain renderable.
    graphs.render_graphs(replace(data, size=(120, 50)), master, spots, 0.5)
    invalid = deque([(0.5, (np.nan, np.inf, -np.inf, np.nan))])
    graphs.render_graphs(snapshot(), invalid, {}, 0.5)


def test_csv_logging_tracks_units_and_spot_generations_and_blocks_pause(tmp_path):
    worker = graphs.GraphWorker()
    try:
        path = worker.start_logging(tmp_path / "temperatures")
        assert path.suffix == ".csv" and worker.logging
        worker.submit(snapshot((((0, 0), 20), ((0, 1), 25)), "F"))
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            first = worker._image
        with pytest.raises(ValueError, match="Stop logging"):
            worker.pause()
        worker.submit(snapshot((((1, 0), 30),), "C"))
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not first, timeout=2)
        assert worker.stop_logging() == path
        assert not worker.logging
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 11
        first_spot = next(row for row in rows if row["series_id"] == "spot.0.1")
        assert float(first_spot["temperature_celsius"]) == 20
        assert float(first_spot["temperature_display"]) == 68
        assert first_spot["display_unit"] == "F"
        replacement = next(row for row in rows if row["series_id"] == "spot.1.1")
        assert float(replacement["temperature_display"]) == 30
        assert replacement["display_unit"] == "C"
        assert replacement["timestamp_utc"].endswith("+00:00")
        assert float(replacement["elapsed_seconds"]) >= 0.45
        assert {row["series_id"] for row in rows if row["series_id"].startswith("scene.")} == {
            "scene.minimum",
            "scene.average",
            "scene.maximum",
            "scene.center",
        }
        worker.pause()
    finally:
        worker.close()


def test_logging_write_failure_stops_logging_and_preserves_error(tmp_path):
    worker = graphs.GraphWorker()
    try:
        worker.start_logging(tmp_path / "failure.csv")
        stream = worker._log_file
        failing = Mock(wraps=stream)
        failing.flush.side_effect = OSError("disk full")
        worker._log_file = failing
        with worker._condition:
            worker._write_log(snapshot(), worker._log_started + 0.5)
        assert not worker.logging
        assert stream.closed
        assert "disk full" in worker.take_logging_error()
        assert worker.take_logging_error() is None
    finally:
        worker.close()


def test_worker_shutdown_finalizes_logfile(tmp_path):
    worker = graphs.GraphWorker()
    path = worker.start_logging(tmp_path / "exit.csv")
    stream = worker._log_file
    with worker._condition:
        worker._write_log(snapshot(), worker._log_started + 0.5)
    worker.close()
    assert stream.closed and not worker.logging
    with path.open(newline="") as source:
        assert len(list(csv.DictReader(source))) == 4


def test_adding_csv_extension_does_not_overwrite_an_unselected_file(tmp_path):
    path = tmp_path / "existing.csv"
    path.write_text("previous data")
    worker = graphs.GraphWorker()
    try:
        with pytest.raises(FileExistsError):
            worker.start_logging(tmp_path / "existing")
        assert path.read_text() == "previous data"
        assert not worker.logging
    finally:
        worker.close()


def test_capture_resizes_real_worker_cache_without_changing_live_graphs():
    worker = graphs.GraphWorker()
    live_size = (576, 850)
    capture_size = (576, 800)
    try:
        assert not worker.image(capture_size, resize=True).any()
        worker.submit(snapshot(size=live_size))
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            cached = worker._image
        worker.pause()
        captured = worker.image(capture_size, resize=True)
        assert captured.shape == (800, 576, 3)
        assert captured.any()
        assert np.array_equal(
            captured, graphs.cv2.resize(cached, capture_size, interpolation=graphs.cv2.INTER_AREA)
        )
        assert worker.image(live_size) is cached
        assert not worker.image(capture_size).any()
        assert len(worker._master) == 1
    finally:
        worker.close()


def test_named_regions_reach_graph_titles_and_csv_without_changing_series_ids(
    tmp_path, monkeypatch
):
    labels = Mock(wraps=graphs.cv2.putText)
    monkeypatch.setattr(graphs.cv2, "putText", labels)
    data = replace(
        snapshot((((0, 0), 23.5), ((0, 1), 24.0))),
        spot_names=(((0, 0), "Left hand, thumb"), ((0, 1), "Motor")),
    )
    worker = graphs.GraphWorker()
    try:
        path = worker.start_logging(tmp_path / "regions.csv")
        worker.submit(data)
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
        worker.stop_logging()
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        named = {row["series_id"]: row for row in rows if row["series_id"].startswith("spot.")}
        assert named["spot.0.1"]["series"] == "Left hand, thumb (Spot 1)"
        assert named["spot.0.2"]["series"] == "Motor (Spot 2)"
        assert float(named["spot.0.1"]["temperature_celsius"]) == 23.5
        text = [call.args[1] for call in labels.call_args_list]
        assert "Left hand, thumb (Spot 1) (C)" in text
        assert "Motor (Spot 2) (C)" in text
    finally:
        worker.close()
