import csv
import threading
import time
from collections import deque
from dataclasses import replace
from itertools import pairwise
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
        assert len(worker._master) == 304
        assert worker._master[0][0] == 0
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


def test_interval_change_wakes_worker_and_controls_csv_cadence(tmp_path):
    worker = graphs.GraphWorker()
    try:
        worker.set_interval(60)
        worker.submit(snapshot())
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
        path = worker.start_logging(tmp_path / "interval.csv")
        with pytest.raises(ValueError, match="Stop logging"):
            worker.set_interval(0.1)
        worker.stop_logging()
        worker.set_interval(0.1)
        worker.start_logging(path)
        with worker._condition:
            assert worker._condition.wait_for(lambda: len(worker._master) >= 4, timeout=2)
            times = [stamp for stamp, _ in worker._master]
            assert worker._master.maxlen is None
        worker.stop_logging()
        assert all(0.08 <= b - a < 1 for a, b in pairwise(times))
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 12
        elapsed = [float(rows[i]["elapsed_seconds"]) for i in (0, 4, 8)]
        assert all(0.08 <= b - a < 1 for a, b in pairwise(elapsed))
    finally:
        worker.close()


def test_interval_editor_applies_valid_values_and_preserves_invalid_input():
    editor = graphs.GraphIntervalEditor()
    editor.begin()
    for character in "0.25":
        editor.key(ord(character))
    assert editor.key(13) == 0.25
    editor.begin()
    editor.key(ord("0"))
    with pytest.raises(ValueError):
        editor.key(13)
    assert editor.text == "0"
    editor.key(27)
    assert editor.text is None
    editor.begin()
    editor.key(8)
    assert editor.text == ""


@pytest.mark.parametrize("value", [True, "1", 0, -1, 61, float("nan"), float("inf")])
def test_invalid_interval_does_not_change_worker(value):
    worker = graphs.GraphWorker()
    try:
        with pytest.raises((ValueError, TypeError)):
            worker.set_interval(value)
        assert worker._interval == 0.5
    finally:
        worker.close()


def test_history_compression_retains_extrema_gaps_recent_samples_and_session_start():
    history = deque()
    recent = []
    for index in range(20_000):
        values = (float(index % 20), float(index % 31), float(index % 47), float(index % 59))
        if index == 10:
            values = (-1000, 2000, -3000, 4000)
        if index == 20:
            values = (float("nan"),) * 4
        history.append((index, values))
        graphs.compress_history(history)
        assert len(history) <= graphs.MAX_HISTORY_SAMPLES
        if index >= 20_000 - graphs.MAX_HISTORY_SAMPLES // 2:
            recent.append((index, values))
    assert history[0][0] == 0 and history[-1][0] == 19_999
    assert list(history)[-len(recent) :] == recent
    values = np.array([value for _, value in history])
    assert np.nanmin(values[:, 0]) == -1000
    assert np.nanmax(values[:, 1]) == 2000
    assert np.nanmin(values[:, 2]) == -3000
    assert np.nanmax(values[:, 3]) == 4000
    assert any(stamp == 20 and not np.isfinite(value).any() for stamp, value in history)
    assert all(a[0] <= b[0] for a, b in pairwise(history))


def test_all_graphs_share_expanding_session_time_range(monkeypatch):
    chart = Mock()
    text = Mock(wraps=graphs.cv2.putText)
    monkeypatch.setattr(graphs, "_draw_chart", chart)
    monkeypatch.setattr(graphs.cv2, "putText", text)
    graphs.render_graphs(
        snapshot((((0, 0), 22),)),
        deque([(0, (18, 20, 24, 21)), (300, (18, 20, 24, 21))]),
        {(0, 0): deque([(275, (22,))])},
        300,
    )
    assert [call.args[-1] for call in chart.call_args_list] == [300, 300]
    assert "History | 5.0 min" in [call.args[1] for call in text.call_args_list]


def test_compressed_nan_buckets_do_not_join_across_measurement_gaps():
    history = deque((index, (float(index) if index % 2 else float("nan"),)) for index in range(500))
    graphs.compress_history(history, limit=128)
    assert len(history) <= 128
    for left, right in pairwise(history):
        if np.isfinite(left[1]).all() and np.isfinite(right[1]).all():
            assert right[0] - left[0] <= 1


def test_reset_clears_histories_without_changing_settings_or_sampling_cadence(tmp_path):
    worker = graphs.GraphWorker()
    try:
        worker.set_interval(60)
        settings = worker._settings.copy()
        worker.submit(snapshot((((0, 0), 22),)))
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            assert worker._master and worker._spots
        worker.clear_history()
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            assert not worker._master and not worker._spots
            assert worker._latest.spots == (((0, 0), 22),)
            assert worker._interval == 60 and worker._settings == settings
        path = worker.start_logging(tmp_path / "reset.csv")
        before = path.read_bytes()
        with pytest.raises(ValueError, match="Stop logging"):
            worker.clear_history()
        assert worker.logging and path.read_bytes() == before
    finally:
        worker.close()


def test_reset_does_not_publish_an_inflight_render_with_old_data(monkeypatch):
    started, release = threading.Event(), threading.Event()
    real_render = graphs.render_graphs
    count = 0

    def render(*args):
        nonlocal count
        count += 1
        if count == 1:
            started.set()
            assert release.wait(timeout=2)
        return real_render(*args)

    monkeypatch.setattr(graphs, "render_graphs", render)
    worker = graphs.GraphWorker()
    try:
        worker.set_interval(60)
        worker.submit(snapshot())
        assert started.wait(timeout=2)
        worker.clear_history()
        release.set()
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            assert count == 2 and not worker._master
    finally:
        release.set()
        worker.close()
