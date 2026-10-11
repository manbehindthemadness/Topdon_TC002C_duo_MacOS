"""
Measurement editor commands, native geometry gestures, locks and graph integration.
"""

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.qt_process import run_popup

from topdon_duo import desktop, graphs


def test_measurements_popup_editing_and_locks(tmp_path: Path) -> None:
    """
    Exercise the actual Qt controls, shape topology and dragging in an isolated process.
    """
    script = r"""
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from topdon_duo.measurements_window import MeasurementsWindow
from topdon_duo.measurements.paths import initial_region
from topdon_duo.measurements.schema import KINDS

app = QApplication([])
messages = []
window = MeasurementsWindow(messages.append)
window.show()
orientation = [0, False, False]
state = {"sensor_size": [32, 24], "orientation": orientation, "spots": [], "locked": False}
window.update_state(state)
for index, kind in enumerate(KINDS):
    window.kind.setCurrentIndex(index)
    window.create.click()
    command = messages[-1]
    assert command["action"] == "create" and command["region"]["kind"] == kind
    entry = {"number": index+1, "name": kind, "enabled": True,
             "pixel": command["pixel"], "region": command["region"]}
    state["spots"].append(entry)
    window.update_state(deepcopy(state))
    assert window._selected == index+1
    if kind == "spot":
        window.diameter.setValue(7)
        assert window.radius.value() == 3.5
        window.radius.setValue(2.5)
        assert window.diameter.value() == 5
        window.x.setValue(5)
        window.y.setValue(6)
    else:
        app.processEvents()
        start = window.preview.display_point(window.region["points"][0]).toPoint()
        target = window.preview.display_point((3, 4)).toPoint()
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(window.preview, target)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton, pos=target)
        assert abs(window.region["points"][0][0] - 3) < .2
        assert abs(window.region["points"][0][1] - 4) < .2
    if kind in ("polygon", "bezier"):
        original = window.points.rowCount()
        window.add_vertex.click()
        count = 3 if kind == "bezier" else 1
        assert window.points.rowCount() == original + count
        window.points.setCurrentCell(original, 1)
        window.remove_vertex.click()
        assert window.points.rowCount() == original
    window.exclude.setChecked(True)
    for control in window.label_fields.values():
        control.setChecked(False)
    window.name.setText("Edited")
    before = deepcopy(window.region)
    window.update_state(deepcopy(state))
    assert window.region == before and window.name.text() == "Edited"
    window.apply.click()
    configure, rename = messages[-2:]
    assert configure["action"] == "configure" and configure["orientation"] == orientation
    assert configure["region"]["exclude"] and configure["region"]["labels"] == []
    if kind == "spot":
        assert configure["pixel"] == [5, 6] and configure["region"]["radius"] == 2.5
    assert rename == {"action": "rename", "spot": index+1, "name": "Edited"}
    entry.update(region=configure["region"], pixel=configure["pixel"], name="Edited")
    window.update_state(deepcopy(state))
    assert not window.clear_history.isEnabled()

# Invalid coordinates cannot commit, and changing orientation drops stale edits.
window.points.item(0, 1).setText("not a coordinate")
count = len(messages)
window.apply.click()
assert len(messages) == count
state["orientation"] = [180, False, False]
window.update_state(deepcopy(state))
assert window.points.item(0, 1).text() != "not a coordinate"
state["locked"] = True
window.update_state(deepcopy(state))
for control in (window.create, window.placing, window.clear, window.delete, window.apply):
    assert not control.isEnabled()
    control.click()
assert len(messages) == count
assert not (window.items.item(0).flags() & Qt.ItemFlag.ItemIsUserCheckable)
window.close()
"""
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stderr


def test_area_graph_csv_gaps_and_independent_history_clear(tmp_path: Path) -> None:
    """
    Export min/average/max in Celsius and preserve three-column invalid-frame gaps.
    """
    worker = graphs.GraphWorker()
    data = graphs.GraphSnapshot(
        (10, 20, 30, 21),
        (((0, 0), (11, 19, 27)), ((0, 1), 15)),
        (600, 500),
        "F",
        (((0, 0), "Bearing"),),
    )
    try:
        worker._sample(data, 0)
        worker._sample(replace(data, measurements_valid=False), 0.5)
        assert len(worker._spots[0, 0][-1][1]) == 3
        assert all(np.isnan(v) for v in worker._spots[0, 0][-1][1])
        worker.clear_spot_history((0, 0))
        assert (0, 0) not in worker._spots and len(worker._spots[0, 1]) == 2
        assert len(worker._master) == 2
        path = worker.start_logging(tmp_path / "areas.csv")
        worker._write_log(data, worker._log_started)
        worker._write_log(replace(data, measurements_valid=False), worker._log_started + 0.5)
        with pytest.raises(ValueError, match="Stop logging"):
            worker.clear_spot_history((0, 1))
        worker.stop_logging()
        with path.open(newline="") as stream:
            rows = {row["series_id"]: row for row in csv.DictReader(stream)}
        assert len(rows) == 8
        assert rows["spot.0.1.min"]["series"] == "Bearing (Measurement 1) Min"
        for label, value in (("min", 11), ("avg", 19), ("max", 27)):
            row = rows[f"spot.0.1.{label}"]
            assert float(row["temperature_celsius"]) == value
            assert float(row["temperature_display"]) == pytest.approx(value * 1.8 + 32)
        assert float(rows["spot.0.2"]["temperature_celsius"]) == 15
        worker._sample(data, 1)
        point = replace(data, spots=(((0, 0), 13),))
        worker.submit(point)
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            assert len(worker._spots[0, 0][-1][1]) == 1
        worker.submit(data)
        with worker._condition:
            assert worker._condition.wait_for(lambda: bool(worker._spots.get((0, 0))), timeout=2)
            assert len(worker._spots[0, 0][-1][1]) == 3
        worker.pause()
        worker.submit(data)
        with worker._condition:
            assert worker._condition.wait_for(lambda: not worker._history_paused, timeout=2)
            assert all(len(values) == 3 for _stamp, values in worker._spots[0, 0])
    finally:
        worker.close()


def test_parent_measurement_commands_reject_locks_and_stale_geometry(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Apply popup commands through the real session without accessing a camera or Qt process.
    """
    from topdon_duo.measurements.paths import initial_region

    panel = Mock()
    panel.state = {}
    events: list[dict] = []
    observed: list[dict] = []

    def poll() -> list[dict]:
        """
        Consume commands queued by the simulated viewer event loop.
        """
        result = events.copy()
        events.clear()
        return result

    def update(state: dict) -> None:
        """
        Keep the latest authoritative geometry for subsequent commands.
        """
        panel.state = state
        observed.append(state)

    panel.poll.side_effect = poll
    panel.update.side_effect = update
    # The facade exports injectable dependencies dynamically.
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(desktop, "SpotsPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    center, region = initial_region("rectangle", (256, 192))
    step = 0

    def key(_delay: int) -> int:
        """
        Queue valid, stale and locked mutations and verify their parent results.
        """
        nonlocal step
        step += 1
        if step == 1:
            events.append(
                {
                    "action": "create",
                    "pixel": list(center),
                    "region": region,
                    "orientation": panel.state["orientation"],
                }
            )
        elif step == 2:
            assert len(panel.state["spots"]) == 1
            events.extend(
                [
                    {"action": "rename", "spot": 1, "name": "Bearing"},
                    {"action": "clear_history", "spot": 1},
                    {
                        "action": "configure",
                        "spot": 1,
                        "region": {**region, "exclude": True},
                        "orientation": [999, False, False],
                    },
                ]
            )
        elif step == 3:
            assert not panel.state["spots"][0]["region"]["exclude"]
            viewer.graphs.logging = True
            events.extend(
                [{"action": "clear_all"}, {"action": "enable", "spot": 1, "enabled": False}]
            )
        elif step == 4:
            assert len(panel.state["spots"]) == 1 and panel.state["spots"][0]["enabled"]
            viewer.graphs.logging = False
            events.append(
                {
                    "action": "configure",
                    "spot": 1,
                    "pixel": [10, 20],
                    "region": {**region, "exclude": True},
                    "orientation": panel.state["orientation"],
                }
            )
        elif step == 5:
            assert panel.state["spots"][0]["pixel"] == [10, 20]
            assert panel.state["spots"][0]["region"]["exclude"]
            events.append({"action": "clear_all"})
        elif step == 6:
            assert panel.state["spots"] == []
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    viewer.graphs.clear_spot_history.assert_called_once_with((0, 0))
    assert any(state["locked"] for state in observed)
