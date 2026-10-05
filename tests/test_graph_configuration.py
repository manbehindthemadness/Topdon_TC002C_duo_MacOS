import subprocess
import sys
from unittest.mock import Mock

import pytest
from test_capture_panel import popup_environment

from topdon_duo import graphs
from topdon_duo.graph_settings import GRAPH_DEFAULTS, validate_graph_settings
from topdon_duo.settings_preferences import load_settings, save_settings


@pytest.mark.parametrize(
    "settings",
    [
        {"range_mode": "bad"},
        {"range_seconds": float("nan")},
        {"history_points": True},
        {"history_points": 255},
        {"compression": 1},
    ],
)
def test_graph_settings_validation(settings):
    with pytest.raises((ValueError, TypeError)):
        validate_graph_settings(settings)


def test_graph_configuration_resizes_history_and_locks_during_logging(tmp_path):
    worker = graphs.GraphWorker()
    data = graphs.GraphSnapshot((18, 20, 24, 21), (), (600, 650))
    try:
        for index in range(300):
            worker._sample(data, index)
        worker.configure({"compression": False, "history_points": 256})
        assert len(worker._master) == 256 and worker._master[0][0] == 44
        worker.start_logging(tmp_path / "sample.csv")
        with pytest.raises(ValueError, match="Stop logging"):
            worker.configure({"range_mode": "fixed"})
    finally:
        worker.close()


def test_range_change_redraws_without_an_extra_temperature_sample(monkeypatch):
    real = graphs.render_graphs
    render = Mock(wraps=real)
    monkeypatch.setattr(graphs, "render_graphs", render)
    worker = graphs.GraphWorker()
    try:
        worker.set_interval(60)
        worker.submit(graphs.GraphSnapshot((18, 20, 24, 21), (), (600, 650)))
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not None, timeout=2)
            first = worker._image
        worker.configure({"range_mode": "fixed", "range_seconds": 300})
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not first, timeout=2)
            assert len(worker._master) == 1
        assert render.call_args.args[0].history_range == 300
    finally:
        worker.close()


def test_graph_preferences_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    settings = {
        **GRAPH_DEFAULTS,
        "range_mode": "fixed",
        "range_seconds": 600,
        "history_points": 8192,
    }
    save_settings({"graph_settings": settings})
    assert load_settings()["graph_settings"] == settings
    save_settings({"graph_settings": {"compression": "yes"}, "show_graph": True})
    assert load_settings() == {"display": {}, "hardware": {}, "show_graph": True}


def test_graph_configuration_popup_controls_and_logging_lock():
    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.graph_window import GraphWindow
from topdon_duo.graph_settings import GRAPH_DEFAULTS
app = QApplication([])
messages = []
window = GraphWindow(messages.append)
assert window.minutes.isEnabled() and window.minutes.value() == 10
assert window.range_mode.currentData() == "fixed"
window.range_mode.setCurrentIndex(0)
assert not window.minutes.isEnabled()
window.range_mode.setCurrentIndex(1)
window.minutes.setValue(5)
window.compression.setChecked(False)
window.points.setValue(8192)
settings = messages[-1]["settings"]
assert settings == {"range_mode": "fixed", "range_seconds": 300, "compression": False, "history_points": 8192}
count = len(messages)
window.update_state({"settings": settings, "locked": True})
assert len(messages) == count
assert all(not control.isEnabled() for control in (window.range_mode, window.minutes, window.compression, window.points))
window.show()
assert window.close_button.isEnabled()
window.close_button.click()
assert not window.isVisible()
assert len(messages) == count
window.update_state({"settings": settings})
assert window.minutes.isEnabled()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
