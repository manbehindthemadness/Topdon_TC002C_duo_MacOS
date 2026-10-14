import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_capture_panel import popup_environment

from based_thermals import graphs
from based_thermals.graph_settings import GRAPH_DEFAULTS, validate_graph_settings
from based_thermals.settings_preferences import load_settings, save_settings


@pytest.mark.parametrize(
    "settings",
    [
        {"range_mode": "bad"},
        {"range_seconds": float("nan")},
        {"history_points": True},
        {"history_points": 255},
        {"compression": 1},
        {"averaging_samples": 0},
        {"averaging_samples": True},
        {"averaging_samples": 257},
        {"interpolation_samples": -1},
        {"interpolation_samples": 33},
        {"interpolation_samples": 1.5},
    ],
)
def test_graph_settings_validation(settings: dict) -> None:
    """
    Reject malformed history and sample-based settings before applying them.
    """
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


def test_range_change_redraws_without_an_extra_temperature_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Changing display settings redraws history without recording an extra sample.
    """
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
        worker.configure({"range_mode": "fixed", "range_seconds": 300,
                          "averaging_samples": 3, "interpolation_samples": 4})
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._image is not first, timeout=2)
            assert len(worker._master) == 1
        assert render.call_args.args[0].history_range == 300
        assert render.call_args.args[0].averaging_samples == 3
        assert render.call_args.args[0].interpolation_samples == 4
        assert worker._master[-1][1] == (18, 20, 24, 21)
    finally:
        worker.close()


def test_graph_preferences_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Persist non-default sample controls alongside existing history preferences.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    settings = {
        **GRAPH_DEFAULTS,
        "range_mode": "fixed",
        "range_seconds": 600,
        "history_points": 8192,
        "averaging_samples": 5,
        "interpolation_samples": 8,
    }
    save_settings({"graph_settings": settings})
    assert load_settings()["graph_settings"] == settings
    save_settings({"graph_settings": {"compression": "yes"}, "show_graph": True})
    assert load_settings() == {"display": {}, "hardware": {}, "show_graph": True}


def test_legacy_graph_preferences_use_current_sample_defaults() -> None:
    """
    Old settings without sample controls receive the new chart defaults.
    """
    settings = validate_graph_settings({"history_points": 8192})
    assert settings["averaging_samples"] == 5
    assert settings["interpolation_samples"] == 0


def test_graph_configuration_popup_controls_and_logging_lock() -> None:
    """
    Sample controls publish settings and follow the shared logging lock.
    """
    script = """
from PySide6.QtWidgets import QApplication
from based_thermals.graph_window import GraphWindow
from based_thermals.graph_settings import GRAPH_DEFAULTS
app = QApplication([])
messages = []
window = GraphWindow(messages.append)
window.show()
app.processEvents()
for control in (window.averaging, window.interpolation):
    control.setValue(control.minimum())
    editor = control.lineEdit()
    assert editor.text() == "Off"
    assert editor.contentsRect().width() >= editor.fontMetrics().horizontalAdvance("Off") + 10
window.update_state({"settings": GRAPH_DEFAULTS})
assert window.points.value() == 2048
assert window.averaging.value() == 5 and window.interpolation.value() == 0
assert window.averaging.specialValueText() == "Off"
assert window.minutes.isEnabled() and window.minutes.value() == 10
assert window.range_mode.currentData() == "fixed"
window.range_mode.setCurrentIndex(0)
assert not window.minutes.isEnabled()
window.range_mode.setCurrentIndex(1)
window.minutes.setValue(5)
window.compression.setChecked(False)
window.points.setValue(8192)
editor = window.averaging.lineEdit()
editor.setText("5")
editor.setModified(True)
window.update_state({"settings": messages[-1]["settings"]})
assert editor.text() == "5"
window.averaging.interpretText()
window.averaging.setValue(5)
window.interpolation.setValue(8)
settings = messages[-1]["settings"]
assert settings == {**GRAPH_DEFAULTS, "range_mode": "fixed", "range_seconds": 300, "compression": False, "history_points": 8192, "averaging_samples": 5, "interpolation_samples": 8}
count = len(messages)
window.update_state({"settings": settings, "locked": True})
assert len(messages) == count
assert all(not control.isEnabled() for control in (window.range_mode, window.minutes, window.compression, window.points, window.averaging, window.interpolation))
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
    assert "RuntimeError" not in result.stderr, result.stderr
