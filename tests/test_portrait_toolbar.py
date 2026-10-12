"""
Portrait chart controls move below the canvas while preserving actions and image geometry.
"""

from pathlib import Path

from support.qt_process import run_popup


def test_chart_controls_relocate_without_losing_state_or_duplicating_events(tmp_path: Path) -> None:
    """
    Exercise live orientation changes, graph locks and visibility using the same Qt widgets.
    """
    result = run_popup(
        '''
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from topdon_duo.viewer_window import ViewerWindow

app = QApplication([])
messages = []
window = ViewerWindow(messages.append)
window.show()
app.processEvents()
toolbar = window.toolbar
bottom = toolbar.graph_toolbar
actions = list(toolbar.graph_widgets)
interval = toolbar.interval
state = {"unit": "C", "show_graph": True, "graph_interval": 2.5,
         "history": "History | 5.0 min", "viewer_layout": "landscape"}
window.update_state(state)
app.processEvents()
assert bottom.isHidden()
assert all(action in toolbar.actions() for action in actions)
assert window.toolBarArea(bottom) == Qt.ToolBarArea.BottomToolBarArea

for _ in range(4):
    messages.clear()
    window.update_state({**state, "viewer_layout": "portrait", "resize": [600, 900]})
    app.processEvents()
    assert bottom.isVisible()
    assert all(action in bottom.actions() and action not in toolbar.actions() for action in actions)
    assert toolbar.interval is interval and interval.value() == 2.5
    assert toolbar.history.text() == "History | 5.0 min"
    assert interval.parentWidget() is bottom
    assert toolbar.history.parentWidget() is bottom
    assert toolbar.buttons["capture"].parentWidget() is toolbar
    assert toolbar.buttons["graph"].parentWidget() is toolbar
    assert [window.canvas.width(), window.canvas.height()] == [600, 900]
    assert bottom.geometry().top() >= window.canvas.geometry().bottom()
    assert not any(message.get("action") in ("graph_interval", "graph_reset", "graph_log")
                   for message in messages)
    toolbar.buttons["graph_reset"].click()
    assert sum(message == {"action": "graph_reset"} for message in messages) == 1
    window.update_state({**state, "resize": [900, 600]})
    app.processEvents()
    assert bottom.isHidden()
    assert all(action in toolbar.actions() and action not in bottom.actions() for action in actions)
    assert toolbar.interval is interval
    assert [window.canvas.width(), window.canvas.height()] == [900, 600], window.canvas.size()

window.update_state({**state, "viewer_layout": "portrait", "graph_locked": True,
                     "logging": True, "resize": [600, 900]})
app.processEvents()
assert not interval.isEnabled() and not toolbar.buttons["graph_reset"].isEnabled()
assert toolbar.buttons["graph_log"].text() == "Stop logging"
messages.clear()
window.update_state({**state, "viewer_layout": "portrait", "show_graph": False,
                     "resize": [600, 900]})
app.processEvents()
assert bottom.isHidden()
assert [window.canvas.width(), window.canvas.height()] == [600, 900]
window.update_state({**state, "viewer_layout": "portrait", "resize": [600, 900]})
app.processEvents()
assert bottom.isVisible() and interval.isEnabled()
messages.clear()
interval.setValue(3)
assert messages == [{"action": "graph_interval", "value": 3.0}]
window._send_geometry()
assert messages[-1]["size"] == [600, 900]
assert messages[-1]["windowed_size"] == [600, 900]
before = window.size()
window.toolbar.buttons["fullscreen"].click()
app.processEvents()
assert window.isFullScreen() and bottom.isVisible()
window.update_state({**state, "viewer_layout": "portrait", "resize": [300, 400]})
app.processEvents()
assert window.isFullScreen()
window.toolbar.buttons["fullscreen"].click()
app.processEvents()
assert not window.isFullScreen() and window.size() == before
assert [window.canvas.width(), window.canvas.height()] == [600, 900]
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
