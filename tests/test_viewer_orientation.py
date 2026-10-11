"""
Portrait graph geometry, preview orientation commands and native Qt popup behavior.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.qt_process import run_popup

from topdon_duo import desktop
from topdon_duo.graphs import graph_interval_rect


@pytest.mark.parametrize("window", [(600, 1000), (1200, 500), (1, 2)])
def test_portrait_graph_geometry_and_pointer_mapping(window: tuple[int, int]) -> None:
    """
    Place graphs below an aspect-preserving camera, excluding padding from measurement input.
    """
    layout = desktop.GraphWindowLayout.fit((768, 576), window, "portrait")
    width, height = layout.camera_size
    assert width <= window[0] and height <= window[1] // 2
    assert layout.graph_size == (window[0], window[1] - height)
    camera = np.full((576, 768, 3), 100, np.uint8)
    graph = np.full((layout.graph_size[1], layout.graph_size[0], 3), 200, np.uint8)
    canvas = layout.compose(camera, graph)
    assert canvas.shape == (window[1], window[0], 3)
    assert np.all(canvas[:height, :width] == 100)
    assert np.all(canvas[height:] == 200)
    assert np.all(canvas[:height, width:] == 0)
    assert desktop.image_position_at(
        0, height, camera.shape, layout.camera_viewport()
    ) is None
    if min(window) > 2:
        x0, y0, x1, y1 = graph_interval_rect(layout.graph_size[0])
        assert layout.graph_control_at((x0+x1)/2, height+(y0+y1)/2, graph_interval_rect)
        assert not layout.graph_control_at(0, 0, graph_interval_rect)


@pytest.mark.usefixtures("viewer")
def test_orientation_commands_preserve_sensor_geometry_and_restore_layout() -> None:
    """
    Reuse clockwise rotation transformations while persisting the independent layout choice.
    """
    session = desktop.DesktopSession(desktop, [])
    native = Mock(size=(1200, 600), closed=False)
    session.viewer = native
    session.show_graph = True
    session.spots.add((12, 18))
    original = session.spots.saved_state()
    session.set_viewer_orientation({"rotation": 270, "layout": "portrait"})
    assert session.renderer.rotation == 270
    assert session.spots.saved_state(270) == original
    assert session.viewer_layout == "portrait"
    native.resize.assert_called_once_with((600, 1200))
    restored = desktop.DesktopSession(desktop, [])
    assert restored.viewer_layout == "portrait" and restored.renderer.rotation == 270
    native.size = (600, 1200)
    session.toggle_graph()
    native.resize.assert_called_with((600, 600))
    native.size = (600, 600)
    session.toggle_graph()
    native.resize.assert_called_with((600, 1200))


def test_orientation_commands_reject_malformed_choices_and_logging_locks(viewer: Any) -> None:
    """
    Reject stale locked UI requests before either rotation or layout can change.
    """
    session = desktop.DesktopSession(desktop, [])
    for command in (
        {"rotation": True, "layout": "portrait"},
        {"rotation": 45, "layout": "portrait"},
        {"rotation": 90, "layout": None},
        {"rotation": 90, "layout": []},
    ):
        with pytest.raises(ValueError):
            session.set_viewer_orientation(command)
    viewer.graphs.logging = True
    with pytest.raises(ValueError, match="Stop logging"):
        session.set_viewer_orientation({"rotation": 90, "layout": "portrait"})
    assert session.viewer_layout == "landscape" and session.renderer.rotation == 0


@pytest.mark.usefixtures("viewer")
@pytest.mark.parametrize("value", [None, [], 1, "diagonal"])
def test_invalid_saved_layout_keeps_legacy_landscape(value: object) -> None:
    """
    Ignore malformed preferences without breaking existing saved configurations.
    """
    desktop.save_settings({"viewer_layout": value})
    assert desktop.DesktopSession(desktop, []).viewer_layout == "landscape"


def test_portrait_session_draws_on_camera_and_never_samples_graphs(
    viewer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Exercise presentation and drawing events with a real vertical graph composition.
    """
    desktop.save_settings({"viewer_layout": "portrait", "show_graph": True})
    native = Mock(size=(768, 1152), windowed_size=(768, 1152), over_image=True, closed=False)
    native.poll.side_effect = [
        [],
        [
            {"action": "pointer", "kind": "press", "button": "left", "x": 30, "y": 30},
            {"action": "pointer", "kind": "release", "button": "left", "x": 90, "y": 900},
        ],
        [
            {"action": "pointer", "kind": "press", "button": "left", "x": 30, "y": 30},
            {"action": "pointer", "kind": "release", "button": "left", "x": 90, "y": 90},
            {"action": "draw_finish"},
        ],
        [{"action": "quit"}],
    ]
    panel = Mock()
    panel.poll.side_effect = [[{"action": "draw", "kind": "rectangle",
                               "orientation": [0, False, False]}], [], [], []]
    monkeypatch.setattr("topdon_duo.desktop.ViewerPanel", lambda: native)
    monkeypatch.setattr("topdon_duo.desktop.SpotsPanel", lambda: panel)
    viewer.graphs.image.side_effect = lambda size, **_kw: np.full(
        (size[1], size[0], 3), (10, 20, 30), np.uint8
    )
    assert desktop.main([]) == 0
    for call in native.present.call_args_list:
        image = call.args[0]
        assert image.shape == (1152, 768, 3)
        assert np.all(image[576:] == (10, 20, 30))
    items = desktop.load_settings()["spots"]["items"]
    assert len(items) == 1 and items[0]["region"]["points"] == [[10, 10], [30, 30]]


def test_rotation_popup_edits_locks_and_authoritative_reopen(tmp_path: Path) -> None:
    """
    Open without rotating, retain pending edits during frames and dispatch one validated command.
    """
    script = r'''
from PySide6.QtWidgets import QApplication
from topdon_duo.viewer_window import ViewerWindow
app = QApplication([])
messages = []
window = ViewerWindow(messages.append)
state = {"unit": "C", "rotation": 90, "viewer_layout": "landscape"}
window.update_state(state)
window.show()
window.toolbar.buttons["rotate"].click()
dialog = window.orientation_dialog
assert dialog.isVisible() and not messages
assert dialog.rotation.currentData() == 90
dialog.rotation.setCurrentIndex(2)
dialog.layout_choice.setCurrentIndex(1)
window.update_state(state)
assert dialog.rotation.currentData() == 180
assert dialog.layout_choice.currentData() == "portrait"
window.update_state({**state, "settings_locked": True})
assert not dialog.apply.isEnabled() and dialog.close_button.isEnabled()
dialog.apply.click()
assert not messages
window.update_state(state)
dialog.apply.click()
assert messages[-1] == {"action": "orientation", "rotation": 180, "layout": "portrait"}
assert not dialog.isVisible()
window.update_state({**state, "rotation": 180, "viewer_layout": "portrait"})
window.toolbar.buttons["rotate"].click()
assert dialog.rotation.currentData() == 180 and dialog.layout_choice.currentData() == "portrait"
dialog.close_button.click()
window.close()
'''
    result = run_popup(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
