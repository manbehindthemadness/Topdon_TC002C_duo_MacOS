"""
Point selection, captured gestures and display-only label positioning regressions.
"""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from support.qt_process import run_popup

from based_thermals.desktop_app.overlays import place_temperature_label
from based_thermals.desktop_app.spots import MousePicker, SampleSpots, SpotDrag
from based_thermals.measurements.editing import EditTarget
from based_thermals.measurements.overlays import label_layout
from based_thermals.measurements.paths import initial_region
from based_thermals.measurements.schema import default_region, validate_region
from based_thermals.render import RenderedThermalFrame, TemperatureStats


def polygon() -> tuple[SampleSpots, MousePicker, SpotDrag]:
    """
    Build known native vertices and a three-times display for pointer tests.
    """
    spots = SampleSpots(native_size=(64, 48))
    spots.create_region(
        (20, 20),
        {
            **default_region("polygon"),
            "points": [[10, 10], [30, 10], [20, 30]],
        },
        spots.native_size,
    )
    return spots, MousePicker(), SpotDrag()


def pointer(
    picker: MousePicker,
    drag: SpotDrag,
    spots: SampleSpots,
    event: int,
    x: int,
    y: int,
    *,
    locked: bool = False,
) -> None:
    """
    Forward one displayed event through the production picker and drag boundary.
    """
    flags = cv2.EVENT_FLAG_LBUTTON if event == cv2.EVENT_MOUSEMOVE else 0
    picker.callback(event, x, y, flags, None)
    drag.update(picker, spots, (144, 192, 3), 3, locked=locked)


def test_click_selects_vertex_nudges_and_empty_click_deselects() -> None:
    """
    Keep a clicked point selected, consume release and restore normal placement after deselection.
    """
    spots, picker, drag = polygon()
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 31, 31)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, 31, 31)
    assert drag.edit.selected == EditTarget(1, "vertex", 0)
    assert not picker.consume_clicks()
    assert drag.edit.nudge(spots, "right")
    assert spots.region(1)["points"] == [[11, 10], [30, 10], [20, 30]]
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 180, 120)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, 180, 120)
    assert drag.edit.selected is None and not picker.consume_clicks()
    assert not drag.edit.nudge(spots, "up")


def test_drag_previews_without_invalidating_masks_then_commits_once() -> None:
    """
    Hold original readings and cached masks through movement; replace only on release.
    """
    spots, picker, drag = polygon()
    original = deepcopy(spots.region(1))
    mask = spots.measurement_mask(1, (20, 20), (48, 64))
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 31, 31)
    for position in (34, 40, 46):
        pointer(picker, drag, spots, cv2.EVENT_MOUSEMOVE, position, 37)
        assert spots.region(1) == original
        assert spots.measurement_mask(1, (20, 20), (48, 64)) is mask
    assert drag.edit.draft["points"][0] == [15, 12]
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, 46, 37)
    assert spots.region(1)["points"][0] == [15, 12]
    assert spots.measurement_mask(1, (20, 20), (48, 64)) is not mask
    assert spots.numbers == [1] and spots.generation == 0
    assert not picker.consume_clicks()


@pytest.mark.parametrize(
    "cancel", ["lock", "delete", "disable", "external", "cancel", "lost_button"]
)
def test_interrupted_drag_drops_preview_and_swallows_release(cancel: str) -> None:
    """
    Never commit stale geometry or place a new spot after a cancelled capture.
    """
    spots, picker, drag = polygon()
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 31, 31)
    pointer(picker, drag, spots, cv2.EVENT_MOUSEMOVE, 46, 37)
    if cancel == "delete":
        spots.clear(1)
    elif cancel == "disable":
        spots.set_enabled(1, False)
    elif cancel == "external":
        spots.configure_region(
            1, {**spots.region(1), "points": [[9, 9], [30, 10], [20, 30]]}, spots.native_size
        )
    elif cancel == "cancel":
        drag.cancel()
    elif cancel == "lost_button":
        picker.callback(cv2.EVENT_MOUSEMOVE, 46, 37, 0, None)
        drag.update(picker, spots, (144, 192, 3), 3)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, 46, 37, locked=cancel == "lock")
    assert drag.edit.selected is None and not picker.consume_clicks()
    if 1 in spots.numbers:
        assert spots.region(1)["points"][0] == ([9, 9] if cancel == "external" else [10, 10])


def test_bezier_anchor_carries_adjacent_handles_and_controls_move_independently() -> None:
    """
    Preserve tangent offsets and expose only controls adjacent to the selected anchor.
    """
    spots = SampleSpots(native_size=(64, 48))
    center, region = initial_region("bezier", spots.native_size)
    spots.create_region(center, region, spots.native_size)
    region = deepcopy(spots.region(1))
    drag, picker = SpotDrag(), MousePicker()
    anchor = region["points"][0]
    x, y = round(anchor[0] * 3) + 1, round(anchor[1] * 3) + 1
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, x, y)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, x, y)
    assert drag.edit.handles(spots, 1) == [0, 3, 6, 9, 11, 1]
    drag.edit.nudge(spots, "left")
    for index in (0, 1, 11):
        assert spots.region(1)["points"][index][0] == region["points"][index][0] - 1
    assert spots.region(1)["points"][2] == region["points"][2]
    drag.edit.selected = EditTarget(1, "vertex", 1)
    before = deepcopy(spots.region(1)["points"])
    drag.edit.nudge(spots, "up")
    assert spots.region(1)["points"][0] == before[0]
    assert spots.region(1)["points"][1][1] == before[1][1] - 1


def test_scaled_viewport_and_rotated_native_bounds() -> None:
    """
    Edit through viewport scaling and clamp using oriented dimensions rather than camera width.
    """
    spots, picker, drag = polygon()
    spots.rotate_clockwise(48)
    picker.callback(cv2.EVENT_LBUTTONDOWN, 112, 31, 0, None)
    picker.callback(cv2.EVENT_MOUSEMOVE, 130, 40, cv2.EVENT_FLAG_LBUTTON, None)
    picker.callback(cv2.EVENT_LBUTTONUP, 130, 40, 0, None)
    drag.update(picker, spots, (192, 144, 3), 3)
    assert spots.region(1)["points"][0] == [43, 13]
    restored = SampleSpots(native_size=(64, 48))
    restored.restore_saved_state(spots.saved_state(rotation=90))
    assert restored.region(1)["points"][0] == [13, 4]
    picker.callback(cv2.EVENT_LBUTTONDOWN, 65, 20, 0, None)
    picker.callback(cv2.EVENT_LBUTTONUP, 65, 20, 0, None)
    drag.update(picker, spots, (192, 144, 3), 3, (72, 96))
    assert drag.edit.selected == EditTarget(1, "vertex", 0)


def test_label_drag_reuses_auto_layout_and_sampling_geometry() -> None:
    """
    Move only the selected label and retain the cached automatic layout and other origins.
    """
    spots, picker, drag = polygon()
    spots.create_region((50, 35), default_region(), spots.native_size)
    temperatures = np.full((48, 64), 20.0)
    rendered = RenderedThermalFrame(
        np.zeros((144, 192, 3), np.uint8), TemperatureStats(20, 20, 20, 20), temperatures, None
    )
    placer = Mock(wraps=place_temperature_label)
    labels, _ = label_layout(rendered, spots, 3, "C", placer)
    first, second = labels
    x, y = first.origin
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, x + 10, y - 3)
    assert drag.edit.selected == EditTarget(1, "label")
    pointer(picker, drag, spots, cv2.EVENT_MOUSEMOVE, x + 10, y + 18)
    spots.display_cache.label_preview = {1: drag.edit.draft["label_position"]}
    preview, _ = label_layout(rendered, spots, 3, "C", placer)
    assert preview[0].origin != first.origin and preview[1].origin == second.origin
    assert placer.call_count == 2 and spots.region(1)["label_position"] is None
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, x + 10, y + 18)
    spots.display_cache.label_preview = {}
    committed, _ = label_layout(rendered, spots, 3, "C", placer)
    assert committed == preview and placer.call_count == 2
    original_points = deepcopy(spots.region(1)["points"])
    drag.edit.nudge(spots, "down")
    assert spots.region(1)["points"] == original_points
    saved = spots.saved_state()
    restored = SampleSpots(native_size=spots.native_size)
    restored.restore_saved_state(saved)
    assert restored.region(1)["label_position"] == spots.region(1)["label_position"]
    assert restored.readings(temperatures) == spots.readings(temperatures)


@pytest.mark.parametrize(
    "position", [True, [], [1], [0, None], [True, 1], [float("nan"), 1], [-1, 0], [64, 0]]
)
def test_invalid_label_position_rejected(position: object) -> None:
    """
    Reject malformed persisted label coordinates at the document boundary.
    """
    with pytest.raises(ValueError):
        validate_region({**default_region(), "label_position": position}, (64, 48))


def test_arrow_keys_forward_only_from_focused_image(tmp_path: Path) -> None:
    """
    Exercise real Qt arrow events without stealing input from numeric controls.
    """
    result = run_popup(
        """
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.viewer_window import ViewerWindow
app = QApplication([])
commands = []
window = ViewerWindow(commands.append)
window.show()
window.canvas.setFocus()
app.processEvents()
QTest.keyClick(window.canvas, Qt.Key.Key_Left)
assert {"action": "measurement_nudge", "direction": "left"} in commands
commands.clear()
window.toolbar.interval.setFocus()
QTest.keyClick(window.toolbar.interval, Qt.Key.Key_Right)
assert not any(item.get("action") == "measurement_nudge" for item in commands)
window.close()
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_release_outside_image_commits_last_preview_and_never_places() -> None:
    """
    Finish a captured vertex gesture at its latest valid position outside the viewport.
    """
    spots, picker, drag = polygon()
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 31, 31)
    pointer(picker, drag, spots, cv2.EVENT_MOUSEMOVE, 46, 37)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, -1, -1)
    assert spots.region(1)["points"][0] == [15, 12]
    assert not drag.edit.captured and not picker.consume_clicks()


def test_queued_nudges_follow_click_selection_and_locks_reject_them() -> None:
    """
    Apply arrows after queued pointer selection, and discard them when logging locks editing.
    """
    spots, picker, drag = polygon()
    picker.callback(cv2.EVENT_LBUTTONDOWN, 31, 31, 0, None)
    picker.callback(cv2.EVENT_LBUTTONUP, 31, 31, 0, None)
    drag.edit.pending_nudges.extend(("right", "down"))
    drag.update(picker, spots, (144, 192, 3), 3)
    assert spots.region(1)["points"][0] == [11, 11]
    drag.edit.pending_nudges.append("left")
    drag.update(picker, spots, (144, 192, 3), 3, locked=True)
    assert spots.region(1)["points"][0] == [11, 11]
    assert drag.edit.selected is None and not drag.edit.pending_nudges


def test_motion_coalesces_without_losing_press_release_or_button_loss() -> None:
    """
    Bound pointer work to the latest motion between button/modifier transitions.
    """
    picker = MousePicker()
    picker.callback(cv2.EVENT_LBUTTONDOWN, 31, 31, 0, None)
    for x in range(32, 100):
        picker.callback(cv2.EVENT_MOUSEMOVE, x, 40, cv2.EVENT_FLAG_LBUTTON, None)
    picker.callback(cv2.EVENT_MOUSEMOVE, 100, 40, 0, None)
    picker.callback(cv2.EVENT_LBUTTONUP, 100, 40, 0, None)
    assert picker.consume_drag_events() == [
        (cv2.EVENT_LBUTTONDOWN, 31, 31, 0),
        (cv2.EVENT_MOUSEMOVE, 99, 40, cv2.EVENT_FLAG_LBUTTON),
        (cv2.EVENT_MOUSEMOVE, 100, 40, 0),
        (cv2.EVENT_LBUTTONUP, 100, 40, 0),
    ]


def test_label_position_rotates_mirrors_and_clamps_on_whole_shape_movement() -> None:
    """
    Persist manual labels in sensor coordinates alongside geometry without invalid offsets.
    """
    spots, _, _ = polygon()
    spots.configure_region(1, {**spots.region(1), "label_position": [63, 47]}, (64, 48))
    saved = spots.saved_state()
    rotated = SampleSpots(native_size=(64, 48))
    rotated.restore_saved_state(saved, rotation=90, horizontal=True)
    assert rotated.region(1)["label_position"] == [47, 63]
    assert rotated.saved_state(rotation=90, horizontal=True) == saved
    spots.move(1, (23, 25))
    assert spots.region(1)["label_position"] == [63, 47]
    assert validate_region(spots.region(1), (64, 48)) == spots.region(1)


def test_overlap_handle_hit_prefers_highest_layer_and_hidden_points_stay_inactive() -> None:
    """
    Choose the frontmost coincident vertex, skipping disabled geometry and hidden curve controls.
    """
    spots, picker, drag = polygon()
    spots.create_region((20, 20), {**spots.region(1), "z_index": 10}, spots.native_size)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 31, 31)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, 31, 31)
    assert drag.edit.selected == EditTarget(2, "vertex", 0)
    spots.set_enabled(2, False)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 31, 31)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, 31, 31)
    assert drag.edit.selected == EditTarget(1, "vertex", 0)


def test_clear_all_does_not_select_reused_identity() -> None:
    """
    Reject an old point selection after clear-all restarts measurement numbering.
    """
    spots, picker, drag = polygon()
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONDOWN, 31, 31)
    pointer(picker, drag, spots, cv2.EVENT_LBUTTONUP, 31, 31)
    spots.clear()
    spots.create_region(
        (20, 20),
        {**default_region("polygon"), "points": [[10, 10], [30, 10], [20, 30]]},
        spots.native_size,
    )
    assert not drag.edit.nudge(spots, "right")
    assert drag.edit.selected is None
