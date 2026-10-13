"""
Live-preview gesture geometry, coordinate mapping and parent lock boundaries.
"""

from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401

from based_thermals import desktop
from based_thermals.measurements.drawing import MeasurementDrawing


@pytest.mark.parametrize("kind", ["spot", "rectangle", "oval"])
def test_single_drag_builds_native_geometry_and_outside_release_discards(kind: str) -> None:
    """
    Commit radius/bounds from the release position, never from a displayed image kernel.
    """
    draft = MeasurementDrawing()
    draft.begin(kind, False, (50, 40), [0, False, False])
    draft.pointer("press", "left", (5, 7))
    draft.pointer("move", "left", (8, 11))
    draft.pointer("release", "left", (8, 11))
    assert draft.state()["finish"]
    draft.pointer("move", "left", (30, 30))
    center, region = draft.geometry()
    if kind == "spot":
        assert center == (5, 7) and region["radius"] == 5
    else:
        assert center == (6, 9) and region["points"] == [[5, 7], [8, 11]]
    assert not region["exclude"]
    draft.pointer("press", "left", (2, 2))
    draft.pointer("release", "left", None)
    with pytest.raises(ValueError):
        draft.geometry()
    draft.cancel()
    assert not draft.state()["active"]


def test_polygon_rubber_band_finish_undo_and_vertex_limit() -> None:
    """
    Keep hover vertices transient, require three committed vertices and bound topology.
    """
    draft = MeasurementDrawing()
    draft.begin("polygon", True, (50, 40), [0, False, False])
    for point in ((2, 2), (8, 2)):
        draft.pointer("press", "left", point)
        draft.pointer("release", "left", point)
    draft.pointer("move", "left", (2, 8))
    assert draft.geometry(preview=True)[1]["points"] == [[2, 2], [8, 2], [2, 8]]
    assert not draft.state()["finish"]
    with pytest.raises(ValueError):
        draft.geometry()
    draft.pointer("press", "left", (2, 8))
    draft.pointer("release", "left", (2, 8))
    assert draft.state()["finish"]
    center, region = draft.geometry()
    assert center == (4, 4) and region["exclude"]
    draft.undo()
    assert not draft.state()["finish"]
    for _ in range(190):
        draft.pointer("press", "left", (3, 3))
        draft.pointer("release", "left", (3, 3))
    draft.pointer("press", "left", (3, 3))
    with pytest.raises(ValueError, match="Maximum"):
        draft.pointer("release", "left", (3, 3))
    assert len(draft.geometry()[1]["points"]) == 192


def test_bezier_anchor_drags_create_closed_control_triples_and_clip_handles() -> None:
    """
    Derive outgoing/incoming handles from independent tangent drags and close the curve.
    """
    draft = MeasurementDrawing()
    draft.begin("bezier", False, (50, 40), [0, False, False])
    draft.pointer("press", "left", (10, 10))
    draft.pointer("release", "left", (15, 10))
    draft.pointer("press", "left", (30, 20))
    draft.pointer("release", "left", (25, 20))
    assert draft.state()["finish"]
    assert draft.geometry()[1]["points"] == [
        [10, 10],
        [15, 10],
        [35, 20],
        [30, 20],
        [25, 20],
        [5, 10],
    ]
    draft.undo()
    assert not draft.state()["finish"]
    draft.pointer("press", "left", (49, 39))
    draft.pointer("release", "left", (0, 0))
    assert draft.geometry()[1]["points"][2] == [49, 39]
    assert draft.geometry()[1]["points"][4] == [0, 0]


@pytest.mark.parametrize("kind,exclude", [(None, False), ("other", False), ("spot", 1)])
def test_invalid_tool_does_not_replace_existing_draft(kind: object, exclude: object) -> None:
    """
    Validate incoming tool commands before modifying a pending drawing.
    """
    draft = MeasurementDrawing()
    draft.begin("polygon", False, (50, 40), [0, False, False])
    draft.pointer("press", "left", (2, 2))
    draft.pointer("release", "left", (2, 2))
    with pytest.raises((TypeError, ValueError)):
        draft.begin(kind, exclude, (50, 40), [0, False, False])
    assert draft.kind == "polygon" and draft.anchors == [(2, 2)]


def start_drawing(session: Any, kind: str = "rectangle", *, exclude: bool = False) -> None:
    """
    Arm a tool through the same validated command used by the popup.
    """
    session.begin_measurement_drawing(
        {"kind": kind, "exclude": exclude, "orientation": session.spots_state()["orientation"]}
    )


def pointer(session: Any, kind: str, x: int, y: int, button: str = "left", flags: int = 0) -> None:
    """
    Deliver actual image-local JSON through the viewer's pointer validation.
    """
    session.viewer_pointer({"kind": kind, "button": button, "x": x, "y": y, "flags": flags})


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.usefixtures("viewer")
def test_parent_drawing_maps_resized_camera_pane_and_suppresses_placement(
    monkeypatch: pytest.MonkeyPatch,
    graph: bool,
) -> None:
    """
    Create the same native region with graphs visible at a different bitmap size.
    """
    panel = Mock()
    monkeypatch.setattr("based_thermals.desktop.SpotsPanel", lambda: panel)
    session = desktop.DesktopSession(desktop, [])
    session.viewer = Mock()
    session.spots.placing = True
    session.show_graph = graph
    session.graph_layout = desktop.GraphWindowLayout((400, 200), (200, 150)) if graph else None
    start_drawing(session)
    if graph:
        start, end = (25, 25), (75, 100)
        expected = [[32, 32], [96, 128]]
    else:
        start, end = (96, 96), (288, 384)
        expected = [[32, 32], [96, 128]]
    pointer(session, "press", *start)
    pointer(session, "move", *end, flags=1)
    assert session.spots.numbers == []
    source = np.full((576, 768, 3), 50, np.uint8)
    original = source.copy()
    overlay = session.draw_measurement_draft(source)
    assert np.any(overlay != source)
    assert np.array_equal(source, original)
    pointer(session, "release", *end)
    assert session.spots.numbers == []
    assert session.measurement_drawing.state()["finish"]
    pointer(session, "move", 0, 0)
    session.measurement_drawing_action("draw_finish")
    assert session.spots.region(1)["points"] == expected
    assert session.spots.pixels == [(64, 80)]
    assert session.spots.placing
    assert not session.picker.consume_clicks() and not session.picker.consume_drag_events()
    assert session.measurement_drawing.kind is None
    panel.open.assert_called_once()
    assert desktop.load_settings()["spots"]["items"][0]["region"]["points"] == expected


def test_graph_padding_invalid_pointers_and_lock_changes_cannot_create_measurements(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reject graph/padding releases and abandon stale drafts on lock/orientation changes.
    """
    session = desktop.DesktopSession(desktop, [])
    session.viewer = Mock()
    session.show_graph = True
    session.graph_layout = desktop.GraphWindowLayout((400, 200), (200, 150))
    start_drawing(session)
    pointer(session, "press", 25, 25)
    pointer(session, "release", 300, 100)
    assert session.spots.numbers == []
    pointer(session, "press", 25, 25)
    pointer(session, "release", 100, 170)
    assert session.spots.numbers == []
    for bad in (None, True, "10", float("nan")):
        session.viewer_pointer({"kind": "press", "button": "left", "x": bad, "y": 10})
    assert session.measurement_drawing.pressed is None
    pointer(session, "press", 25, 25)
    viewer.graphs.logging = True
    pointer(session, "release", 50, 50)
    assert session.spots.numbers == [] and session.measurement_drawing.kind is None
    with pytest.raises(ValueError, match="Stop logging"):
        start_drawing(session)
    viewer.graphs.logging = False
    start_drawing(session)
    session.renderer.rotation = 90
    pointer(session, "release", 50, 50)
    assert session.measurement_drawing.kind is None
    assert session.spots.numbers == []


@pytest.mark.usefixtures("viewer")
def test_polygon_finish_and_cancel_use_viewer_commands_without_context_menu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Commit an exclusion on right-click and keep incomplete/cancelled drafts unsaved.
    """
    panel = Mock()
    monkeypatch.setattr("based_thermals.desktop.SpotsPanel", lambda: panel)
    session = desktop.DesktopSession(desktop, [])
    native_viewer = Mock(closed=False)
    session.viewer = native_viewer
    start_drawing(session, "polygon", exclude=True)
    for point in ((30, 30), (90, 30), (30, 90)):
        pointer(session, "press", *point)
        pointer(session, "release", *point)
    pointer(session, "release", 30, 90, "right")
    assert session.spots.region(1)["exclude"]
    assert session.spots.graph_readings(np.zeros((192, 256))) == ()
    assert not session.picker.consume_context_clicks()
    before = deepcopy(session.spots.saved_state())
    start_drawing(session, "polygon")
    native_viewer.poll.return_value = [{"action": "draw_finish"}]
    assert session.poll_viewer_commands()
    assert session.measurement_drawing.kind == "polygon"
    native_viewer.poll.return_value = [{"action": "draw_cancel"}]
    assert session.poll_viewer_commands()
    assert session.measurement_drawing.kind is None
    assert session.spots.saved_state() == before


@pytest.mark.usefixtures("viewer")
def test_cancelling_mid_drag_consumes_release_and_finishing_early_cannot_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Prevent the cancelled drawing's mouse-up from becoming a point-placement click.
    """
    session = desktop.DesktopSession(desktop, [])
    native_viewer = Mock(closed=False)
    session.viewer = native_viewer
    session.spots.placing = True
    start_drawing(session)
    pointer(session, "press", 30, 30)
    pointer(session, "move", 90, 90, flags=1)
    native_viewer.poll.return_value = [{"action": "draw_finish"}]
    assert session.poll_viewer_commands()
    assert session.spots.numbers == [] and session.measurement_drawing.kind == "rectangle"
    native_viewer.poll.return_value = [{"action": "draw_cancel"}]
    assert session.poll_viewer_commands()
    pointer(session, "release", 90, 90)
    assert session.picker.consume_clicks() == []
    pointer(session, "press", 30, 30)
    pointer(session, "release", 30, 30)
    assert session.picker.consume_clicks() == [(30, 30)]


@pytest.mark.usefixtures("viewer")
def test_macos_control_click_finishes_polygon_without_adding_a_vertex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Retain the existing macOS Control-click alternative to a secondary click.
    """
    session = desktop.DesktopSession(desktop, [])
    session.viewer = Mock()
    session.spots_panel = Mock()
    start_drawing(session, "polygon")
    for point in ((30, 30), (90, 30), (30, 90)):
        pointer(session, "press", *point)
        pointer(session, "release", *point)
    monkeypatch.setattr(desktop.sys, "platform", "darwin")
    pointer(session, "press", 60, 60, flags=8)
    pointer(session, "release", 60, 60, flags=8)
    assert session.spots.region(1)["points"] == [[10, 10], [30, 10], [10, 30]]


def test_full_session_popup_tool_draws_on_video_and_persists_completed_shape(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Exercise the real tick order from popup command to video draft and saved region.
    """
    native_viewer = Mock(size=None, windowed_size=(768, 576), over_image=True, closed=False)
    native_viewer.poll.side_effect = [
        [],
        [
            {"action": "pointer", "kind": "press", "button": "left", "x": 30, "y": 30},
            {"action": "pointer", "kind": "move", "button": "left", "x": 90, "y": 90, "flags": 1},
        ],
        [{"action": "pointer", "kind": "release", "button": "left", "x": 90, "y": 90}],
        [{"action": "draw_finish"}],
        [{"action": "quit"}],
    ]
    panel = Mock()
    panel.poll.side_effect = [
        [
            {
                "action": "draw",
                "kind": "rectangle",
                "exclude": False,
                "orientation": [0, False, False],
            }
        ],
        [],
        [],
        [],
        [],
    ]
    monkeypatch.setattr("based_thermals.desktop.ViewerPanel", lambda: native_viewer)
    monkeypatch.setattr("based_thermals.desktop.SpotsPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", Mock(side_effect=AssertionError("No HighGUI")))
    assert desktop.main([]) == 0
    saved = desktop.load_settings()["spots"]
    assert len(saved["items"]) == 1
    assert saved["items"][0]["region"]["points"] == [[10, 10], [30, 30]]
    states = [
        call.args[1]["measurement_drawing"]["active"]
        for call in native_viewer.present.call_args_list
    ]
    assert states == [True, True, True, False]
    assert not np.array_equal(
        native_viewer.present.call_args_list[0].args[0],
        native_viewer.present.call_args_list[1].args[0],
    )
    panel.open.assert_called_once()
    assert not viewer.draw_toolbar.called
