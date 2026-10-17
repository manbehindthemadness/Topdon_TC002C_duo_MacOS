"""
Opt-in dynamic extrema/center overlays leave native measurements and saved spots unchanged.
"""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from support.frames import make_frame
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.camera import HEADER_U16, SENSOR_PIXELS
from based_thermals.desktop_app.dynamic_markers import draw_dynamic_markers
from based_thermals.desktop_app.overlays import draw_contrasting_overlay
from based_thermals.render import RenderedThermalFrame, ThermalRenderer
from based_thermals.settings_preferences import load_settings


def marked_frame(
    scale: int = 2, rotation: int = 0, horizontal: bool = False, vertical: bool = False,
) -> RenderedThermalFrame:
    """
    Render independent, asymmetric hot/cold sensor targets using the normal orientation path.
    """
    words = np.frombuffer(make_frame(), dtype="<u2").copy()
    sensor = words[HEADER_U16:HEADER_U16 + SENSOR_PIXELS].reshape(192, 256)
    sensor[10, 12] = 15_000
    sensor[129, 145] = 25_000
    renderer = ThermalRenderer(scale=scale, rotation=rotation, smoothing=1)
    renderer.mirror_horizontal, renderer.mirror_vertical = horizontal, vertical
    try:
        rendered = renderer.render_detailed(words.tobytes())
    finally:
        renderer.pipeline_processor.close()
    return rendered


def full_canvas_markers(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    """
    Preserve the original full-canvas rendering as an independent pixel reference.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    for point in points:
        cv2.circle(mask, point, 5, 255, 1, cv2.LINE_AA)
    halo = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    result = draw_contrasting_overlay(image, halo, np.zeros_like(mask))
    for point, color in zip(points, ((255, 128, 0), (0, 0, 255), (0, 255, 0)), strict=True):
        cv2.circle(result, point, 5, color, 1, cv2.LINE_AA)
    return result


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_fixed_diameter_markers_follow_measurements_and_preserve_inputs(
    scale: int,
) -> None:
    """
    Mark known sensor targets with fixed display-size circles, never palette extrema.
    """
    rendered = marked_frame(scale)
    image = np.full_like(rendered.image, 120)
    image[0, 0] = 0
    image[-1, -1] = 255
    before_image = image.copy()
    before_temperatures = rendered.temperatures_celsius.copy()
    assert rendered.raw_counts is not None
    before_counts = rendered.raw_counts.copy()
    points = [
        (12 * scale + scale // 2, 10 * scale + scale // 2),
        (145 * scale + scale // 2, 129 * scale + scale // 2),
        (128 * scale + scale // 2, 96 * scale + scale // 2),
    ]
    result = draw_dynamic_markers(image, rendered)
    np.testing.assert_array_equal(result, full_canvas_markers(image, points))
    assert np.any(result != image)
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(rendered.temperatures_celsius, before_temperatures)
    np.testing.assert_array_equal(rendered.raw_counts, before_counts)
    changed_y, changed_x = np.where(np.any(result != image, axis=2))
    for x, y in zip(changed_x, changed_y, strict=True):
        assert any((x - px) ** 2 + (y - py) ** 2 <= 9 ** 2 for px, py in points)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)])
def test_markers_follow_camera_rotation_and_mirrors(
    rotation: int, horizontal: bool, vertical: bool,
) -> None:
    """
    Sensor targets land at the corresponding oriented camera pixels in every viewing direction.
    """
    rendered = marked_frame(2, rotation, horizontal, vertical)
    result = draw_dynamic_markers(rendered.image, rendered)
    expected = []
    for x, y in ((12, 10), (145, 129)):
        width, height = 256, 192
        for _ in range(rotation // 90):
            x, y = height - 1 - y, x
            width, height = height, width
        if horizontal:
            x = width - 1 - x
        if vertical:
            y = height - 1 - y
        expected.append((x * 2 + 1, y * 2 + 1))
    height, width = rendered.temperatures_celsius.shape
    expected.append((width // 2 * 2 + 1, height // 2 * 2 + 1))
    np.testing.assert_array_equal(result, full_canvas_markers(rendered.image, expected))


def test_markers_refresh_when_the_same_temperature_array_is_reused() -> None:
    """
    Moving extrema must not retain positions from an earlier frame or a cached array identity.
    """
    rendered = marked_frame(1)
    draw_dynamic_markers(rendered.image, rendered)
    rendered.temperatures_celsius.fill(20)
    rendered.temperatures_celsius[40, 50] = 0
    rendered.temperatures_celsius[80, 100] = 100
    result = draw_dynamic_markers(rendered.image, rendered)
    np.testing.assert_array_equal(
        result, full_canvas_markers(rendered.image, [(50, 40), (100, 80), (128, 96)]),
    )


@pytest.mark.parametrize("size", [(1024, 768), (768, 1024), (23, 19)])
@pytest.mark.parametrize("layout", ["spread", "overlap", "chain", "edges", "flat"])
def test_local_regions_match_original_compositing(size: tuple[int, int], layout: str) -> None:
    """
    Match full-canvas pixels at edges, shared halos, chained regions and coincident extrema.
    """
    width, height = size
    temperatures = np.full((height, width), 20.0)
    positions = {
        "spread": ((width // 4, height // 4), (width * 3 // 4, height * 3 // 4)),
        "overlap": ((width // 2 - 2, height // 2), (width // 2 + 2, height // 2)),
        "chain": ((width // 2 - 18, height // 2), (width // 2 - 36, height // 2)),
        "edges": ((0, 0), (width - 1, height - 1)),
        "flat": ((0, 0), (0, 0)),
    }
    minimum, maximum = positions[layout]
    minimum = (max(0, minimum[0]), minimum[1])
    maximum = (max(0, maximum[0]), maximum[1])
    if layout != "flat":
        temperatures[minimum[1], minimum[0]] = 0
        temperatures[maximum[1], maximum[0]] = 100
    if minimum == maximum:
        minimum = (0, 0)
    image = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    rendered = replace(marked_frame(1), image=image, temperatures_celsius=temperatures)
    points = [minimum, maximum, (width // 2, height // 2)]
    np.testing.assert_array_equal(
        draw_dynamic_markers(image, rendered), full_canvas_markers(image, points),
    )


def test_separated_markers_composite_only_small_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Bound contrast work to three 21-pixel patches even when markers span a large display.
    """
    rendered = marked_frame(4)
    spy = Mock(wraps=draw_contrasting_overlay)
    monkeypatch.setattr("based_thermals.desktop_app.dynamic_markers.draw_contrasting_overlay", spy)
    draw_dynamic_markers(rendered.image, rendered)
    assert spy.call_count == 3
    assert [call.args[0].shape for call in spy.call_args_list] == [(21, 21, 3)] * 3


@pytest.mark.parametrize("background", [0, 127, 128, 255])
def test_colored_lines_and_inverted_halos_stay_visible_on_neutral_backgrounds(background: int) -> None:
    """
    Keep identifiable colored lines and neutral contrasting halos even on middle gray.
    """
    rendered = marked_frame(1)
    image = np.full_like(rendered.image, background)
    result = draw_dynamic_markers(image, rendered)
    for (x, y), channel in zip(((12, 10), (145, 129), (128, 96)), (0, 2, 1), strict=True):
        patch = result[y - 8:y + 9, x - 8:x + 9]
        neutral = (patch[..., 0] == patch[..., 1]) & (patch[..., 1] == patch[..., 2])
        neutral_difference = np.abs(patch[..., 0].astype(np.int16) - background)
        assert int(neutral_difference[neutral].max()) >= 64
        other_channels = [index for index in (0, 1, 2) if index != channel]
        assert np.any(patch[..., channel].astype(np.int16)
                      > patch[..., other_channels].max(axis=2).astype(np.int16) + 20)


@pytest.mark.parametrize("invalid", ["invalid", "mapping", "radiometry", "nan", "empty", "shape"])
def test_unavailable_measurements_do_not_draw_markers(invalid: str) -> None:
    """
    Suppress unsupported, invalid, unregistered and unavailable readings without pixel changes.
    """
    rendered = marked_frame()
    if invalid == "invalid":
        rendered = replace(rendered, measurements_valid=False)
    elif invalid == "mapping":
        rendered = replace(rendered, measurement_mapping_valid=False)
    elif invalid == "radiometry":
        rendered = replace(rendered, radiometry_available=False)
    elif invalid == "nan":
        rendered.temperatures_celsius[0, 0] = np.nan
    else:
        rendered = replace(rendered, temperatures_celsius=np.empty((0, 0) if invalid == "empty" else (1,)))
    assert draw_dynamic_markers(rendered.image, rendered) is rendered.image


@pytest.fixture
def session(viewer: Any) -> Iterator[Any]:
    """
    Provide a real command controller with fake camera and window boundaries.
    """
    current = desktop.DesktopSession(desktop, [])
    current.spots_panel = Mock()
    assert current.hardware is viewer.hardware
    try:
        yield current
    finally:
        current.renderer.pipeline_processor.close()
        current.support_export.close()
        current.support_logging.close()


def test_checkbox_command_defaults_off_persists_and_does_not_create_spots(session: Any) -> None:
    """
    Opt in through the measurement message boundary and restore that explicit preference.
    """
    assert not session.dynamic_markers
    assert not session.spots_state()["dynamic_markers"]
    before = deepcopy(session.spots.saved_state())
    session.spots_panel.poll.return_value = [{"action": "dynamic_markers", "enabled": True}]
    session.poll_spot_and_graph_commands()
    assert session.dynamic_markers and session.spots_state()["dynamic_markers"]
    assert load_settings()["dynamic_markers"] is True
    assert session.spots.saved_state() == before
    assert not session.settings_snapshot().get("spot_hardware")
    session.hardware.set.assert_not_called()
    restarted = desktop.DesktopSession(desktop, [])
    try:
        assert restarted.dynamic_markers
    finally:
        restarted.renderer.pipeline_processor.close()
        restarted.support_export.close()
        restarted.support_logging.close()


@pytest.mark.parametrize("enabled", [None, 0, 1, "true"])
def test_invalid_checkbox_command_cannot_enable_markers(session: Any, enabled: object) -> None:
    """
    Reject non-boolean message values before changing display or preferences.
    """
    session.spots_panel.poll.return_value = [{"action": "dynamic_markers", "enabled": enabled}]
    session.poll_spot_and_graph_commands()
    assert not session.dynamic_markers


def test_marker_toggle_obeys_logging_lock(session: Any) -> None:
    """
    Match the measurement panel's existing lock behavior in the parent process.
    """
    session.graphs.logging = True
    session.spots_panel.poll.return_value = [{"action": "dynamic_markers", "enabled": True}]
    session.poll_spot_and_graph_commands()
    assert not session.dynamic_markers


def test_viewer_draws_dynamic_markers_only_while_enabled(
    viewer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Exercise real frame presentation so default-off and toggle-off both omit the overlay.
    """
    panel = Mock()
    panel.poll.side_effect = [[], [{"action": "dynamic_markers", "enabled": True}],
                              [{"action": "dynamic_markers", "enabled": False}]]
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(desktop, "SpotsPanel", lambda: panel)
    spy = Mock(wraps=draw_dynamic_markers)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(desktop, "draw_dynamic_markers", spy)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events([-1, -1, "q"]))
    assert desktop.main([]) == 0
    spy.assert_called_once()
    assert load_settings()["dynamic_markers"] is False


@pytest.mark.parametrize("suppression", ["emissivity", "reflected", "distance", "invalid", "mapping"])
def test_calibration_and_invalid_frames_hide_dynamic_overlays(
    session: Any, monkeypatch: pytest.MonkeyPatch, suppression: str,
) -> None:
    """
    Calibration targets and invalid spatial readings retain precedence over the optional overlay.
    """
    session.dynamic_markers = True
    session.rendered = marked_frame()
    session.layout = desktop.toolbar_layout(session.rendered.image.shape[1])
    session.display = session.rendered.image.copy()
    if suppression == "emissivity":
        session.emissivity_calibration.active = True
    elif suppression == "reflected":
        session.reflected_calibration.active = True
    elif suppression == "distance":
        session.distance_calibration.selecting = "reference"
    elif suppression == "invalid":
        session.rendered = replace(session.rendered, measurements_valid=False,
                                   measurement_status="Camera calibrating")
    else:
        session.rendered = replace(session.rendered, measurement_mapping_valid=False)
    spy = Mock(wraps=draw_dynamic_markers)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(desktop, "draw_dynamic_markers", spy)
    session.spots.display_cache.begin_frame()
    session.draw_status()
    spy.assert_not_called()


def test_dynamic_markers_checkbox_position_refresh_and_locks(tmp_path: Path) -> None:
    """
    Verify the real checkbox sits directly right of placement and never emits on state refresh.
    """
    result = run_popup('''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from based_thermals.measurements_window import MeasurementsWindow
app = QApplication([])
messages = []
window = MeasurementsWindow(messages.append)
window.show()
app.processEvents()
control = window.dynamic_markers
assert control.text() == "Dynamic markers"
assert not control.isChecked()
assert control.geometry().left() > window.placing.geometry().right()
assert control.geometry().top() == window.placing.geometry().top()
QTest.mouseClick(control, Qt.MouseButton.LeftButton)
assert messages == [{"action": "dynamic_markers", "enabled": True}]
state = {"dynamic_markers": True, "mapping_available": True}
for _ in range(3):
    window.update_state(state)
assert len(messages) == 1 and control.isChecked()
for lock in ({"locked": True}, {"mapping_available": False}):
    window.update_state({**state, **lock})
    assert not control.isEnabled() and control.isChecked()
    control.click()
    assert len(messages) == 1
window.update_state(state)
control.click()
assert messages[-1] == {"action": "dynamic_markers", "enabled": False}
window.close()
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


__all__ = ["viewer_fixture"]
