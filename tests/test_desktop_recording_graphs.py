from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from test_camera import make_frame

from topdon_duo import desktop


def test_show_graph_doubles_window_width_and_hides_back_to_original(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import load_settings
    from topdon_duo.window_preferences import save_main_window_size

    save_main_window_size((930, 710))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        if step == 1:
            viewer.click_control("graph")
            return -1
        if step == 2:
            toolbar = desktop.toolbar_layout(768)
            layout = desktop.GraphWindowLayout.fit((768, 576 + toolbar.height), (1860, 710))
            x0, y0, x1, y1 = toolbar.buttons["graph"]
            viewer.set_mouse.call_args.args[1](
                cv2.EVENT_LBUTTONUP,
                round((x0 + x1) / 2 * layout.camera_size[0] / 768),
                round((y0 + y1) / 2 * layout.camera_size[1] / (576 + toolbar.height)),
                0,
                None,
            )
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert [call.args[1:] for call in desktop.cv2.resizeWindow.call_args_list] == [
        (930, 710),
        (1860, 710),
        (930, 710),
    ]
    assert [image.shape[1] for image in viewer.displayed] == [768, 1860, 768]
    graph_layout = desktop.GraphWindowLayout.fit(
        (768, 576 + desktop.toolbar_layout(768).height), (1860, 710)
    )
    assert np.count_nonzero(viewer.displayed[1][34:, graph_layout.camera_size[0] :]) == 0
    assert load_settings()["show_graph"] is False


def test_show_graph_survives_restart_without_doubling_again(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import load_settings
    from topdon_duo.window_preferences import load_main_window_size

    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("g"))
    viewer.camera.frames.return_value = [make_frame()]
    assert desktop.main([]) == 0
    assert load_settings()["show_graph"] is True
    size = load_main_window_size()
    assert size[0] == 1536
    desktop.cv2.resizeWindow.reset_mock()
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    desktop.cv2.resizeWindow.assert_called_once_with(desktop.WINDOW_NAME, *size)
    assert viewer.displayed[-1].shape[1] == 1536
    assert np.count_nonzero(viewer.displayed[-1][34:, 768:]) == 0


def test_graph_area_does_not_sample_thermal_pixels(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    height = 576 + desktop.toolbar_layout(768).height
    monkeypatch.setattr(desktop, "mouse_viewport_size", lambda: (1536, height))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    samples = []
    draw_picker = desktop.draw_picker

    def sample(*args: Any, **kwargs: Any) -> Any:
        """
        Sample.
        """
        image, pixel = draw_picker(*args, **kwargs)
        samples.append(pixel)
        return image, pixel

    monkeypatch.setattr(desktop, "draw_picker", sample)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        callback = viewer.set_mouse.call_args.args[1]
        if step == 1:
            callback(cv2.EVENT_MOUSEMOVE, 1000, 180 + height - 576, 0, None)
        elif step == 2:
            callback(cv2.EVENT_MOUSEMOVE, 240, 180 + height - 576, 0, None)
        else:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert samples == [None, None, (80, 60)]


def test_graph_submission_uses_frame_orientation_and_pauses_when_hidden(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        if step == 1:
            return ord("p")
        if step == 2:
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 750, 570 + desktop.toolbar_layout(768).height, 0, None)
            viewer.click_control("rotate")
            return -1
        if step == 3:
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    data = viewer.graphs.submit.call_args.args[0]
    assert data.spots == (((0, 0), 20000 / 64 - 50),)
    assert data.stats == (20000 / 64 - 50,) * 4
    viewer.graphs.close.assert_called_once()
    viewer.graphs.submit.reset_mock()
    viewer.graphs.pause.reset_mock()
    save_settings({"show_graph": False})
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    viewer.graphs.submit.assert_not_called()
    viewer.graphs.pause.assert_called_once()


def test_graph_logging_button_locks_hiding_until_logging_stops(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.graphs import graph_log_button_rect
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def click_log() -> None:
        """
        Click log.
        """
        x0, y0, x1, y1 = graph_log_button_rect(768)
        callback = viewer.set_mouse.call_args.args[1]
        callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        if step == 1:
            click_log()
            viewer.click_control("graph")  # Locked while choosing the logfile.
        elif step == 4:
            assert viewer.graphs.logging
            viewer.click_control("graph")  # Toolbar route must be locked as well as G.
            return ord("g")
        elif step == 5:
            assert viewer.graphs.logging
            click_log()
        elif step == 6:
            assert not viewer.graphs.logging
            viewer.click_control("graph")
        elif step == 7:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert viewer.dialog.open_calls == [(None, {"suffix": ".csv", "kind": "temperatures"})]
    viewer.graphs.start_logging.assert_called_once_with(viewer.dialog.selected)
    viewer.graphs.stop_logging.assert_called_once()
    assert [image.shape[1] for image in viewer.displayed] == [1536] * 6 + [768]
    assert len(desktop.cv2.resizeWindow.call_args_list) == 2
    assert any(call.kwargs["graph_locked"] for call in viewer.draw_toolbar.call_args_list)


def test_cancel_graph_log_dialog_unlocks_graphs(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    # L starts choosing a file; pressing L again cancels before a path is selected.
    monkeypatch.setattr(desktop.cv2, "waitKey", viewer.key_events(["l", "l", "g", "q"]))
    assert desktop.main([]) == 0
    viewer.graphs.start_logging.assert_not_called()
    assert viewer.displayed[-1].shape[1] == 768


def test_graph_logging_button_maps_resized_viewport() -> None:
    from topdon_duo.graphs import graph_log_button_rect

    x0, y0, x1, y1 = graph_log_button_rect(768)
    for scale in (0.5, 1, 1.5):
        viewport = (768 * scale, 650 * scale)  # Thermal half of combined viewport.
        assert desktop.graph_logging_button_at(
            round((768 + (x0 + x1) / 2) * scale), round((y0 + y1) / 2 * scale), 768, 650, viewport
        )
        assert not desktop.graph_logging_button_at(20, 20, 768, 650, viewport)


def test_logging_blocks_camera_settings_toolbar_and_shortcuts_then_unlocks(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    viewer.graphs.logging = True
    panel = Mock()
    panel.poll.side_effect = [
        [
            {"action": "setting", "name": "mirror_horizontal", "value": True},
            {"action": "hardware", "name": "ambient", "value": 35, "enabled": True},
            {"action": "restore_hardware"},
            {"action": "reset"},
            {"action": "advanced_auto", "value": False},
        ],
        [],
        [],
        [],
        [],
    ]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        if step == 1:
            for action in ("unit", "rotate", "spots"):
                viewer.click_control(action)
            return ord("f")
        if step == 2:
            return ord("o")
        if step == 3:
            return ord("p")
        if step == 4:
            viewer.graphs.logging = False
            return ord("f")  # Works again after logging stops.
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    viewer.hardware.set.assert_not_called()
    viewer.hardware.restore.assert_called_once()  # Normal exit restoration only.
    saved = load_settings()
    assert saved["rotation"] == 0
    assert saved["display"]["mirror_horizontal"] is False
    assert saved["display"]["temperature_unit"] == "F"
    assert saved["advanced_auto"] is True
    assert saved["hardware"] == {}
    assert [call.args[0]["settings_locked"] for call in panel.update.call_args_list] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert [call.kwargs["settings_locked"] for call in viewer.draw_toolbar.call_args_list] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert all(not call.args[0].spots for call in viewer.graphs.submit.call_args_list)


@pytest.mark.parametrize("mode", ["video", "timelapse"])
@pytest.mark.parametrize("include,visible", [(False, True), (True, False), (True, True)])
def test_optional_graph_recording_uses_visible_pane_and_keeps_dimensions(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, mode: Any, include: Any, visible: Any
) -> None:
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    from topdon_duo.graphs import GraphWorker

    cache = GraphWorker()
    cache.close()
    # Use the real cache lookup: the live pane includes the toolbar's height,
    # while saved images include the temperature readout instead.
    cache._image = np.full(
        (576 + desktop.toolbar_layout(768).height, 768, 3), (12, 34, 56), dtype=np.uint8
    )
    viewer.graphs.image.side_effect = cache.image
    step = 0

    def key(_delay: Any) -> Any:
        """
        Key.
        """
        nonlocal step
        step += 1
        viewer.clock[0] += 1 if mode == "timelapse" else 0.04
        if step == 1 and visible:
            return ord("g")
        if step == 2:
            viewer.panel.events.extend(
                [
                    {"action": "graphs", "value": include},
                    {
                        "action": mode,
                        "frames_per_minute": 60,
                        "capture_cursor": False,
                        "capture_graphs": include,
                    },
                ]
            )
        if step == 7:
            viewer.panel.events.append({"action": "graphs", "value": False})
            if visible:
                return ord("g")
        return ord("q") if step == 10 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    frames = viewer.writer.frames
    assert frames
    included = include and visible
    assert all(
        frame.shape == (576 + desktop.READOUT_HEIGHT, 1536 if included else 768, 3)
        for frame in frames
    )
    if included:
        assert np.all(frames[0][:, 768:] == (12, 34, 56))
        assert not frames[-1][:, 768:].any()


@pytest.mark.parametrize("visible", [False, True])
def test_image_graph_capture_preserves_native_radiometric_data(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, visible: Any
) -> None:
    import json

    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    from topdon_duo.graphs import GraphWorker

    cache = GraphWorker()
    cache.close()
    # Use the real cache lookup: the live pane includes the toolbar's height,
    # while saved images include the temperature readout instead.
    cache._image = np.full(
        (576 + desktop.toolbar_layout(768).height, 768, 3), (12, 34, 56), dtype=np.uint8
    )
    viewer.graphs.image.side_effect = cache.image
    viewer.dialog.selected = viewer.dialog.selected.with_suffix(".png")
    step = 0

    def key(_delay: Any) -> Any:
        """
        Key.
        """
        nonlocal step
        step += 1
        if step == 1 and visible:
            return ord("g")
        if step == 2:
            viewer.panel.events.extend([{"action": "graphs", "value": True}, {"action": "image"}])
        return ord("q") if step == 6 else -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    image = cv2.imread(str(viewer.dialog.selected))
    assert image.shape == (576 + desktop.READOUT_HEIGHT, 1536 if visible else 768, 3)
    if visible:
        assert np.all(image[:, 768:] == (12, 34, 56))
    metadata = json.loads(viewer.dialog.selected.with_suffix(".json").read_text())
    assert metadata["graphs_included"] is visible
    with np.load(viewer.dialog.selected.with_suffix(".npz")) as data:
        assert data["raw_counts"].shape == (192, 256)
        assert data["temperatures_celsius"].shape == (192, 256)


def test_graph_interval_field_applies_and_remembers_value(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.graphs import graph_interval_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True, "graph_interval": 2})
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    keys = iter([-1, ord("0"), ord("."), ord("2"), ord("5"), 13, ord("q")])
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_interval_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        return next(keys)

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert [call.args for call in viewer.graphs.set_interval.call_args_list] == [(2,), (0.25,)]
    assert load_settings()["graph_interval"] == 0.25


def test_graph_interval_field_is_locked_during_logging(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.graphs import graph_interval_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    viewer.graphs.logging = True
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_interval_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    viewer.graphs.set_interval.assert_called_once_with(0.5)
    assert load_settings()["graph_interval"] == 0.5


def test_graph_interval_field_maps_resized_viewport() -> None:
    from topdon_duo.graphs import graph_interval_rect

    x0, y0, x1, y1 = graph_interval_rect(768)
    for scale in (0.5, 1, 1.5):
        assert desktop.graph_interval_at(
            round((768 + (x0 + x1) / 2) * scale),
            round((y0 + y1) / 2 * scale),
            768,
            650,
            (768 * scale, 650 * scale),
        )
        assert not desktop.graph_interval_at(20, 20, 768, 650, (768 * scale, 650 * scale))


def test_popup_covering_window_center_preserves_graph_layout(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import save_settings
    from topdon_duo.window_preferences import load_main_window_size

    save_settings({"show_graph": True})
    window_size = (2400, 1200)
    covered = False
    step = 0
    monkeypatch.setattr(
        desktop, "window_resize_size", lambda _name: None if covered else window_size
    )
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal covered, step, window_size
        step += 1
        if step == 1:
            covered = True
        elif step == 3:
            covered = False
            window_size = (2600, 1300)
        elif step == 4:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert [image.shape[:2] for image in viewer.displayed] == [
        (1200, 2400),
        (1200, 2400),
        (1200, 2400),
        (1300, 2600),
    ]
    assert load_main_window_size() == (2600, 1300)


@pytest.mark.parametrize("event_scale", [None, 0.5])
def test_resized_graph_window_keeps_spots_and_controls_aligned(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, event_scale: Any
) -> None:
    from topdon_duo.graphs import graph_interval_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    native_height = 576 + desktop.toolbar_layout(768).height
    window = [1536, native_height]
    spots = desktop.SampleSpots(pixels=[(80, 60)])
    monkeypatch.setattr(desktop, "SampleSpots", lambda: spots)
    monkeypatch.setattr(desktop, "window_resize_size", lambda _name: tuple(window))
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(
        desktop,
        "mouse_viewport_size",
        lambda: tuple(value * event_scale for value in window) if event_scale else None,
    )
    step = 0

    def wait_key(_delay: Any) -> Any:
        """
        Wait key.
        """
        nonlocal step
        step += 1
        callback = viewer.set_mouse.call_args.args[1]
        if step == 1:
            window[:] = [2400, 1200]
            layout = desktop.GraphWindowLayout.fit((768, native_height), tuple(window))
            factor = event_scale or 1
            scale_x = layout.camera_size[0] / 768 * factor
            scale_y = layout.camera_size[1] / native_height * factor
            toolbar = desktop.toolbar_layout(768).height
            callback(
                cv2.EVENT_LBUTTONDOWN,
                round(240 * scale_x),
                round((180 + toolbar) * scale_y),
                0,
                None,
            )
            callback(
                cv2.EVENT_MOUSEMOVE,
                round(270 * scale_x),
                round((210 + toolbar) * scale_y),
                cv2.EVENT_FLAG_LBUTTON,
                None,
            )
            callback(
                cv2.EVENT_LBUTTONUP, round(270 * scale_x), round((210 + toolbar) * scale_y), 0, None
            )
        elif step == 2:
            assert spots.pixels == [(90, 70)]
            layout = desktop.GraphWindowLayout.fit((768, native_height), tuple(window))
            x0, y0, x1, y1 = graph_interval_rect(layout.graph_size[0])
            factor = event_scale or 1
            callback(
                cv2.EVENT_LBUTTONUP,
                round((layout.camera_size[0] + (x0 + x1) / 2) * factor),
                round((y0 + y1) / 2 * factor),
                0,
                None,
            )
        elif step == 3:
            return ord("2")
        elif step == 4:
            return 13
        elif step == 5:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert viewer.displayed[0].shape[:2] == (native_height, 1536)
    assert all(image.shape[:2] == (1200, 2400) for image in viewer.displayed[1:])
    assert viewer.graphs.submit.call_args.args[0].size == (1200, 1200)
    assert spots.pixels == [(90, 70)]
    assert load_settings()["graph_interval"] == 2


def test_graph_configuration_button_settings_persist_and_lock(
    viewer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.graph_settings import GRAPH_DEFAULTS
    from topdon_duo.graphs import graph_config_rect
    from topdon_duo.settings_preferences import load_settings, save_settings

    save_settings({"show_graph": True})
    panel = Mock()
    panel.poll.return_value = []
    monkeypatch.setattr(desktop, "GraphPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    settings = {
        **GRAPH_DEFAULTS,
        "range_mode": "fixed",
        "range_seconds": 600,
        "history_points": 8192,
    }
    step = 0

    def key(_delay: Any) -> Any:
        """
        Key.
        """
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_config_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
            panel.poll.return_value = [{"action": "settings", "settings": settings}]
        elif step == 2:
            assert load_settings()["graph_settings"] == settings
            viewer.graphs.logging = True
            panel.poll.return_value = [{"action": "settings", "settings": GRAPH_DEFAULTS}]
        elif step == 3:
            assert load_settings()["graph_settings"] == settings
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    panel.open.assert_called_once()
    assert panel.update.call_args.args[0]["locked"]
    assert [call.args[0] for call in viewer.graphs.configure.call_args_list] == [
        GRAPH_DEFAULTS,
        settings,
    ]
    panel.close.assert_called_once()


@pytest.mark.parametrize("logging", [False, True])
def test_graph_reset_button_only_clears_chart_history_and_obeys_logging_lock(
    viewer: Any, monkeypatch: pytest.MonkeyPatch, logging: Any
) -> None:
    from topdon_duo.graphs import graph_reset_rect
    from topdon_duo.settings_preferences import save_settings

    save_settings({"show_graph": True})
    spots = desktop.SampleSpots(pixels=[(80, 60)])
    spots.rename(1, "Motor")
    monkeypatch.setattr(desktop, "SampleSpots", lambda: spots)
    viewer.graphs.logging = logging
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    step = 0

    def key(_delay: Any) -> Any:
        """
        Key.
        """
        nonlocal step
        step += 1
        if step == 1:
            x0, y0, x1, y1 = graph_reset_rect(768)
            callback = viewer.set_mouse.call_args.args[1]
            callback(cv2.EVENT_LBUTTONUP, 768 + (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
            return -1
        return ord("q")

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    assert viewer.graphs.clear_history.call_count == (0 if logging else 1)
    assert spots.pixels == [(80, 60)] and spots.name(1) == "Motor"
    viewer.graphs.configure.assert_called_once()


__all__ = ["viewer_fixture"]
