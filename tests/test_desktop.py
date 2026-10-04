import json
from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from test_camera import make_frame

from topdon_duo.desktop import (
    LinuxSaveDialog,
    MacSaveDialog,
    MousePicker,
    SampleSpots,
    ambient_to_trackbar,
    clamp_ambient,
    draw_control_instructions,
    draw_picker,
    draw_sample_spots,
    draw_toolbar,
    image_position_at,
    mouse_viewport_size,
    save_capture,
    toolbar_action_at,
    toolbar_layout,
    trackbar_to_ambient,
)
from topdon_duo.render import ThermalRenderer


def test_detailed_render_keeps_oriented_radiometric_arrays():
    rendered = ThermalRenderer(rotation=90).render_detailed(make_frame())
    assert rendered.temperatures_celsius.shape == (256, 192)
    assert rendered.raw_counts.shape == (256, 192)
    assert rendered.image.shape == (768, 576, 3)


def test_save_capture_preserves_raw_and_temperature_data(tmp_path):
    chosen_path = tmp_path / "living-room.png"
    rendered = ThermalRenderer().render_detailed(make_frame())
    png_path, data_path, json_path = save_capture(
        rendered,
        tmp_path,
        ambient_celsius=21.9,
        rotation=0,
        selected_pixel=(10, 20),
        display_unit="F",
        base_path=chosen_path,
    )
    assert png_path == chosen_path
    assert png_path.exists()
    with np.load(data_path) as data:
        assert np.array_equal(data["raw_counts"], rendered.raw_counts)
        assert np.allclose(data["temperatures_celsius"], rendered.temperatures_celsius)
        assert float(data["ambient_celsius"]) == np.float32(21.9)
    metadata = json.loads(json_path.read_text())
    assert metadata["selected_pixel"]["x"] == 10
    assert metadata["selected_pixel"]["raw_count"] == 20_000
    assert metadata["display_unit"] == "F"


def test_native_save_dialog_returns_selected_path(monkeypatch, tmp_path):
    chosen_path = tmp_path / "thermal.png"

    class FinishedProcess:
        returncode = 0

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def communicate():
            return f"{chosen_path}\n", ""

    monkeypatch.setattr(
        "topdon_duo.desktop.subprocess.Popen", lambda *_args, **_kwargs: FinishedProcess()
    )
    dialog = MacSaveDialog()
    assert dialog.open(tmp_path)
    assert dialog.is_open
    assert dialog.poll() == (True, chosen_path)
    assert not dialog.is_open


def test_save_dialog_commands_preserve_mac_and_support_linux(tmp_path):
    # Paths with spaces remain a single subprocess argument.
    directory = str(tmp_path / "thermal captures")
    mac_command = MacSaveDialog()._command("capture.png", directory)
    assert mac_command[0] == "/usr/bin/osascript"
    assert mac_command[-2:] == ["capture.png", directory]
    linux_command = LinuxSaveDialog()._command("capture.png", directory)
    assert linux_command[0] == "zenity"
    assert "--confirm-overwrite" in linux_command
    assert f"--filename={directory}/capture.png" in linux_command


def test_linux_save_cancellation_and_nonblocking_poll(monkeypatch, caplog):
    process = Mock(returncode=1)
    process.poll.side_effect = [None, 1]
    process.communicate.return_value = ("", "")
    monkeypatch.setattr("topdon_duo.desktop.subprocess.Popen", lambda *a, **kw: process)
    dialog = LinuxSaveDialog()
    assert dialog.open()
    assert not dialog.open()
    assert dialog.poll() == (False, None)
    process.communicate.assert_not_called()
    assert dialog.poll() == (True, None)
    assert not dialog.is_open
    assert not caplog.records


def test_control_instructions_preserve_shape_and_draw_overlay():
    image = np.zeros((576, 768, 3), dtype=np.uint8)
    result = draw_control_instructions(image)
    assert result.shape == image.shape
    assert np.count_nonzero(result) > 0


def test_mouse_wheel_accumulates_signed_ambient_steps():
    picker = MousePicker()
    picker.callback(10, 0, 0, 120 << 16, None)
    picker.callback(10, 0, 0, 0xFF88 << 16, None)
    picker.callback(11, 0, 0, 120 << 16, None)
    assert picker.consume_ambient_steps() == 1
    assert picker.consume_ambient_steps() == 0


def test_ambient_trackbar_conversion_and_clamping():
    assert ambient_to_trackbar(21.9) == 719
    assert trackbar_to_ambient(719) == 21.9
    assert clamp_ambient(-100.0) == -50.0
    assert clamp_ambient(200.0) == 100.0


def test_picker_maps_resized_viewport_to_sensor_pixel():
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    picker = MousePicker(x=192, y=144)
    _image, selected = draw_picker(rendered, picker, scale=3, viewport_size=(384, 288))
    assert selected == (128, 96)


def test_mouse_picker_matches_thin_spot_marker():
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    image, selected = draw_picker(rendered, MousePicker(x=123, y=201), scale=3)
    assert selected == (41, 67)
    assert tuple(image[201, 123]) == (80, 255, 80)
    assert tuple(image[204, 123]) == (80, 255, 80)
    assert np.array_equal(image[204, 124], rendered.image[204, 124])
    assert np.array_equal(image[207, 123], rendered.image[207, 123])


def test_mouse_picker_hides_on_window_exit_without_mouse_event_and_reappears(monkeypatch):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    picker = MousePicker(x=123, y=201)
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr("topdon_duo.desktop.cv2.putText", put_text)
    image, selected = draw_picker(rendered, picker, scale=3, pointer_over_image=False)
    assert selected is None
    assert np.array_equal(image, rendered.image)
    put_text.assert_not_called()
    # The last callback coordinates are unchanged when leaving the window.
    assert (picker.x, picker.y) == (123, 201)
    image, selected = draw_picker(rendered, picker, scale=3, pointer_over_image=True)
    assert selected == (41, 67)
    assert not np.array_equal(image, rendered.image)
    assert put_text.call_count == 2


@pytest.mark.parametrize("position", [(100, 0), (-1, 200), (768, 200), None])
def test_mouse_picker_hides_marker_and_reading_outside_image(position):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    layout = toolbar_layout(rendered.image.shape[1])
    if position is None:
        position = (100, rendered.image.shape[0] + layout.height)
    image, selected = draw_picker(
        rendered,
        MousePicker(x=position[0], y=position[1]),
        scale=3,
        toolbar_height=layout.height,
    )
    assert selected is None
    assert np.array_equal(image, rendered.image)


@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize("viewport_scale", [0.5, 1.0, 1.5])
def test_linux_mouse_coordinates_are_already_image_pixels(monkeypatch, rotation, viewport_scale):
    monkeypatch.setattr("topdon_duo.desktop.sys.platform", "linux")
    rendered = ThermalRenderer(scale=3, rotation=rotation).render_detailed(make_frame())
    layout = toolbar_layout(rendered.image.shape[1])
    canvas_height = rendered.image.shape[0] + layout.height
    get_rect = Mock(
        return_value=(
            100,
            200,
            int(rendered.image.shape[1] * viewport_scale),
            int(canvas_height * viewport_scale),
        )
    )
    monkeypatch.setattr("topdon_duo.desktop.cv2.getWindowImageRect", get_rect)
    viewport = mouse_viewport_size()
    # Qt/GTK have already mapped the physical pointer to these canvas pixels.
    picker = MousePicker()
    picker.callback(cv2.EVENT_MOUSEMOVE, 123, 201 + layout.height, 0, None)
    image, selected = draw_picker(
        rendered, picker, scale=3, viewport_size=viewport, toolbar_height=layout.height
    )
    assert selected == (41, 67)
    assert tuple(image[201, 123]) == (80, 255, 80)
    picker.callback(cv2.EVENT_MOUSEMOVE, 123, layout.height - 1, 0, None)
    assert (
        draw_picker(
            rendered, picker, scale=3, viewport_size=viewport, toolbar_height=layout.height
        )[1]
        is None
    )

    x0, y0, x1, y1 = layout.buttons["unit"]
    picker.callback(cv2.EVENT_LBUTTONUP, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
    click_x, click_y = picker.consume_clicks()[0]
    assert (
        toolbar_action_at(
            click_x,
            click_y,
            rendered.image.shape[1],
            viewport_size=viewport,
            canvas_height=canvas_height,
        )
        == "unit"
    )
    get_rect.assert_not_called()


def test_mac_mouse_coordinates_keep_viewport_scaling(monkeypatch):
    monkeypatch.setattr("topdon_duo.desktop.sys.platform", "darwin")
    get_rect = Mock(return_value=(100, 200, 384, 288))
    monkeypatch.setattr("topdon_duo.desktop.cv2.getWindowImageRect", get_rect)
    assert mouse_viewport_size() == (384, 288)
    get_rect.assert_called_once()
    get_rect.side_effect = cv2.error("Image rectangle unavailable")
    assert mouse_viewport_size() is None


def test_toolbar_draws_above_image_and_maps_resized_clicks():
    image = np.zeros((576, 768, 3), dtype=np.uint8)
    layout = toolbar_layout(image.shape[1])
    result = draw_toolbar(image, ambient_celsius=21.9, temperature_unit="F")
    assert result.shape == (576 + layout.height, 768, 3)
    x0, y0, x1, y1 = layout.buttons["unit"]
    viewport = (384, result.shape[0] // 2)
    action = toolbar_action_at(
        (x0 + x1) // 4,
        (y0 + y1) // 4,
        image.shape[1],
        viewport_size=viewport,
        canvas_height=result.shape[0],
    )
    assert action == "unit"


def test_renderer_toggles_fahrenheit_display_conversion():
    renderer = ThermalRenderer()
    assert renderer.display_temperature(20.0) == 20.0
    assert renderer.toggle_temperature_unit() == "F"
    assert renderer.display_temperature(20.0) == 68.0


@pytest.mark.parametrize("viewport_scale", [0.5, 1.0, 1.5])
def test_spot_placement_maps_viewport_and_rejects_toolbar_and_outside(viewport_scale):
    shape = (576, 768, 3)
    toolbar_height = toolbar_layout(shape[1]).height
    viewport = (int(shape[1] * viewport_scale), int((shape[0] + toolbar_height) * viewport_scale))
    assert image_position_at(
        round(240 * viewport_scale),
        round((180 + toolbar_height) * viewport_scale),
        shape,
        viewport,
        toolbar_height,
    ) == (240, 180)
    for x, y in ((-2, 200), (768, 200), (200, 0), (200, 576 + toolbar_height)):
        assert (
            image_position_at(
                round(x * viewport_scale),
                round(y * viewport_scale),
                shape,
                viewport,
                toolbar_height,
            )
            is None
        )


def test_spots_stay_on_same_sensor_pixels_through_four_rotations():
    from topdon_duo.camera import HEADER_U16, SENSOR_PIXELS

    values = np.frombuffer(make_frame(), dtype="<u2").copy()
    values[HEADER_U16 : HEADER_U16 + SENSOR_PIXELS] = np.arange(SENSOR_PIXELS) + 10000
    frame = values.tobytes()
    renderer = ThermalRenderer()
    spots = SampleSpots()
    spots.add((41, 67))
    assert spots.pixels == []
    spots.toggle()
    spots.add((41, 67))
    spots.add((41, 67))
    spots.add(None)
    for _ in range(4):
        before = renderer.render_detailed(frame)
        x, y = spots.pixels[0]
        temperature = before.temperatures_celsius[y, x]
        spots.rotate_clockwise(before.temperatures_celsius.shape[0])
        renderer.rotate_clockwise()
        after = renderer.render_detailed(frame)
        x, y = spots.pixels[0]
        assert after.temperatures_celsius[y, x] == temperature
    assert spots.pixels == [(41, 67)]
    spots.toggle()
    assert not spots.placing
    assert spots.pixels == []


def test_spots_draw_thin_green_markers_and_live_readings(monkeypatch):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    spots = SampleSpots(placing=True, pixels=[(41, 67), (80, 90)])
    original = rendered.image.copy()
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr("topdon_duo.desktop.cv2.putText", put_text)
    for temperature, unit, expected in ((20, "C", "20.00 C"), (25, "F", "77.00 F")):
        frame = replace(rendered, temperatures_celsius=np.full((192, 256), temperature))
        result = draw_sample_spots(rendered.image, frame, spots, 3, unit)
        assert [call.args[1] for call in put_text.call_args_list] == [expected] * 4
        put_text.reset_mock()
        for x, y in spots.pixels:
            ix, iy = x * 3 + 1, y * 3 + 1
            assert tuple(result[iy, ix]) == (80, 255, 80)
            assert tuple(result[iy + 3, ix]) == (80, 255, 80)
            assert np.array_equal(result[iy + 3, ix + 1], original[iy + 3, ix + 1])
    assert np.array_equal(rendered.image, original)


def test_desktop_spot_control_places_multiple_spots_rotates_and_clears(monkeypatch):
    from topdon_duo import desktop

    camera = Mock()
    camera.frames.return_value = [make_frame()] * 8
    monkeypatch.setattr(desktop, "TC002CDuoCamera", lambda: camera)
    monkeypatch.setattr(desktop, "mouse_viewport_size", lambda: None)
    pointer_monitor = Mock()
    pointer_monitor.over_image.return_value = True
    monkeypatch.setattr(desktop, "PointerMonitor", lambda _name: pointer_monitor)
    for name in ("namedWindow", "createTrackbar", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(desktop.cv2, name, Mock())
    set_callback = Mock()
    monkeypatch.setattr(desktop.cv2, "setMouseCallback", set_callback)
    states = []
    draw_spots = desktop.draw_sample_spots

    def record_spots(image, rendered, spots, scale, unit):
        states.append((spots.placing, spots.pixels.copy()))
        center = rendered.image[rendered.image.shape[0] // 2, rendered.image.shape[1] // 2]
        assert bool(np.all(center == 255)) == (not spots.placing)
        return draw_spots(image, rendered, spots, scale, unit)

    monkeypatch.setattr(desktop, "draw_sample_spots", record_spots)
    step = 0

    def wait_key(_delay):
        nonlocal step
        step += 1
        callback = set_callback.call_args.args[1]
        width = 576 if step >= 6 else 768
        layout = toolbar_layout(width)

        def click(x, y):
            callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)

        if step == 1:
            click(123, 201 + layout.height)  # Placement is initially disabled.
        elif step in (2, 6):
            x0, y0, x1, y1 = layout.buttons["spots"]
            click((x0 + x1) // 2, (y0 + y1) // 2)
        elif step == 3:
            click(123, 201 + layout.height)
            click(240, 270 + layout.height)
            click(123, 201 + layout.height)  # Duplicate sensor pixel.
            click(-10, 200)
            click(0, 0)  # Toolbar padding is not an image pixel.
        elif step == 5:
            return ord("o")
        elif step == 8:
            return ord("q")
        return -1

    monkeypatch.setattr(desktop.cv2, "waitKey", wait_key)
    assert desktop.main([]) == 0
    assert states == [
        (False, []),
        (False, []),
        (False, []),
        (True, []),
        (True, [(41, 67), (80, 90)]),
        (True, [(124, 41), (101, 80)]),
        (True, [(124, 41), (101, 80)]),
        (False, []),
    ]
    camera.close.assert_called_once()
    pointer_monitor.close.assert_called_once()
