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
    draw_control_instructions,
    draw_picker,
    draw_sample_spots,
    draw_toolbar,
    image_position_at,
    mouse_viewport_size,
    save_capture,
    toolbar_action_at,
    toolbar_layout,
)
from topdon_duo.render import READOUT_HEIGHT, ThermalRenderer, draw_temperature_readout


@pytest.mark.parametrize("platform,expected", [("linux", 32), ("darwin", 0)])
def test_usb_queue_default_and_synchronous_override(monkeypatch, platform, expected):
    from topdon_duo import desktop

    monkeypatch.setattr(desktop.sys, "platform", platform)
    assert desktop.parse_args([]).usb_queue_depth == expected
    assert desktop.parse_args(["--usb-queue-depth", "0"]).usb_queue_depth == 0


def test_detailed_render_keeps_oriented_radiometric_arrays():
    rendered = ThermalRenderer(rotation=90).render_detailed(make_frame())
    assert rendered.temperatures_celsius.shape == (256, 192)
    assert rendered.raw_counts.shape == (256, 192)
    assert rendered.image.shape == (768, 576, 3)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("unit", ["C", "F"])
def test_readout_preserves_entire_heatmap_and_top_row_sampling(rotation, unit, tmp_path):
    renderer = ThermalRenderer(rotation=rotation, temperature_unit=unit)
    rendered = renderer.render_detailed(make_frame())
    # A uniform sensor frame must stay uniform, including the formerly covered rows.
    assert np.all(rendered.image == rendered.image[-1, 0])
    annotated = draw_temperature_readout(rendered.image, rendered.stats, 22.0, unit)
    assert np.array_equal(annotated[READOUT_HEIGHT:], rendered.image)
    assert np.any(annotated[:READOUT_HEIGHT])
    web_image, _stats = renderer.render(make_frame())
    assert np.array_equal(web_image, annotated)
    png_path, *_ = save_capture(rendered, tmp_path, 22.0, rotation, display_unit=unit)
    assert np.array_equal(cv2.imread(str(png_path)), annotated)

    layout = toolbar_layout(rendered.image.shape[1])
    display = draw_toolbar(rendered.image, 22.0, unit, stats=rendered.stats)
    assert np.array_equal(display[layout.height - READOUT_HEIGHT :], annotated)
    for viewport_scale in (1, 0.5, 2):
        viewport = (
            round(display.shape[1] * viewport_scale),
            round(display.shape[0] * viewport_scale),
        )
        _image, selected = draw_picker(
            rendered,
            MousePicker(x=0, y=round(layout.height * viewport_scale)),
            renderer.scale,
            viewport_size=viewport,
            toolbar_height=layout.height,
        )
        assert selected == (0, 0)
        assert (
            image_position_at(
                0,
                round((layout.height - 2) * viewport_scale),
                rendered.image.shape,
                viewport,
                layout.height,
            )
            is None
        )


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
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
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
    csv_command = LinuxSaveDialog()._command("temperatures.csv", directory)
    assert "--title=Save temperature log" in csv_command
    assert "--file-filter=CSV logs | *.csv" in csv_command
    assert "Save temperature log" in MacSaveDialog()._command("temperatures.csv", directory)[2]


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


def test_picker_maps_resized_viewport_to_sensor_pixel():
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    picker = MousePicker(x=192, y=144)
    _image, selected = draw_picker(rendered, picker, scale=3, viewport_size=(384, 288))
    assert selected == (128, 96)


def test_mouse_picker_matches_thin_spot_marker():
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    image, selected = draw_picker(rendered, MousePicker(x=123, y=201), scale=3)
    assert selected == (41, 67)
    assert np.array_equal(image[201, 123], 255 - rendered.image[201, 123])
    assert np.array_equal(image[204, 123], 255 - rendered.image[204, 123])
    assert tuple(image[204, 124]) in ((0, 0, 0), (255, 255, 255))
    assert np.array_equal(image[204, 125], rendered.image[204, 125])
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
    assert put_text.call_count == 1


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
    assert np.array_equal(image[201, 123], 255 - rendered.image[201, 123])
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
    assert spots.pixels == [(41, 67)]


def test_spots_draw_inverted_markers_and_live_readings(monkeypatch):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    spots = SampleSpots(placing=True, pixels=[(41, 67), (80, 90)])
    original = rendered.image.copy()
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr("topdon_duo.desktop.cv2.putText", put_text)
    for temperature, unit, expected in ((20, "C", "20.00 C"), (25, "F", "77.00 F")):
        frame = replace(rendered, temperatures_celsius=np.full((192, 256), temperature))
        result = draw_sample_spots(rendered.image, frame, spots, 3, unit)
        assert [call.args[1] for call in put_text.call_args_list] == [
            f"1: {expected}",
            f"2: {expected}",
        ]
        put_text.reset_mock()
        for x, y in spots.pixels:
            ix, iy = x * 3 + 1, y * 3 + 1
            assert np.array_equal(result[iy, ix], 255 - original[iy, ix])
            assert np.array_equal(result[iy + 3, ix], 255 - original[iy + 3, ix])
            assert tuple(result[iy + 3, ix + 1]) in ((0, 0, 0), (255, 255, 255))
            assert np.array_equal(result[iy + 3, ix + 2], original[iy + 3, ix + 2])
    assert np.array_equal(rendered.image, original)


def test_desktop_spot_control_places_multiple_spots_rotates_and_stops_placement(
    monkeypatch, tmp_path
):
    from topdon_duo import desktop

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    graphs = Mock(logging=False)
    graphs.take_logging_error.return_value = None
    monkeypatch.setattr(desktop, "GraphWorker", lambda: graphs)
    hardware = Mock(original={}, enabled=set(), error="", preview_active=False)
    hardware.state.return_value = {}
    monkeypatch.setattr(desktop, "HardwareControls", lambda _camera: hardware)
    camera = Mock()
    camera.frames.return_value = [make_frame()] * 8
    monkeypatch.setattr(desktop, "TC002CDuoCamera", lambda: camera)

    class FramePump:
        def __init__(self, source):
            self._source = source

        def __iter__(self):
            return iter(self._source.frames())

        def close(self):
            pass

    monkeypatch.setattr(desktop, "CameraFramePump", FramePump)
    awake = Mock()
    awake.check.return_value = None
    monkeypatch.setattr(desktop, "DisplayAwake", lambda: awake)
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
        assert not np.all(center == 255)
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
        (False, [(124, 41), (101, 80)]),
    ]
    camera.close.assert_called_once()
    pointer_monitor.close.assert_called_once()


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 255, 255), (127, 127, 127), (20, 180, 240)])
@pytest.mark.parametrize("sampler", ["fixed", "mouse"])
def test_sampler_contrast_on_dark_bright_gray_and_colored_backgrounds(color, sampler):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    background = np.full_like(rendered.image, color)
    rendered = replace(rendered, image=background)
    if sampler == "fixed":
        # Duplicate/overlapping spots must not cancel the inversion.
        image = draw_sample_spots(background, rendered, SampleSpots(pixels=[(41, 67)] * 2), 3)
        ix, iy = 124, 202
    else:
        ix, iy = 123, 201
        image, selected = draw_picker(rendered, MousePicker(x=ix, y=iy), 3)
        assert selected == (41, 67)
    assert np.array_equal(image[iy, ix], 255 - background[iy, ix])
    assert tuple(image[iy + 3, ix + 1]) in ((0, 0, 0), (255, 255, 255))
    assert np.max(np.abs(image[iy, ix].astype(int) - image[iy + 3, ix + 1])) >= 128
    # Labels keep white interiors on every background, including hot/cold edges.
    label_region = image[: iy - 5, ix + 8 :]
    assert np.any(np.all(label_region == 255, axis=2))
    assert np.array_equal(image[-1, -1], background[-1, -1])
    assert np.all(background == color)


def label_bounds(call):
    text, (x, y) = call.args[1:3]
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    return x - 3, y - height - 3, x + width + 3, y + baseline + 3


def rectangles_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize(
    "pixels",
    [
        [(30, 41), (37, 41), (46, 41)],
        [(0, 0), (1, 0), (2, 0), (255, 0), (255, 191), (0, 191)],
        [(128, 96)] * 8,
    ],
)
@pytest.mark.parametrize("unit", ["C", "F"])
def test_spot_labels_avoid_each_other_markers_and_image_edges(monkeypatch, pixels, unit):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    spots = SampleSpots(pixels=pixels)
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr("topdon_duo.desktop.cv2.putText", put_text)
    image = draw_sample_spots(rendered.image, rendered, spots, 3, unit)
    boxes = [label_bounds(call) for call in put_text.call_args_list]
    assert len(boxes) == len(pixels)
    markers = [(x * 3 + 1 - 6, y * 3 + 1 - 6, x * 3 + 1 + 7, y * 3 + 1 + 7) for x, y in pixels]
    for index, box in enumerate(boxes):
        assert 0 <= box[0] < box[2] <= image.shape[1]
        assert 0 <= box[1] < box[3] <= image.shape[0]
        assert all(not rectangles_overlap(box, other) for other in boxes[index + 1 :])
        assert all(not rectangles_overlap(box, marker) for marker in markers)
    # Placement is deterministic as frames update.
    put_text.reset_mock()
    draw_sample_spots(rendered.image, rendered, spots, 3, unit)
    assert [label_bounds(call) for call in put_text.call_args_list] == boxes


def test_mouse_reading_avoids_fixed_spot_labels(monkeypatch):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    spots = SampleSpots(pixels=[(30, 41), (37, 41), (46, 41)])
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr("topdon_duo.desktop.cv2.putText", put_text)
    draw_sample_spots(rendered.image, rendered, spots, 3)
    fixed_boxes = [label_bounds(call) for call in put_text.call_args_list]
    put_text.reset_mock()
    _, selected = draw_picker(rendered, MousePicker(x=112, y=140), 3, spots=spots)
    assert selected == (37, 46)
    mouse_box = label_bounds(put_text.call_args)
    assert all(not rectangles_overlap(mouse_box, box) for box in fixed_boxes)


def test_all_labels_including_first_have_connecting_lines(monkeypatch):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    # These overlapping anchors place readings just below/right and left of
    # the crosshair, both within the old 16-pixel cutoff for connecting lines.
    spots = SampleSpots(pixels=[(41, 67)] * 3)
    line = Mock(wraps=cv2.line)
    monkeypatch.setattr("topdon_duo.desktop.cv2.line", line)
    draw_sample_spots(rendered.image, rendered, spots, 3)
    assert line.call_count == 3
    for call in line.call_args_list:
        start, end = call.args[1:3]
        assert start == (124, 202)
        assert 0 < max(abs(end[0] - start[0]), abs(end[1] - start[1])) <= 16


def test_displaced_mouse_label_has_connecting_line(monkeypatch):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    spots = SampleSpots(pixels=[(41, 67)])
    line = Mock(wraps=cv2.line)
    monkeypatch.setattr("topdon_duo.desktop.cv2.line", line)
    _, selected = draw_picker(rendered, MousePicker(x=124, y=214), 3, spots=spots)
    assert selected == (41, 71)
    line.assert_called_once()
    assert line.call_args.args[1] == (124, 214)


@pytest.mark.parametrize("viewport_size", [None, (384, 303)])
@pytest.mark.parametrize(
    "offset, enabled, dragging, hidden",
    [
        (0, True, False, True),
        (9, True, False, True),
        (11, True, False, False),
        (0, False, False, False),
        (40, True, True, True),
    ],
)
def test_cursor_label_hidden_over_enabled_spot_or_during_drag(
    monkeypatch, viewport_size, offset, enabled, dragging, hidden
):
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    spots = SampleSpots(pixels=[(41, 67)])
    spots.set_enabled(1, enabled)
    toolbar_height = 30
    image_x, image_y = 124, 202
    if viewport_size is None:
        x, y = image_x + offset, image_y + toolbar_height
    else:
        x, y = image_x // 2 + offset, (image_y + toolbar_height) // 2
    put_text = Mock(wraps=cv2.putText)
    line = Mock(wraps=cv2.line)
    monkeypatch.setattr("topdon_duo.desktop.cv2.putText", put_text)
    monkeypatch.setattr("topdon_duo.desktop.cv2.line", line)
    image, selected = draw_picker(
        rendered,
        MousePicker(x=x, y=y),
        3,
        viewport_size=viewport_size,
        toolbar_height=toolbar_height,
        spots=spots,
        dragging_spot=dragging,
    )
    assert selected is not None
    assert put_text.call_count == (0 if hidden else 1)
    assert line.call_count == (0 if hidden else 1)
    # Inspection remains active and the cursor crosshair still draws.
    cursor_x = image_x + offset * (2 if viewport_size else 1)
    assert np.array_equal(image[image_y, cursor_x], 255 - rendered.image[image_y, cursor_x])
    put_text.reset_mock()
    draw_sample_spots(image, rendered, spots, 3)
    assert put_text.call_count == (1 if enabled else 0)
