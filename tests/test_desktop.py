import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from test_camera import make_frame

from topdon_duo.desktop import (
    LinuxSaveDialog,
    MacSaveDialog,
    MousePicker,
    ambient_to_trackbar,
    clamp_ambient,
    draw_control_instructions,
    draw_picker,
    draw_toolbar,
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
