import json

import numpy as np
from test_camera import make_frame

from topdon_duo.desktop import (
    MousePicker,
    draw_control_instructions,
    draw_picker,
    save_capture,
)
from topdon_duo.render import ThermalRenderer


def test_detailed_render_keeps_oriented_radiometric_arrays():
    rendered = ThermalRenderer(rotation=90).render_detailed(make_frame())
    assert rendered.temperatures_celsius.shape == (256, 192)
    assert rendered.raw_counts.shape == (256, 192)
    assert rendered.image.shape == (768, 576, 3)


def test_save_capture_preserves_raw_and_temperature_data(tmp_path):
    rendered = ThermalRenderer().render_detailed(make_frame())
    png_path, data_path, json_path = save_capture(
        rendered,
        tmp_path,
        ambient_celsius=21.9,
        rotation=0,
        selected_pixel=(10, 20),
    )
    assert png_path.exists()
    with np.load(data_path) as data:
        assert np.array_equal(data["raw_counts"], rendered.raw_counts)
        assert np.allclose(data["temperatures_celsius"], rendered.temperatures_celsius)
        assert float(data["ambient_celsius"]) == np.float32(21.9)
    metadata = json.loads(json_path.read_text())
    assert metadata["selected_pixel"]["x"] == 10
    assert metadata["selected_pixel"]["raw_count"] == 20_000


def test_control_instructions_preserve_shape_and_draw_overlay():
    image = np.zeros((576, 768, 3), dtype=np.uint8)
    result = draw_control_instructions(image)
    assert result.shape == image.shape
    assert np.count_nonzero(result) > 0


def test_mouse_wheel_accumulates_signed_ambient_steps():
    picker = MousePicker()
    picker.callback(10, 0, 0, 120 << 16, None)
    picker.callback(10, 0, 0, 0xFF88 << 16, None)
    picker.callback(10, 0, 0, 120 << 16, None)
    assert picker.consume_ambient_steps() == 1
    assert picker.consume_ambient_steps() == 0


def test_picker_maps_resized_viewport_to_sensor_pixel():
    rendered = ThermalRenderer(scale=3).render_detailed(make_frame())
    picker = MousePicker(x=192, y=144)
    _image, selected = draw_picker(rendered, picker, scale=3, viewport_size=(384, 288))
    assert selected == (128, 96)
