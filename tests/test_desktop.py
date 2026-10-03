import json

import numpy as np
from test_camera import make_frame

from topdon_duo.desktop import save_capture
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
