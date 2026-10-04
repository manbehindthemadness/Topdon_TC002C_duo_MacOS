import cv2
import numpy as np
import pytest
from test_camera import make_frame

from topdon_duo.camera import IMAGE_OFFSET, SENSOR_HEIGHT, SENSOR_WIDTH, decode_duo_frame
from topdon_duo.desktop import MousePicker, SampleSpots, draw_picker, draw_sample_spots
from topdon_duo.render import ThermalRenderer


def frame_with_preview(preview_scale=2):
    preview = np.zeros((SENSOR_HEIGHT * preview_scale, SENSOR_WIDTH * preview_scale), np.uint8)
    # Place detail beyond the first 256x192 pixels of the large preview.
    preview[preview.shape[0] // 2 :, preview.shape[1] // 2 :] = 200
    words = preview.astype("<u2") | 0x8000
    return make_frame()[: IMAGE_OFFSET * 2] + words.tobytes(), preview


@pytest.mark.parametrize("preview_scale", [1, 2])
def test_decode_keeps_complete_preview_and_discards_chroma(preview_scale):
    frame, expected = frame_with_preview(preview_scale)
    _, raw, preview = decode_duo_frame(frame)
    assert np.array_equal(preview, expected)
    assert raw.shape == (192, 256)
    assert np.all(raw == 20_000)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_camera_preview_preserves_detail_and_rotates_with_measurements(rotation):
    frame, preview = frame_with_preview()
    rendered = ThermalRenderer(scale=2, rotation=rotation).render_detailed(frame)
    expected = cv2.applyColorMap(
        np.where(preview == 200, 255, 0).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    expected = np.rot90(expected, -(rotation // 90))
    assert np.array_equal(rendered.image, expected)
    assert rendered.image_source == "preview"
    assert rendered.image.shape[:2] == tuple(size * 2 for size in rendered.raw_counts.shape)
    assert np.all(rendered.temperatures_celsius == 22)
    assert np.all(rendered.raw_counts == 20_000)


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_preview_keeps_mouse_and_spots_on_sensor_pixels(scale):
    frame, _ = frame_with_preview()
    rendered = ThermalRenderer(scale=scale, rotation=270).render_detailed(frame)
    x, y = 7 * scale + scale // 2, 11 * scale + scale // 2
    image, selected = draw_picker(rendered, MousePicker(x=x, y=y), scale)
    assert selected == (7, 11)
    assert np.array_equal(image[y, x], 255 - rendered.image[y, x])
    image = draw_sample_spots(rendered.image, rendered, SampleSpots(pixels=[selected]), scale)
    assert np.array_equal(image[y, x], 255 - rendered.image[y, x])


def test_view_switch_changes_only_image_and_empty_preview_falls_back():
    frame, _ = frame_with_preview()
    renderer = ThermalRenderer(scale=2)
    camera = renderer.render_detailed(frame)
    assert renderer.toggle_image_source() == "raw"
    raw = renderer.render_detailed(frame)
    assert not np.array_equal(camera.image, raw.image)
    assert camera.stats == raw.stats
    assert np.array_equal(camera.raw_counts, raw.raw_counts)
    assert np.array_equal(camera.temperatures_celsius, raw.temperatures_celsius)
    assert raw.image_source == "raw"
    assert renderer.toggle_image_source() == "preview"
    empty = renderer.render_detailed(make_frame())
    assert empty.image_source == "raw"
    assert np.array_equal(empty.image, raw.image)
