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
    frame, preview = frame_with_preview(preview_scale=2)
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
    frame, _ = frame_with_preview(preview_scale=2)
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


def test_app_palette_is_independent_of_camera_colors_and_switching_preserves_temperatures():
    frame, _ = frame_with_preview(preview_scale=2)
    words = np.frombuffer(frame, dtype="<u2").copy()
    words[2320 : 2320 + SENSOR_WIDTH * SENSOR_HEIGHT] = (
        np.arange(SENSOR_WIDTH * SENSOR_HEIGHT) // 32 + 20000
    )
    original = words.tobytes()
    changed = words.copy()
    changed[IMAGE_OFFSET:] = 0x4020
    renderer = ThermalRenderer(scale=1, smoothing=1)
    renderer.camera_preview = renderer.camera_color = True
    renderer.set_view_setting("palette_source", "app")
    first = renderer.render_detailed(original)
    second = renderer.render_detailed(changed.tobytes())
    assert first.image_source == "raw"
    assert np.array_equal(first.image, second.image)
    renderer.set_view_setting("palette_source", "camera")
    camera = renderer.render_detailed(original)
    assert camera.image_source == "preview"
    assert not np.array_equal(first.image, camera.image)
    assert np.array_equal(first.temperatures_celsius, camera.temperatures_celsius)
    assert np.array_equal(first.raw_counts, camera.raw_counts)


def thermal_scene(temperature=30, hot_edge=False):
    from topdon_duo.camera import HEADER_U16, SENSOR_PIXELS

    words = np.frombuffer(make_frame(), dtype='<u2').copy()
    plane = words[HEADER_U16:HEADER_U16 + SENSOR_PIXELS].reshape(192, 256)
    plane[:] = round((temperature + 50) * 64)
    if hot_edge:
        plane[:, :32] = (150 + 50) * 64
    return words.tobytes()


def test_analyze_fixed_colors_do_not_follow_scene_or_display_units():
    renderer = ThermalRenderer(scale=1, smoothing=1, image_source='analyze')
    renderer.native_temperatures = True
    renderer.raw_sharpen_amount = 0
    renderer.raw_upsampling = "off"
    baseline = renderer.render_detailed(thermal_scene())
    changed = renderer.render_detailed(thermal_scene(hot_edge=True))
    assert np.array_equal(baseline.image[96, 128], [128, 128, 128])
    assert np.array_equal(changed.image[96, 128], baseline.image[96, 128])
    assert np.array_equal(changed.image[96, 0], [255, 255, 255])
    renderer.temperature_unit = 'F'
    imperial = renderer.render_detailed(thermal_scene(hot_edge=True))
    assert np.array_equal(changed.image, imperial.image)
    assert changed.temperatures_celsius[96, 128] == 30


def test_analyze_uses_its_own_native_anime_passes_and_preserves_measurements(monkeypatch):
    renderer = ThermalRenderer(scale=2, smoothing=1, image_source='analyze')
    renderer.raw_upsampling = 'anime4k09'
    renderer.native_temperatures = True
    renderer.upsampling = 'acnet-legacy-hdn3'
    renderer.anime4k_passes = 5
    calls = []

    def enhance(image, model, amount, passes):
        assert np.array_equal(image[..., 0], image[..., 1])
        assert np.array_equal(image[..., 0], image[..., 2])
        calls.append((image.shape, model, amount, passes))
        return cv2.resize(image, (512, 384))

    monkeypatch.setattr(renderer.upsampler, 'apply', enhance)
    result = renderer.render_detailed(thermal_scene())
    assert calls == [((192, 256, 3), 'anime4k09', 1.0, 3)]
    assert result.image.shape == (384, 512, 3)
    assert np.all(result.temperatures_celsius == 30)
    assert np.all(result.raw_counts == 5120)
    assert result.stats.minimum == result.stats.maximum == 30


def test_analyze_bounds_reject_crossing_and_invalid_values():
    renderer = ThermalRenderer(image_source='analyze')
    for name, value in [('raw_temperature_low', 45), ('raw_temperature_high', 15),
                        ('raw_temperature_low', float('nan')),
                        ('raw_sharpen_amount', 1.1), ('raw_anime4k', 1)]:
        with pytest.raises(ValueError):
            renderer.set_view_setting(name, value)
    assert (renderer.raw_temperature_low, renderer.raw_temperature_high) == (15, 45)


def test_analyze_toggle_restores_source_and_does_not_change_measurements():
    renderer = ThermalRenderer(scale=1, smoothing=1, image_source='preview')
    renderer.native_temperatures = True
    renderer.raw_upsampling = "off"
    original = renderer.render_detailed(thermal_scene())
    renderer.set_view_setting('analyze_mode', True)
    analyzed = renderer.render_detailed(thermal_scene())
    assert renderer.image_source == 'preview'
    assert analyzed.image_source == 'raw'
    assert np.array_equal(original.temperatures_celsius, analyzed.temperatures_celsius)
    renderer.set_view_setting('analyze_mode', False)
    restored = renderer.render_detailed(thermal_scene())
    assert renderer.image_source == 'preview'
    assert np.array_equal(original.image, restored.image)


def test_analyze_model_receives_grayscale_before_optional_palette(monkeypatch):
    renderer = ThermalRenderer(scale=2, smoothing=1, image_source='raw')
    renderer.set_view_setting('analyze_mode', True)
    renderer.set_view_setting('raw_upsampling', 'acnet-legacy-hdn0')
    renderer.set_view_setting('raw_palette', 'inferno')
    calls = []

    def enhance(image, model, amount, passes):
        assert np.array_equal(image[..., 0], image[..., 1])
        assert np.array_equal(image[..., 0], image[..., 2])
        calls.append(model)
        return np.full((384, 512, 3), 128, dtype=np.uint8)

    monkeypatch.setattr(renderer.upsampler, 'apply', enhance)
    result = renderer.render_detailed(thermal_scene())
    assert calls == ['acnet-legacy-hdn0']
    expected = cv2.applyColorMap(np.full((384, 512), 128, dtype=np.uint8), cv2.COLORMAP_INFERNO)
    assert np.array_equal(result.image, expected)


def test_sensor_interpolation_uses_float_plane_without_enhancement(monkeypatch):
    renderer = ThermalRenderer(scale=2, smoothing=1, image_source='analyze')
    assert renderer.raw_upsampling == 'bicubic'
    assert renderer.raw_sharpen_amount == 0
    counts = np.tile(np.linspace(64 * 70, 64 * 90, 256, dtype=np.float32), (192, 1))

    def forbidden(*args, **kwargs):
        raise AssertionError('Sensor interpolation must not invoke model enhancement')

    monkeypatch.setattr(renderer.upsampler, 'apply', forbidden)
    actual = renderer._render_raw(counts)
    intensity = (counts / 64 - 50 - 15) * (255 / 30)
    expected = cv2.resize(intensity, (512, 384), interpolation=cv2.INTER_CUBIC)
    expected = np.clip(expected, 0, 255).round().astype(np.uint8)
    assert np.array_equal(actual, cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR))
