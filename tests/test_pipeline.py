"""Pipeline behavior: ordered image operations, safe ownership, and portable state."""

from copy import deepcopy
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np
import pytest
from support.pipeline import fake_hardware, process, raw_pipeline
from test_render import frame_with_preview

from topdon_duo.camera import (
    HEADER_U16,
    IMAGE_OFFSET,
    SENSOR_PIXELS,
    FrameAssembler,
    decode_duo_frame,
)
from topdon_duo.pipeline import (
    default_pipeline,
    geometry,
    migrate_pipeline,
    node,
    thermal_source,
    validate_pipeline,
)
from topdon_duo.pipeline_hardware import PipelineHardware, desired_hardware
from topdon_duo.pipeline_processing import PipelineProcessor, PipelineWorker
from topdon_duo.render import ThermalRenderer


@pytest.mark.parametrize("input_kind", ["source", "blend", "mask", "branch"])
def test_raw_visual_inputs_ignore_measurement_average(input_kind: Any) -> None:
    document = raw_pipeline()
    if input_kind in ("blend", "mask"):
        document["software"][0]["params"]["source"] = "preview"
        document["software"].insert(
            1,
            node(
                "software",
                "combine",
                tab="raw",
                mode="opacity",
                opacity=0.75,
                mask_source="raw" if input_kind == "mask" else "none",
            ),
        )
    elif input_kind == "branch":
        document["software"][0]["params"]["source"] = "preview"
        document["branches"]["B"] = [node("software", "source", source="raw")]
        document["software"].insert(1, node("software", "combine", tab="B", mode="opacity"))
    frame, _ = frame_with_preview()
    words = np.frombuffer(frame, dtype="<u2").copy()
    raw = words[HEADER_U16 : HEADER_U16 + SENSOR_PIXELS].reshape(192, 256)
    raw[:] = (20 + 50) * 64
    raw[:, :64] = (45 + 50) * 64
    frame = words.tobytes()
    averaged = np.full(raw.shape, (30 + 50) * 64, np.float32)
    snapshot = averaged.copy()
    processor = PipelineProcessor()
    try:
        instantaneous, _ = processor.process(frame, None, document, scale=1)
        with_average, _ = processor.process(frame, averaged, document, scale=1)
        assert np.array_equal(with_average, instantaneous)
        assert np.array_equal(averaged, snapshot)
    finally:
        processor.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(version=5),
        lambda d: d["software"].append(node("software", "source")),
        lambda d: d["software"].insert(0, node("software", "gamma")),
        lambda d: d["software"][0].update(bypass=True),
        lambda d: d["software"].append({**node("software", "gamma"), "type": "unknown"}),
        lambda d: d["software"].append(node("software", "gamma", amount=float("nan"))),
        lambda d: d["software"].append(node("software", "range", low=60.0, high=10.0)),
        lambda d: d["software"].append(node("software", "enhance", passes=1.2)),
        lambda d: d["software"].append(deepcopy(d["software"][0])),
        lambda d: d["hardware"].extend(
            [node("hardware", "brightness"), node("hardware", "brightness", value=20)]
        ),
        lambda d: d["hardware"].append(node("hardware", "detail", enabled=False, fixed=True)),
        lambda d: d["hardware"].append(node("hardware", "humidity", value=10.123)),
    ],
)
def test_reject_invalid_pipeline_atomically(mutation: Any) -> None:
    document = default_pipeline()
    mutation(document)
    with pytest.raises((ValueError, TypeError)):
        validate_pipeline(document)


def test_repeat_nodes_and_portable_preferences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = raw_pipeline(node("software", "gamma"), node("software", "gamma", amount=2.0))
    document["software"][1].update(bypass=True, expanded=True)
    valid = validate_pipeline(document)
    assert valid == document and valid is not document
    save_settings({"pipeline": document, "capture_cursor": True})
    assert load_settings()["pipeline"] == document
    assert load_settings()["capture_cursor"] is True


def test_pipeline_order_and_repetition_have_visible_effects() -> None:
    bright = node("software", "brightness", amount=20.0)
    gamma = node("software", "gamma", amount=2.0)
    first = process(raw_pipeline(bright, gamma))
    second = process(raw_pipeline(gamma, bright))
    assert not np.array_equal(first, second)
    once = process(raw_pipeline(gamma))
    twice = process(raw_pipeline(gamma, node("software", "gamma", amount=2.0)))
    assert not np.array_equal(once, twice)
    gamma["bypass"] = True
    assert np.array_equal(process(raw_pipeline(gamma)), process(raw_pipeline()))


def test_range_after_processing_preserves_previous_operations() -> None:
    narrow = node("software", "range", low=15.0, high=55.0)
    boosted = process(raw_pipeline(node("software", "brightness", amount=10.0), narrow))
    original = process(raw_pipeline(narrow))
    assert not np.array_equal(boosted, original)
    # First range defines the source normalization, even with a filter before it.
    before = raw_pipeline(node("software", "filter", filter="gaussian", amount=1.0))
    after = deepcopy(before)
    after["software"][1:-1] = after["software"][1:-1][::-1]
    assert np.array_equal(process(before), process(after))


def test_camera_preview_range_recolors_thermal_and_keeps_previous_filters() -> None:
    document = default_pipeline()
    document["hardware"].append(node("hardware", "camera_colors", palette=11))
    document["software"] = [
        document["software"][0],
        node("software", "brightness", amount=20.0),
        node("software", "range", low=10.0, high=60.0),
        node("software", "output"),
    ]
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    processor = PipelineProcessor()
    enhanced, source = processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert source == "raw" and thermal_source(document)
    document["software"] = [document["software"][0], *document["software"][2:]]
    plain, _ = processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert not np.array_equal(enhanced, plain)
    document["software"] = [document["software"][0], document["software"][-1]]
    _, source = processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert source == "preview"


def test_macos_unverified_preview_falls_back_to_radiometric_image() -> None:
    frame, _ = frame_with_preview(preview_scale=1)
    image, source = PipelineProcessor().process(frame, None, default_pipeline(), scale=1)
    assert source == "raw"
    assert image.shape == (192, 256, 3)


def test_tolerant_capture_keeps_full_processed_camera_preview() -> None:
    frame, _ = frame_with_preview(preview_scale=2)
    # Retain real chroma, so a raw or grayscale fallback cannot pass this check.
    packed = np.frombuffer(frame, np.uint8, offset=IMAGE_OFFSET * 2).copy()
    packed[1::4] = 90
    packed[3::4] = 170
    frame = frame[: IMAGE_OFFSET * 2] + packed.tobytes()
    assembler = FrameAssembler()
    captured = None
    for offset in range(0, len(frame), 16266):
        captured = assembler.feed(b"\x02\x80" + frame[offset : offset + 16266]) or captured
    captured = assembler.feed(b"\x02\x82") or captured
    assert captured == frame
    document = default_pipeline()
    document["software"] = [document["software"][0], document["software"][-1]]
    processor = PipelineProcessor()
    try:
        image, source = processor.process(captured, None, document, scale=2)
    finally:
        processor.close()
    expected = cv2.cvtColor(packed.reshape(384, 512, 2), cv2.COLOR_YUV2BGR_YUY2)
    assert source == "preview"
    assert np.array_equal(image, expected)


def test_filters_interpolation_aa_are_real_cumulative_operations() -> None:
    document = raw_pipeline(
        node("software", "interpolation", method="nearest", scale=2),
        node("software", "antialiasing", amount=1.0),
    )
    once = process(document, 3)
    document["software"].insert(-1, node("software", "antialiasing", amount=1.0))
    twice = process(document, 3)
    assert not np.array_equal(once, twice)
    document["software"][2]["params"]["method"] = "lanczos"
    assert not np.array_equal(twice, process(document, 3))


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
def test_pipeline_mirrors_match_sensor_coordinates_after_final_rotation(rotation: Any) -> None:
    frame, _ = frame_with_preview()
    document = raw_pipeline(node("software", "mirror", horizontal=True))
    renderer = ThermalRenderer(scale=1, rotation=rotation)
    renderer.set_pipeline(document)
    rendered = renderer.render_detailed(frame)
    _, raw, _ = decode_duo_frame(frame)
    assert np.array_equal(rendered.raw_counts, np.rot90(np.fliplr(raw), -(rotation // 90)))
    assert (renderer.mirror_horizontal, renderer.mirror_vertical) == geometry(document, rotation)
    document["software"].insert(-1, node("software", "mirror", horizontal=True))
    renderer.set_pipeline(document)
    assert np.array_equal(
        renderer.render_detailed(frame).raw_counts, np.rot90(raw, -(rotation // 90))
    )


def test_display_operations_never_modify_temperature_statistics() -> None:
    frame, _ = frame_with_preview()
    baseline = ThermalRenderer(scale=1).render_detailed(frame)
    document = raw_pipeline(
        node("software", "contrast", amount=2.0),
        node("software", "colors", palette="turbo"),
        node("software", "enhance", model="anime4k09", passes=1),
    )
    renderer = ThermalRenderer(scale=1)
    renderer.set_pipeline(document)
    enhanced = renderer.render_detailed(frame)
    assert enhanced.stats == baseline.stats
    assert np.array_equal(enhanced.raw_counts, baseline.raw_counts)
    assert np.array_equal(enhanced.temperatures_celsius, baseline.temperatures_celsius)


def test_cap_precedes_expensive_allocation() -> None:
    document = raw_pipeline(*[node("software", "interpolation", scale=4) for _ in range(3)])
    with pytest.raises(ValueError, match="4 megapixel"):
        process(document)


@pytest.mark.parametrize("denoise", range(4))
def test_acnet_default_runs_on_full_camera_preview(denoise: Any) -> None:
    frame, _ = frame_with_preview(preview_scale=2)
    document = default_pipeline()
    document["software"].insert(-1, node("software", "enhance", model="acnet", denoise=denoise))
    processor = PipelineProcessor()
    try:
        image, source = processor.process(frame, None, document, scale=2)
        assert image.shape == (384, 512, 3)
        assert source == "preview"
    finally:
        processor.close()


def test_acnet_excess_passes_are_clamped_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    frame, _ = frame_with_preview(preview_scale=2)
    document = default_pipeline()
    document["software"].insert(-1, node("software", "enhance", model="acnet", passes=3))
    calls = []

    def upscale(self, image: Any, model: Any, amount: Any, passes: Any) -> Any:
        """
        Upscale.
        """
        calls.append(image.shape[:2])
        return cv2.resize(image, None, fx=2, fy=2)

    monkeypatch.setattr("topdon_duo.processing.branch.VisionUpsampler.apply", upscale)
    processor = PipelineProcessor()
    try:
        image, _ = processor.process(frame, None, document)
        assert image.shape == (576, 768, 3)
        assert calls == [(384, 512), (768, 1024)]
    finally:
        processor.close()


def test_fixed_dependency_and_raw_preview_exclusion() -> None:
    document = default_pipeline()
    document["hardware"].append(node("hardware", "detail", enabled=True, fixed=True))
    controller = PipelineHardware(fake_hardware())
    controller.apply(document)
    assert controller.hardware.fixed_range
    document["software"].insert(-1, node("software", "range"))
    controls, _, _, _, fixed = desired_hardware(document)
    assert not controls and not fixed
    controller.apply(document)
    assert not controller.hardware.fixed_range
    document["software"] = [document["software"][0], document["software"][-1]]
    controller.apply(document)
    assert controller.hardware.fixed_range
    document["hardware"].append(node("hardware", "gamma", value=30))
    with pytest.raises(ValueError, match="Fixed detail"):
        validate_pipeline(document)


def test_migration_retains_analyze_settings_and_old_mirror_geometry() -> None:
    document = migrate_pipeline(
        {
            "rotation": 90,
            "display": {
                "analyze_mode": True,
                "mirror_horizontal": True,
                "raw_temperature_low": 20.0,
                "raw_temperature_high": 55.0,
                "raw_upsampling": "anime4k09",
                "raw_sharpen_amount": 0.6,
                "raw_palette": "plasma",
            },
            "hardware": {"palette": 11.0, "ambient": 22.0, "emissivity": 0.95},
        }
    )
    assert geometry(document, 90) == (True, False)
    assert thermal_source(document)
    assert [n["type"] for n in document["software"]] == [
        "source",
        "range",
        "filter",
        "enhance",
        "colors",
        "mirror",
        "antialiasing",
        "output",
    ]
    assert not any(n["type"] in ("ambient", "emissivity") for n in document["hardware"])
    assert next(n for n in document["software"] if n["type"] == "range")["params"] == {
        "low": 20.0,
        "high": 55.0,
    }


def test_worker_overwrites_pending_frames_and_rejects_stale_results() -> None:
    worker = PipelineWorker()
    started, release = Event(), Event()
    calls = []

    def slow(frame: Any, *_args: Any) -> Any:
        """
        Slow.
        """
        calls.append(frame)
        started.set()
        release.wait(2)
        return np.zeros((3, 3, 3), np.uint8), "raw"

    worker.processor.process = slow
    try:
        worker.submit(b"first", None, default_pipeline(), 1, 1, 0, 1)
        assert started.wait(1)
        before = monotonic()
        worker.submit(b"discard", None, default_pipeline(), 2, 1, 0, 1)
        worker.submit(b"newest", None, default_pipeline(), 3, 1, 0, 1)
        assert monotonic() - before < 0.1
        assert worker.latest(3) is None
        release.set()
        deadline = monotonic() + 2
        while worker.latest(3) is None and monotonic() < deadline:
            sleep(0.01)
        assert worker.latest(3) is not None
        assert calls == [b"first", b"newest"]
        assert worker.latest(1) is None
    finally:
        release.set()
        worker.close()


def test_legacy_preview_default_migrates_with_matching_colors() -> None:
    frame, _ = frame_with_preview()
    baseline = ThermalRenderer(scale=2).render_detailed(frame)
    renderer = ThermalRenderer(scale=2)
    renderer.set_pipeline(migrate_pipeline({}))
    migrated = renderer.render_detailed(frame)
    # Antialiasing now has an explicit smoothing pass; broad uniform areas retain colors.
    assert np.array_equal(migrated.image[30, 30], baseline.image[30, 30])
    assert np.array_equal(migrated.image[-30, -30], baseline.image[-30, -30])
    assert migrated.stats == baseline.stats


def test_preview_unknown_baseline_palette_reports_error_without_killing_worker() -> None:
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    document = default_pipeline()
    document["software"] = [
        document["software"][0],
        node("software", "range", low=10.0, high=60.0),
        document["software"][-1],
    ]
    worker = PipelineWorker()
    try:
        worker.submit(frame, raw.astype(np.float32), document, 1, 1, 0, 99)
        deadline = monotonic() + 2
        while worker.latest(1) is None and monotonic() < deadline:
            sleep(0.01)
        assert "Unknown baseline" in worker.latest(1)[-1]
        worker.submit(frame, raw.astype(np.float32), document, 2, 1, 0, 1)
        deadline = monotonic() + 2
        while worker.latest(2) is None and monotonic() < deadline:
            sleep(0.01)
        assert worker.latest(2)[1] is not None and not worker.latest(2)[-1]
    finally:
        worker.close()


def test_version_one_source_migrates_to_first_software_node_with_identity_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    current = raw_pipeline(node("software", "gamma", amount=2.0))
    current["hardware"].append(node("hardware", "brightness", value=65))
    source = current["software"][0]
    source["expanded"] = True
    legacy = deepcopy(current)
    legacy["version"] = 1
    legacy.pop("branches")
    legacy["software"].pop()
    legacy["hardware"].insert(0, legacy["software"].pop(0))
    upgraded = validate_pipeline(legacy)
    assert upgraded["hardware"] == current["hardware"]
    assert upgraded["software"][:-1] == current["software"][:-1]
    assert upgraded["software"][0] == source
    assert legacy["version"] == 1  # validation doesn't mutate the input file
    save_settings({"pipeline": legacy})
    assert load_settings()["pipeline"]["software"][:-1] == current["software"][:-1]
    assert np.array_equal(process(upgraded), process(current))


@pytest.mark.parametrize("legacy,canonical", [("camera_1", "white_hot"), ("camera_2", "black_hot")])
def test_app_palette_cleanup_keeps_saved_choices_and_pixels(legacy: Any, canonical: Any) -> None:
    from topdon_duo.pipeline import CAMERA_GRADIENTS, PALETTES
    from topdon_duo.pipeline_processing import colorize

    selected = node("software", "colors", palette=legacy)
    saved = raw_pipeline(selected)
    before = deepcopy(saved)
    restored = validate_pipeline(saved)
    assert saved == before
    migrated = next(n for n in restored["software"] if n["id"] == selected["id"])
    assert migrated["params"]["palette"] == canonical
    assert legacy not in PALETTES
    assert list(PALETTES)[-len(CAMERA_GRADIENTS) :] == list(CAMERA_GRADIENTS)
    assert list(PALETTES.values()).count("White hot") == 1
    assert list(PALETTES.values()).count("Black hot") == 1
    assert all("approx" not in name for name in PALETTES.values())
    gray = np.arange(256, dtype=np.float32).reshape(16, 16)
    np.testing.assert_array_equal(colorize(gray, legacy), colorize(gray, canonical))


__all__ = ["fake_hardware", "process", "raw_pipeline"]
