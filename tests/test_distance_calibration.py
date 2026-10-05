import copy
from pathlib import Path

import numpy as np
import pytest

from topdon_duo.distance_calibration import (
    DistanceCalibrator,
    DistanceReference,
    detect_hand_square,
    square_edge_pixels,
)
from topdon_duo.settings_preferences import load_settings, save_settings

SIZE = (256, 192)


def square(edge=48):
    return [(64, 48), (64 + edge, 48), (64 + edge, 48 + edge), (64, 48 + edge)]


def thermal_square(edge=48, note=22, hand=32):
    thermal = np.full((192, 256), 20, np.float32)
    thermal[30:170, 40:210] = hand
    thermal[48 : 48 + edge, 64 : 64 + edge] = note
    return thermal


def select(calibrator, mode, edge=48):
    calibrator.begin(mode)
    for _ in range(calibrator.STABLE_FRAMES):
        calibrator.update(thermal_square(edge))


def test_known_square_reference_predicts_held_out_distances_and_orientation():
    reference = DistanceReference(0.076, 0.5, 48)
    for edge, expected in ((24, 1), (96, 0.25), (48, 0.5)):
        assert reference.estimate(square(edge), SIZE) == pytest.approx(expected)
    # In-plane rotation and reflection do not alter the square's projected size.
    points = np.asarray(square(24), float)
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = (points - points.mean(axis=0)) @ rotation + points.mean(axis=0)
    assert reference.estimate(rotated, SIZE) == pytest.approx(1)
    reflected = points.copy()
    reflected[:, 0] = 255 - reflected[:, 0]
    assert reference.estimate(reflected, SIZE) == pytest.approx(1)
    assert reference.estimate(points[:, ::-1], SIZE[::-1]) == pytest.approx(1)


@pytest.mark.parametrize(
    "corners",
    [
        square(11),
        [square()[i] for i in (0, 2, 1, 3)],
        [(64, 48), (112, 48), (112, 72), (64, 72)],
        [(64, 48), (112, 48), (124, 96), (76, 96)],
        [(64, 48), (112, 48), (112, 96), (float("nan"), 96)],
        [(250, 48), (298, 48), (298, 96), (250, 96)],
        [(0, 0), (0, 0), (0, 0), (0, 0)],
        square()[:3],
    ],
)
def test_invalid_square_rejected(corners):
    with pytest.raises(ValueError):
        square_edge_pixels(corners, SIZE)


def test_invalid_selection_and_cancel_preserve_saved_reference():
    calibrator = DistanceCalibrator()
    with pytest.raises(ValueError, match="reference"):
        calibrator.begin("measure")
    select(calibrator, "reference")
    calibrator.save_reference(0.076, 0.5)
    reference = calibrator.reference
    select(calibrator, "reference", 8)
    assert calibrator.reference == reference
    assert not calibrator.state()["ready"]
    with pytest.raises(ValueError):
        calibrator.save_reference(0.076, 1)
    select(calibrator, "measure", 24)
    calibrator.cancel()
    assert calibrator.reference == reference
    assert calibrator.estimated_m is None and not calibrator.corners
    calibrator.clear()
    assert calibrator.reference is None


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 48),
        (0.076, 0.25, 48),
        (0.076, 100, 48),
        (0.076, 1, 8),
        (True, 1, 48),
        (0.076, float("inf"), 48),
    ],
)
def test_invalid_reference_rejected(values):
    with pytest.raises(ValueError):
        DistanceReference(*values)


def test_detector_finds_note_despite_gradient_noise_rotation_and_mirroring():
    thermal = thermal_square()
    thermal += np.linspace(0, 1, 256)[None, :]
    thermal += np.random.default_rng(17).normal(0, 0.15, thermal.shape)
    for plane in (thermal, np.rot90(thermal), np.fliplr(thermal)):
        detected = detect_hand_square(plane)
        assert detected is not None
        assert square_edge_pixels(detected, plane.shape[::-1]) == pytest.approx(48, abs=1)


def test_captured_post_it_on_hand_recovers_unevenly_warmed_outline():
    with np.load(Path(__file__).parent / "fixtures" / "post-it-hand-reference.npz") as sample:
        thermal = sample["temperatures_celsius"]
    for plane in (thermal, np.rot90(thermal), np.fliplr(thermal)):
        detected = detect_hand_square(plane)
        assert detected is not None
        assert square_edge_pixels(detected, plane.shape[::-1]) == pytest.approx(51, abs=2)
    calibrator = DistanceCalibrator()
    calibrator.begin("reference")
    for _ in range(calibrator.STABLE_FRAMES):
        calibrator.update(thermal)
    assert calibrator.state()["ready"]
    center = np.asarray(calibrator.corners).mean(axis=0)
    assert center == pytest.approx((105, 79), abs=2)


@pytest.mark.parametrize(
    "scene", ["no_hand", "equal_temperature", "hot_note", "rectangle", "two_notes", "nonfinite"]
)
def test_detector_rejects_missing_hand_weak_contrast_and_ambiguous_targets(scene):
    thermal = thermal_square()
    if scene == "no_hand":
        thermal = thermal_square(hand=20)
    elif scene == "equal_temperature":
        thermal = thermal_square(note=31.8)
    elif scene == "hot_note":
        thermal = thermal_square(note=38)
    elif scene == "rectangle":
        thermal[48:96, 64:112] = 32
        thermal[48:76, 64:112] = 22
    elif scene == "two_notes":
        thermal[100:148, 144:192] = 22
    else:
        thermal[0, 0] = np.nan
    assert detect_hand_square(thermal) is None


def test_stable_automatic_capture_measurement_loss_and_motion():
    calibrator = DistanceCalibrator()
    calibrator.begin("reference")
    for _ in range(4):
        assert not calibrator.update(thermal_square())
    assert calibrator.state()["stable_frames"] == 4
    calibrator.update(np.full((192, 256), 20, np.float32))
    assert not calibrator.corners and calibrator.state()["stable_frames"] == 0
    for _ in range(4):
        calibrator.update(thermal_square())
    calibrator.update(np.roll(thermal_square(), 4, axis=1))
    assert calibrator.state()["stable_frames"] == 1
    select(calibrator, "reference")
    assert calibrator.state()["ready"]
    calibrator.save_reference(0.076, 0.5)
    select(calibrator, "measure", 24)
    assert calibrator.estimated_m == pytest.approx(1, rel=0.03)
    captured = calibrator.corners.copy()
    assert not calibrator.update(thermal_square(96))
    assert calibrator.corners == captured


def test_reference_persists_and_corrupt_profiles_do_not_break_other_preferences(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    profile = DistanceReference(0.076, 0.5, 48).as_dict()
    save_settings({"distance_calibration": profile, "display": {"temperature_unit": "F"}})
    loaded = load_settings()
    restored = DistanceCalibrator(loaded["distance_calibration"])
    assert restored.reference.estimate(square(24), SIZE) == pytest.approx(1)
    assert restored.selecting is None and restored.estimated_m is None
    for field, value in (
        ("version", True),
        ("native_size", [192, 256]),
        ("distance_m", -1),
        ("edge_pixels", "48"),
    ):
        invalid = copy.deepcopy(profile)
        invalid[field] = value
        save_settings({"distance_calibration": invalid, "display": {"temperature_unit": "F"}})
        assert load_settings() == {"display": {"temperature_unit": "F"}, "hardware": {}}
