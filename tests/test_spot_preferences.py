import copy

import pytest

from topdon_duo.desktop import SampleSpots
from topdon_duo.settings_preferences import load_settings, save_settings


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(
    "horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)]
)
def test_spot_round_trip_preserves_sensor_locations_numbers_and_metadata(
    rotation, horizontal, vertical
):
    spots = SampleSpots(pixels=[(0, 0), (255, 191), (80, 60)])
    spots.rename(3, "Motor")
    spots.clear(2)
    spots.set_enabled(3, False)
    spots.placing = True
    saved = spots.saved_state()
    restored = SampleSpots()
    restored.restore_saved_state(saved, rotation, horizontal, vertical)
    assert restored.saved_state(rotation, horizontal, vertical) == saved
    assert restored.numbers == [1, 3] and restored.next_number == 4
    assert restored.name(3) == "Motor" and restored.disabled == {3}
    # Loading in another orientation still points to the same physical pixels.
    canonical = SampleSpots()
    canonical.restore_saved_state(restored.saved_state(rotation, horizontal, vertical))
    assert canonical.pixels == spots.pixels


def test_invalid_spots_do_not_discard_independent_preferences(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    saved = SampleSpots(pixels=[(80, 60)]).saved_state()
    save_settings({"spots": saved, "rotation": 90})
    assert load_settings()["spots"] == saved
    for key, value in (
        ("x", -1),
        ("x", 256),
        ("y", 192),
        ("number", True),
        ("enabled", 1),
        ("name", "bad\nname"),
    ):
        invalid = copy.deepcopy(saved)
        invalid["items"][0][key] = value
        save_settings({"spots": invalid, "rotation": 90})
        assert load_settings() == {"display": {}, "hardware": {}, "rotation": 90}
    save_settings({"spots": {**saved, "version": True}, "rotation": 90})
    assert "spots" not in load_settings()
    for items in ([None], [saved["items"][0]] * 2):
        invalid = {**saved, "items": items}
        save_settings({"spots": invalid, "rotation": 90})
        assert "spots" not in load_settings()
