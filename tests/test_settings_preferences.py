import json

from topdon_duo.settings_preferences import load_settings, save_settings


def test_invalid_preferences_preserve_independent_valid_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "topdon-duo" / "settings.json"
    assert load_settings() == {}
    path.parent.mkdir()
    for contents in ("{", "null", "[]"):
        path.write_text(contents)
        assert load_settings() == {}
    path.write_text(
        json.dumps(
            {
                "display": {"temperature_unit": "F", "image_source": "invalid", "antialiasing": 1},
                "hardware": {"ambient": 35.1, "distance": -1, "emissivity": True, "unknown": 2},
                "rotation": True,
                "advanced_auto": "yes",
            }
        )
    )
    assert load_settings() == {
        "display": {"temperature_unit": "F"},
        "hardware": {"ambient": 35.1},
    }


def test_preferences_round_trip_without_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    saved = {
        "display": {"color_palette": "inferno", "enhancement_amount": 0.5},
        "hardware": {"ambient": 27.5, "palette": 11, "detail_enabled": 0},
        "rotation": 270,
        "advanced_auto": False,
    }
    save_settings(saved)
    assert load_settings() == saved
    assert not list(tmp_path.rglob("*.tmp"))
