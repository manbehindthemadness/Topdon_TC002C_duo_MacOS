import json

from topdon_duo.settings_preferences import load_settings, save_settings


def test_fixed_mode_persists_and_retired_bounds_are_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_settings({"fixed_range": True, "fixed_range_bounds": [4800, 5600]})
    assert load_settings()["fixed_range"] is True
    assert "fixed_range_bounds" not in load_settings()
    save_settings({"fixed_range": "yes"})
    assert "fixed_range" not in load_settings()


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
                "capture_cursor": "yes",
                "capture_graphs": 1,
                "timelapse_fpm": True,
            }
        )
    )
    assert load_settings() == {
        "display": {"temperature_unit": "F"},
        "hardware": {"ambient": 35.1},
    }


def test_retired_enhancements_fall_back_without_losing_other_settings(monkeypatch, tmp_path):
    from topdon_duo.render import ThermalRenderer

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for model in ("tidy", "dncnn-gray-blind"):
        save_settings({"display": {
            "upsampling": model,
            "tidy_model_path": "/usr/src/models/retired.onnx",
            "color_palette": "plasma",
        }})
        saved = load_settings()
        assert saved["display"] == {"color_palette": "plasma"}
        renderer = ThermalRenderer()
        for name, value in saved["display"].items():
            renderer.set_view_setting(name, value)
        assert renderer.upsampling == "off"
        assert renderer.color_palette == "plasma"


def test_preferences_round_trip_without_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    saved = {
        "display": {
            "color_palette": "inferno",
            "enhancement_amount": 0.5,
            "upsampling": "acnet-legacy-hdn1",
        },
        "hardware": {"ambient": 27.5, "palette": 11, "detail_enabled": 0},
        "rotation": 270,
        "advanced_auto": False,
        "auto_calibrate": True,
        "capture_cursor": True,
        "capture_graphs": False,
        "timelapse_fpm": 120,
        "graph_interval": 0.2,
    }
    save_settings(saved)
    assert load_settings() == saved
    assert not list(tmp_path.rglob("*.tmp"))


def test_invalid_emissivity_reference_does_not_discard_other_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for value in (
        None,
        {},
        {"version": 1, "emissivity": True, "known_celsius": 40},
        {"version": 1, "emissivity": 0.555, "known_celsius": 40},
    ):
        save_settings({"emissivity_calibration": value, "display": {"temperature_unit": "F"}})
        assert load_settings() == {"display": {"temperature_unit": "F"}, "hardware": {}}


def test_invalid_graph_interval_preserves_other_preferences(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for value in (True, "0.5", None, 0, 61, float("inf")):
        path = tmp_path / "topdon-duo" / "settings.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"graph_interval": value, "show_graph": True}))
        assert load_settings() == {"display": {}, "hardware": {}, "show_graph": True}


def test_invalid_timelapse_rate_preserves_valid_preferences(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for value in (True, "120", 0, -1, 1501, 120.5, None):
        save_settings({"timelapse_fpm": value, "capture_graphs": True})
        assert load_settings() == {"display": {}, "hardware": {}, "capture_graphs": True}


def test_processing_preset_preferences(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for preset in ("balanced", "shadow", "soft"):
        save_settings({"processing_preset": preset})
        assert load_settings()["processing_preset"] == preset
    for invalid in (None, 0, True, [], "manual", "Shadow"):
        save_settings({"processing_preset": invalid})
        assert "processing_preset" not in load_settings()


def test_tone_preferences_roundtrip_and_reject_invalid_values(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    save_settings({'camera_gamma': 25, 'camera_boost': True})
    assert load_settings()['camera_gamma'] == 25
    assert load_settings()['camera_boost'] == 3
    for invalid in (-1, 101, True, 25.5, '25'):
        save_settings({'camera_gamma': invalid, 'camera_boost': 'yes'})
        assert 'camera_gamma' not in load_settings()
        assert 'camera_boost' not in load_settings()


def test_boost_modes_persist_and_migrate_old_checkbox(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for value, expected in ((False, 0), (True, 3), (0, 0), (1, 1), (2, 2), (3, 3)):
        save_settings({"camera_boost": value})
        actual = load_settings()["camera_boost"]
        assert type(actual) is int and actual == expected
    for invalid in (-1, 4, 1.0, "1", None):
        save_settings({"camera_boost": invalid})
        assert "camera_boost" not in load_settings()
