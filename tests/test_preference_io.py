"""
Preference saves preserve complete documents across failures and overlapping writes.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from topdon_duo.preference_io import save_json
from topdon_duo.settings_preferences import load_settings, save_settings


def test_overlapping_saves_use_independent_temporary_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Finish a second save between the first save's write and atomic replacement.
    """
    target = tmp_path / "preferences.json"
    replace = cast(Callable[[Path, Path], Path], Path.replace)
    sources: list[Path] = []

    def overlapping_replace(source: Path, destination: Path) -> Path:
        """
        Interleave two complete writes without threads or timing assumptions.
        """
        sources.append(source)
        if len(sources) == 1:
            save_json(target, {"writer": "second"})
            assert json.loads(target.read_text()) == {"writer": "second"}
        result = replace(source, destination)
        return result

    monkeypatch.setattr(Path, "replace", overlapping_replace)
    save_json(target, {"writer": "first"})
    assert len(set(sources)) == 2
    assert json.loads(target.read_text()) == {"writer": "first"}
    assert list(tmp_path.iterdir()) == [target]


def test_failed_replacement_preserves_preferences_and_cleans_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    A filesystem failure never truncates the previously published document.
    """
    target = tmp_path / "preferences.json"
    save_json(target, {"saved": True})

    def failed_replace(_source: Path, _destination: Path) -> Path:
        """
        Simulate an unavailable replacement at the filesystem boundary.
        """
        raise OSError("replacement unavailable")

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="replacement unavailable"):
        save_json(target, {"saved": False})
    assert json.loads(target.read_text()) == {"saved": True}
    assert list(tmp_path.iterdir()) == [target]


def test_invalid_json_values_preserve_existing_document(tmp_path: Path) -> None:
    """
    Reject non-finite numbers before creating a temporary file.
    """
    target = tmp_path / "preferences.json"
    save_json(target, {"saved": True})
    with pytest.raises(ValueError):
        save_json(target, {"invalid": float("nan")})
    assert json.loads(target.read_text()) == {"saved": True}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("section", ["ambient_input_celsius", "hardware", "display"])
def test_oversized_preference_numbers_preserve_independent_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, section: str,
) -> None:
    """
    Ignore numeric overflow in external JSON instead of aborting viewer startup.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    huge = 10 ** 400
    values = {
        "ambient_input_celsius": huge,
        "hardware": {"ambient": huge, "humidity": 50},
        "display": {"enhancement_amount": huge, "temperature_unit": "F"},
    }
    save_settings({section: values[section], "show_graph": True})
    loaded = load_settings()
    assert loaded["show_graph"] is True
    assert "ambient_input_celsius" not in loaded
    assert loaded["hardware"] == ({"humidity": 50} if section == "hardware" else {})
    assert loaded["display"] == ({"temperature_unit": "F"} if section == "display" else {})
