"""Bundled pipeline presets plus user-created overrides beside preferences."""

import json
from pathlib import Path

from .pipeline import validate_pipeline
from .preference_io import save_json
from .window_preferences import _path


def _load_presets(path: Path) -> dict[str, dict | None]:
    """
    Read valid documents and explicit deletions, ignoring malformed entries.
    """
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    presets = {}
    for name, document in saved.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if document is None:
            presets[name] = None
            continue
        try:
            presets[name] = validate_pipeline(document)
        except (ValueError, TypeError, KeyError):
            continue
    return presets


def predefined_presets() -> dict[str, dict]:
    """
    Load shipped presets without user overrides or deletions.
    """
    saved = _load_presets(Path(__file__).with_name("predefined_pipeline_presets.json"))
    return {name: document for name, document in saved.items() if document is not None}


def load_presets() -> dict[str, dict]:
    """
    Merge user overrides and hide presets explicitly deleted by the user.
    """
    saved = {
        **predefined_presets(),
        **_load_presets(_path().with_name("pipeline-presets.json")),
    }
    return {name: document for name, document in saved.items() if document is not None}


def save_presets(presets: dict[str, dict]) -> None:
    """
    Persist the complete visible collection, including deletions of bundled presets.
    """
    validated = {name: validate_pipeline(document) for name, document in presets.items()}
    predefined = predefined_presets()
    # Do not copy unmodified bundled presets into user preferences. An explicit
    # replacement remains a user override and never changes the shipped preset.
    validated = {name: doc for name, doc in validated.items() if doc != predefined.get(name)}
    validated.update({name: None for name in predefined if name not in presets})
    path = _path().with_name("pipeline-presets.json")
    save_json(path, validated, indent=2)
