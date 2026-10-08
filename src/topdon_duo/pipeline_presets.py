"""Bundled pipeline presets plus user-created overrides beside preferences."""

import json
from pathlib import Path

from .pipeline import validate_pipeline
from .window_preferences import _path


def _load_presets(path):
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
        try:
            presets[name] = validate_pipeline(document)
        except (ValueError, TypeError, KeyError):
            continue
    return presets


def predefined_presets():
    return _load_presets(Path(__file__).with_name("predefined_pipeline_presets.json"))


def load_presets():
    return {
        **predefined_presets(),
        **_load_presets(_path().with_name("pipeline-presets.json")),
    }


def save_presets(presets):
    validated = {name: validate_pipeline(document) for name, document in presets.items()}
    predefined = predefined_presets()
    # Do not copy unmodified bundled presets into user preferences. An explicit
    # replacement remains a user override and never changes the shipped preset.
    validated = {name: doc for name, doc in validated.items() if doc != predefined.get(name)}
    path = _path().with_name("pipeline-presets.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(validated, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
