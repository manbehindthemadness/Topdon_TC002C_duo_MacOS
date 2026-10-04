"""Main viewer size preferences without loading a second Qt binding."""

import json
import os
import sys
from pathlib import Path


def _path() -> Path:
    config = os.environ.get("XDG_CONFIG_HOME")
    if config:
        base = Path(config)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".config"
    return base / "topdon-duo" / "main-window.json"


def load_main_window_size() -> tuple[int, int] | None:
    try:
        value = json.loads(_path().read_text())
    except (OSError, ValueError):
        return None
    if isinstance(value, list) and len(value) == 2 and all(type(n) is int and n > 0 for n in value):
        return tuple(value)
    return None


def save_main_window_size(size: tuple[int, int]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(size) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
