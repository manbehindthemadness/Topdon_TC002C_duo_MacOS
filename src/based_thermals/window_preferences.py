"""Main viewer size preferences without loading a second Qt binding."""

import json
import os
import sys
from pathlib import Path

from .preference_io import save_json


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
    """
    Restore a positive integer window size, ignoring malformed preferences.
    """
    try:
        value = json.loads(_path().read_text())
    except (OSError, ValueError):
        return None
    if isinstance(value, list) and len(value) == 2 and all(type(n) is int and n > 0 for n in value):
        size = (value[0], value[1])
        return size
    return None


def save_main_window_size(size: tuple[int, int]) -> None:
    """
    Atomically persist the main viewer dimensions.
    """
    path = _path()
    save_json(path, size)
