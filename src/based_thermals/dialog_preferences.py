"""Independent last-used directories for file dialogs across processes and launches."""

import json
import logging
from pathlib import Path

from .preference_io import save_json
from .window_preferences import _path

LOG = logging.getLogger(__name__)
KINDS = {"image", "video", "timelapse", "temperatures", "pipeline_import", "pipeline_export"}


def _directory_path(kind):
    if kind not in KINDS:
        raise ValueError("Unknown file dialog")
    return _path().parent / "dialog-directories" / f"{kind}.json"


def load_dialog_directory(kind, default_directory=None):
    # An explicit --output directory takes precedence over remembered locations.
    if default_directory is not None:
        directory = Path(default_directory).expanduser()
        if directory.is_dir():
            return directory.resolve()
    try:
        saved = json.loads(_directory_path(kind).read_text())
        if isinstance(saved, str):
            directory = Path(saved)
            if directory.is_absolute() and directory.is_dir():
                return directory
    except (OSError, ValueError):
        pass
    return None


def remember_dialog_directory(kind: str, selected_file: str | Path) -> None:
    """
    Remember an accepted dialog directory, logging filesystem failures without failing capture.
    """
    path = _directory_path(kind)
    try:
        directory = Path(selected_file).expanduser().resolve().parent
        if not directory.is_dir():
            return
        save_json(path, str(directory))
    except OSError as exc:
        LOG.warning("Could not remember %s dialog directory: %s", kind, exc)
