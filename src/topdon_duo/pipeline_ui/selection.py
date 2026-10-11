"""
Persist a preset's association with the independently restored working pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from ..preference_io import save_json
from ..window_preferences import _path
from .presets import pipeline_content

if TYPE_CHECKING:
    from .editor import PipelineEditor

LOG = logging.getLogger(__name__)


def fingerprint(document: dict[str, Any]) -> str:
    """
    Identify processing settings without storing embedded package/model data a second time.
    """
    text = json.dumps(pipeline_content(document), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    return digest


class PresetSelection:
    """
    Restore once after the first viewer state, and write only when the association changes.
    """

    def __init__(self) -> None:
        """
        Load metadata without changing the editor's temporary startup document.
        """
        self.path = _path().with_name("pipeline-selection.json")
        self.pending = True
        self.saved: dict[str, Any] | None = None
        try:
            value = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if (
            isinstance(value, dict)
            and "name" in value
            and (value.get("name") is None or isinstance(value.get("name"), str))
            and type(value.get("recognize")) is bool
            and isinstance(value.get("fingerprint"), str)
        ):
            self.saved = value

    def restore(self, editor: PipelineEditor) -> None:
        """
        Recover the loaded name only for the working pipeline that was saved with it.
        """
        if not self.pending:
            return
        self.pending = False
        if self.saved is None or self.saved["fingerprint"] != fingerprint(editor.document):
            return
        name = self.saved["name"]
        if name is None or name in editor.presets:
            editor.current_preset_name = name
            editor.recognize_preset = self.saved["recognize"]

    def remember(self, editor: PipelineEditor, content: dict[str, Any]) -> None:
        """
        Save selection changes atomically; unavailable preference storage leaves editing usable.
        """
        if self.pending:
            return
        text = json.dumps(content, sort_keys=True, separators=(",", ":"))
        value = {
            "name": editor.current_preset_name,
            "recognize": editor.recognize_preset,
            "fingerprint": hashlib.sha256(text.encode()).hexdigest(),
        }
        if value == self.saved:
            return
        try:
            save_json(self.path, value)
        except OSError as exc:
            LOG.warning("Could not save pipeline preset selection: %s", exc)
            return
        self.saved = value
