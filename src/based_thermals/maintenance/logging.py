"""
Persistent support logging with disk writes isolated from the acquisition thread.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Queue

from ..preference_io import save_json
from ..window_preferences import _path


def debug_enabled() -> bool:
    """
    Read the global debug switch, ignoring malformed local preferences.
    """
    try:
        value = json.loads(_path().with_name("maintenance.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(value, dict) and value.get("debug_logs") is True


def save_debug(enabled: bool) -> None:
    """
    Persist the switch independently of per-camera preferences.
    """
    save_json(_path().with_name("maintenance.json"), {"debug_logs": enabled})


class SupportLogging:
    """
    Retain ordinary errors and optionally DEBUG records in a rotating local log.
    """

    def __init__(self, directory: Path) -> None:
        """
        Prepare passive state without creating files until the viewer runs.
        """
        self.directory = directory
        self.handler: QueueHandler | None = None
        self.listener: QueueListener | None = None
        self.output: RotatingFileHandler | None = None
        self.previous_level = logging.getLogger().level

    def start(self, debug: bool) -> None:
        """
        Start queued file logging; retain at most four 4 MB application log files.
        """
        if self.handler is not None:
            self.set_debug(debug)
            return
        self.previous_level = logging.getLogger().level
        self.directory.mkdir(parents=True, exist_ok=True)
        output = RotatingFileHandler(
            self.directory / "application.log", maxBytes=4_000_000, backupCount=3, encoding="utf-8"
        )
        output.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s")
        )
        handler = QueueHandler(Queue())
        listener = QueueListener(handler.queue, output, respect_handler_level=True)
        self.output, self.handler, self.listener = output, handler, listener
        logging.getLogger().addHandler(handler)
        self.set_debug(debug)
        listener.start()

    def set_debug(self, enabled: bool) -> None:
        """
        Apply verbosity to application records without changing unrelated handler settings.
        """
        level = logging.DEBUG if enabled else logging.INFO
        if self.handler is not None:
            self.handler.setLevel(level)
        logging.getLogger().setLevel(min(self.previous_level or logging.INFO, level))

    def flush(self) -> None:
        """
        Drain queued records before taking a file snapshot, then resume logging.
        """
        if self.listener is not None:
            self.listener.stop()
            if self.output is not None:
                self.output.flush()
            self.listener.start()

    def close(self) -> None:
        """
        Drain and detach support logging before reset or process exit.
        """
        if self.handler is not None:
            logging.getLogger().removeHandler(self.handler)
        if self.listener is not None:
            self.listener.stop()
        if self.output is not None:
            self.output.close()
        logging.getLogger().setLevel(self.previous_level)
        self.handler = self.listener = self.output = None
