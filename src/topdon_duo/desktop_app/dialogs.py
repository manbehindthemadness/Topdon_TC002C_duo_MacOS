"""
Nonblocking native save dialogs on macOS and Linux.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from ..dialog_preferences import load_dialog_directory, remember_dialog_directory
from .constants import (
    LOG,
    SAVE_DIALOG_SCRIPT,
)


class MacSaveDialog:
    """Non-blocking macOS save panel so USB capture continues behind it."""

    def __init__(self) -> None:
        """
        Init.
        """
        self._process: subprocess.Popen[str] | None = None
        self._directory_kind = "image"

    @property
    def is_open(self) -> bool:
        """
        Is open.
        """
        return self._process is not None

    def open(
        self, default_directory: Path | None = None, *, suffix: str = ".png", kind: str = ""
    ) -> bool:
        """
        Open.
        """
        if self._process is not None:
            return False
        prefix = f"TC002C-Duo-{kind}-" if kind else "TC002C-Duo-"
        default_name = prefix + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + suffix
        self._directory_kind = (
            kind
            if kind in ("video", "timelapse", "temperatures")
            else "video"
            if suffix.lower() == ".mp4"
            else "temperatures"
            if suffix.lower() == ".csv"
            else "image"
        )
        remembered = load_dialog_directory(self._directory_kind, default_directory)
        directory = str(remembered) if remembered else ""
        command = self._command(default_name, directory)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True

    def _command(self, default_name: str, directory: str) -> list[str]:
        """
        Command.
        """
        script = SAVE_DIALOG_SCRIPT
        if Path(default_name).suffix.lower() == ".mp4":
            script = script.replace("Save thermal capture", "Save thermal recording")
        elif Path(default_name).suffix.lower() == ".csv":
            script = script.replace("Save thermal capture", "Save temperature log")
        return [
            "/usr/bin/osascript",
            "-e",
            script,
            "--",
            default_name,
            directory,
        ]

    def _cancelled(self, returncode: int, stderr: str) -> bool:
        """
        Cancelled.
        """
        return "User canceled" in stderr

    def poll(self) -> tuple[bool, Path | None]:
        """
        Return (finished, selected path); cancellation yields (True, None).
        """
        if self._process is None or self._process.poll() is None:
            return False, None
        process, self._process = self._process, None
        stdout, stderr = process.communicate()
        if process.returncode == 0 and stdout.strip():
            selected = Path(stdout.strip())
            remember_dialog_directory(self._directory_kind, selected)
            return True, selected
        if not self._cancelled(process.returncode, stderr):
            LOG.error("Save dialog failed: %s", stderr.strip() or process.returncode)
        return True, None

    def close(self) -> None:
        """
        Close.
        """
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None


class LinuxSaveDialog(MacSaveDialog):
    """Non-blocking GTK save panel supplied by Ubuntu's zenity package."""

    def _cancelled(self, returncode: int, stderr: str) -> bool:
        """
        Cancelled.
        """
        return returncode == 1

    def _command(self, default_name: str, directory: str) -> list[str]:
        """
        Command.
        """
        video = Path(default_name).suffix.lower() == ".mp4"
        csv_log = Path(default_name).suffix.lower() == ".csv"
        return [
            "zenity",
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title=Save temperature log"
            if csv_log
            else "--title=Save thermal recording"
            if video
            else "--title=Save thermal capture",
            f"--filename={Path(directory) / default_name}",
            "--file-filter=CSV logs | *.csv"
            if csv_log
            else "--file-filter=MP4 videos | *.mp4"
            if video
            else "--file-filter=PNG images | *.png",
        ]
