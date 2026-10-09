"""
Isolated Qt subprocesses with temporary preset storage and file-dialog output.
"""

import os
import subprocess
import sys
from pathlib import Path


def popup_environment() -> dict[str, str]:
    """
    Select offscreen Qt 6 without inheriting OpenCV's Qt plugin settings.
    """
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    environment.pop("QT_QPA_FONTDIR", None)
    return environment


def run_popup(script: str, directory: Path) -> subprocess.CompletedProcess[str]:
    """
    Run an offscreen popup script without reading or writing the user's presets.
    """
    environment = popup_environment()
    environment["XDG_CONFIG_HOME"] = str(directory / "config")
    command = [sys.executable, "-c", script, str(directory)]
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=20, check=False
    )
    return result
