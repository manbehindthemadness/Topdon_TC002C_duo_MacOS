"""
Prevent diagnostic camera acquisition alongside renamed desktop launch commands.
"""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("command,running", [
    (["/venv/bin/based-thermals-desktop", "--rotate", "90"], True),
    (["/venv/bin/python", "/venv/bin/based-thermals-desktop"], True),
    (["/venv/bin/python", "-m", "based_thermals.desktop"], True),
    (["python", "-m", "based_thermals.desktop", "--camera", "duo"], True),
    (["python", "-m", "based_thermals.runtime_setup"], False),
    (["python", "-c", "based_thermals.desktop"], False),
    (["based-thermals-desktop-helper"], False),
    (["based-thermals-models"], False),
    ([], False),
])
def test_desktop_process_detection(command: list[str], running: bool, tmp_path: Path) -> None:
    """
    Inspect fake process files without opening cameras or reading host processes.
    """
    path = Path(__file__).resolve().parents[1] / "experiments" / "capture_processes.py"
    spec = importlib.util.spec_from_file_location("capture_processes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in command))
    (tmp_path / "456").mkdir()
    assert module.desktop_is_running(tmp_path) is running
