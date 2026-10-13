"""
Identify desktop viewers before camera-opening diagnostic experiments.
"""

from itertools import pairwise
from pathlib import Path


def desktop_is_running(proc_root: Path = Path("/proc")) -> bool:
    """
    Recognize executable and Python module launches from Linux process command lines.
    """
    for process in proc_root.glob("[0-9]*/cmdline"):
        try:
            command = [arg.decode(errors="replace") for arg in process.read_bytes().split(b"\0")
                       if arg]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(arg).name == "based-thermals-desktop" for arg in command):
            return True
        if any(first == "-m" and second == "based_thermals.desktop"
               for first, second in pairwise(command)):
            return True
    return False
