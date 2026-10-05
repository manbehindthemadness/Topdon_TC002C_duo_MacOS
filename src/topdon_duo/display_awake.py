"""Temporary display-idle inhibition without changing system power preferences."""

import logging
import os
import shutil
import subprocess
import sys

LOG = logging.getLogger(__name__)
REASON = "Keep the thermal camera viewer responsive while capturing measurements"


class DisplayAwake:
    def __init__(self):
        self._process = None
        self._waits_for_eof = False
        self._startup_error = None

    def start(self):
        if self._process is not None:
            return
        if sys.platform == "darwin":
            command = ["/usr/bin/caffeinate", "-d", "-w", str(os.getpid())]
        elif sys.platform.startswith("linux"):
            self._waits_for_eof = True
            gnome = shutil.which("gnome-session-inhibit")
            command = [sys.executable, "-m", "topdon_duo.display_awake_helper"]
            if gnome and "GNOME" in os.environ.get("XDG_CURRENT_DESKTOP", "").upper():
                command.extend(["--gnome", gnome])
        else:
            return
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except OSError as exc:
            self._startup_error = f"Could not keep the display awake: {exc}"
            LOG.warning("%s", self._startup_error)

    def check(self):
        if self._startup_error is not None:
            error, self._startup_error = self._startup_error, None
            return error
        if self._process is None or self._process.poll() is None:
            return None
        process, self._process = self._process, None
        _, stderr = process.communicate()
        message = "Could not keep the display awake"
        detail = stderr.decode(errors="replace").strip()
        if detail:
            message += f": {detail[:300]}"
        LOG.warning("%s", message)
        return message

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if not self._waits_for_eof and process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        if process.stderr is not None:
            process.stderr.close()
