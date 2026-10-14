"""
Verify converter ownership without downloading models or launching uv.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from based_thermals.export_jobs import ExportJobs


def test_shutdown_reaps_process_and_rejects_old_work(tmp_path: Path) -> None:
    """
    Kill active converters and keep old cancellation tokens invalid after restart.
    """
    jobs = ExportJobs()
    token = jobs.token()
    child = jobs.launch(
        [sys.executable, "-c", "import time; time.sleep(60)"], os.environ.copy(), tmp_path, token
    )
    try:
        jobs.close()
        assert child.poll() is not None
        assert token.is_set()
        jobs.begin()
        assert not jobs.token().is_set()
        with pytest.raises(ValueError, match="cancelled"):
            jobs.launch([sys.executable], {}, tmp_path, token)
    finally:
        jobs.close()
        child.communicate(timeout=5)


def test_interpreter_exit_stops_daemon_owned_converter(tmp_path: Path) -> None:
    """
    Exercise atexit cleanup with a real detached child started from a daemon thread.
    """
    script = """
import os, sys, threading
from pathlib import Path
from based_thermals.model_exports import run_export
from based_thermals.export_jobs import EXPORT_JOBS
started = threading.Event()
original = EXPORT_JOBS.launch
def launch(*args, **kwargs):
    process = original(*args, **kwargs)
    Path(sys.argv[1]).write_text(str(process.pid))
    started.set()
    return process
EXPORT_JOBS.launch = launch
threading.Thread(target=run_export,
    args=([sys.executable, "-c", "import time; time.sleep(60)"], os.environ.copy(),
          Path(sys.argv[1]).parent), daemon=True).start()
assert started.wait(5)
"""
    pid_file = tmp_path / "pid"
    parent = subprocess.run(
        [sys.executable, "-c", script, str(pid_file)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert parent.returncode == 0, parent.stderr
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
