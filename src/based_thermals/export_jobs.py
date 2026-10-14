"""
Own converter subprocess groups across viewer shutdown and interpreter exit.
"""

import atexit
import os
import signal
import subprocess
from pathlib import Path
from threading import Event, Lock


class ExportJobs:
    """
    Serialize launches with cancellation and reap children before their owner exits.
    """

    def __init__(self) -> None:
        """
        Start with an uncancelled generation and no conversion processes.
        """
        self.lock = Lock()
        self.cancelled = Event()
        self.processes: dict[int, subprocess.Popen[str]] = {}

    def begin(self) -> None:
        """
        Permit a new session without reviving cancelled work from an older generation.
        """
        with self.lock:
            if self.cancelled.is_set():
                self.cancelled = Event()

    def token(self) -> Event:
        """
        Capture cancellation for both cache-lock waiting and subprocess execution.
        """
        with self.lock:
            return self.cancelled

    def launch(
        self,
        command: list[str],
        environment: dict[str, str],
        directory: Path,
        token: Event,
    ) -> subprocess.Popen[str]:
        """
        Register a dedicated process group atomically with respect to shutdown.
        """
        with self.lock:
            if token.is_set() or self.cancelled.is_set():
                raise ValueError("Model preparation cancelled")
            process = subprocess.Popen(
                command,
                env=environment,
                cwd=directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.processes[id(process)] = process
            return process

    def finished(self, process: subprocess.Popen[str]) -> None:
        """
        Remove a reaped child from shutdown ownership.
        """
        with self.lock:
            self.processes.pop(id(process), None)

    @staticmethod
    def terminate(process: subprocess.Popen[str]) -> None:
        """
        Stop the whole conversion group, including uv's Python child.
        """
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def close(self) -> None:
        """
        Cancel queued work, kill active groups and wait for process exit before returning.
        """
        with self.lock:
            self.cancelled.set()
            processes = list(self.processes.values())
        for process in processes:
            self.terminate(process)
        for process in processes:
            process.wait(timeout=5)
            self.finished(process)


EXPORT_JOBS = ExportJobs()
atexit.register(EXPORT_JOBS.close)
