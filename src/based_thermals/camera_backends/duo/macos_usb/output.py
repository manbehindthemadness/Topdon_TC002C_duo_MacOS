"""
Deliver only the newest complete frame without blocking USB drainage on viewer reads.
"""

import threading
from collections.abc import Callable
from typing import Any


class LatestOutput:
    """
    Own one pending frame and diagnostics snapshot behind a pipe-writing thread.
    """

    def __init__(self, publish: Callable[[dict, bytes], None]) -> None:
        """
        Bind serialized protocol delivery while acquisition only replaces pending snapshots.
        """
        self.publish = publish
        self.condition = threading.Condition()
        self.frame: bytes | None = None
        self.diagnostics: dict[str, Any] | None = None
        self.stopping = False
        self.thread = threading.Thread(target=self.run, name="usb-helper-output", daemon=True)
        self.thread.start()

    def submit_frame(self, frame: bytes) -> None:
        """
        Replace an older complete frame without dropping partial USB packets.
        """
        with self.condition:
            if not self.stopping:
                self.frame = frame
                self.condition.notify()

    def submit_diagnostics(self, event: dict[str, Any]) -> None:
        """
        Keep reporting callbacks independent of output pipe waits.
        """
        with self.condition:
            if not self.stopping:
                self.diagnostics = event
                self.condition.notify()

    def run(self) -> None:
        """
        Serialize snapshots through the service's shared frame/reply publication lock.
        """
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.stopping or self.frame is not None
                        or self.diagnostics is not None
                    )
                    if self.stopping:
                        return
                    frame, diagnostics = self.frame, self.diagnostics
                    self.frame = self.diagnostics = None
                if diagnostics is not None:
                    self.publish({"kind": "diagnostics", "event": diagnostics}, b"")
                if frame is not None:
                    self.publish({"kind": "frame"}, frame)
        except (OSError, ValueError):
            # A disconnected viewer is handled by command EOF and normal USB cleanup.
            with self.condition:
                self.stopping = True
                self.frame = self.diagnostics = None

    def close(self) -> None:
        """
        Stop pending delivery without delaying camera release for a stalled viewer.
        """
        with self.condition:
            self.stopping = True
            self.frame = self.diagnostics = None
            self.condition.notify()
        self.thread.join(timeout=2)
