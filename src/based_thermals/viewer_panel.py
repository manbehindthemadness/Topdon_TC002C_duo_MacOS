"""
Nonblocking Qt viewer bridge with one acknowledged, memory-mapped image buffer.
"""

from __future__ import annotations

import mmap
import tempfile
from contextlib import ExitStack
from typing import Any

import numpy as np

from .capture_panel import CapturePanel

FRAME_CAPACITY = 64 * 1024 * 1024


class ViewerPanel(CapturePanel):
    """
    Send display-only frames without queuing images or importing Qt in acquisition.
    """

    window_module = "based_thermals.viewer_window"
    window_label = "Viewer"

    def __init__(self) -> None:
        """
        Allocate image storage lazily when the viewer opens.
        """
        super().__init__()
        self._resources = ExitStack()
        self._file: Any = None
        self._mapping: mmap.mmap | None = None
        self._pending_frame: int | None = None
        self._serial = 0
        self.size: tuple[int, int] | None = None
        self.windowed_size: tuple[int, int] | None = None
        self.over_image = False
        self.closed = False
        self._resize: tuple[int, int] | None = None

    def open(self, state: dict) -> None:
        """
        Start Qt with a private temporary image file, removed on shutdown.
        """
        self.close()
        self.closed = False
        try:
            # CapturePanel.open calls close before launching, so create storage afterwards.
            super().open(state)
            self._file = self._resources.enter_context(
                tempfile.NamedTemporaryFile(prefix="based-thermals-viewer-")  # noqa: SIM115
            )
            self._file.truncate(FRAME_CAPACITY)
            self._mapping = mmap.mmap(self._file.fileno(), FRAME_CAPACITY)
        except OSError:
            self.close()
            raise

    def present(self, image: np.ndarray, state: dict) -> None:
        """
        Publish at most one frame until Qt has copied and acknowledged it.
        """
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Viewer frames must be uint8 BGR images")
        if image.nbytes > FRAME_CAPACITY:
            raise ValueError("Viewer image exceeds the display buffer capacity")
        message = dict(state)
        if self._resize is not None:
            message["resize"] = list(self._resize)
        if self._pending_frame is None and self._mapping is not None:
            frame = np.ascontiguousarray(image)
            self._mapping[:frame.nbytes] = frame.tobytes()
            self._serial += 1
            self._pending_frame = self._serial
            message["frame"] = {
                "path": self._file.name,
                "serial": self._serial,
                "width": frame.shape[1],
                "height": frame.shape[0],
            }
        # Never replace an unsent frame descriptor with a state-only update.
        if "frame" in message or not self._outgoing:
            self.update(message)
            self._resize = None

    def poll(self) -> list[dict]:
        """
        Consume acknowledgements and geometry; return user actions to the session.
        """
        commands = []
        for message in super().poll():
            action = message.get("action")
            if action == "frame_read":
                if message.get("serial") == self._pending_frame:
                    self._pending_frame = None
            elif action == "geometry":
                for name in ("size", "windowed_size"):
                    value = message.get(name)
                    if (
                        isinstance(value, list) and len(value) == 2
                        and all(type(n) is int and 0 < n <= 16384 for n in value)
                    ):
                        setattr(self, name, tuple(value))
            elif action == "hover":
                self.over_image = message.get("inside") is True
            else:
                commands.append(message)
        if not self.is_open:
            self.closed = True
        return commands

    def resize(self, size: tuple[int, int]) -> None:
        """
        Request image-area dimensions without changing the image buffer ownership.
        """
        self._resize = size

    def close(self) -> None:
        """
        Stop Qt before unmapping and removing its image storage.
        """
        super().close()
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._resources.close()
            self._file = None
        self._pending_frame = None
        self._resize = None
