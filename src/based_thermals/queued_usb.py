"""
Public import facade; the implementation belongs to the Duo backend.
"""

from .camera_backends.duo.queued_usb import (
    QueuedBulkReader,
)

__all__ = [
    "QueuedBulkReader",
]
