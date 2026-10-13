"""
Common camera failures independent of transport and device protocol.
"""


class CameraError(RuntimeError):
    """
    Base error for camera discovery, setup, or streaming.
    """
