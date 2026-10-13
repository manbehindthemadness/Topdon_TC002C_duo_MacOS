"""
Explicit backend registration; importing a profile never opens a camera.
"""

from .contracts import CameraBackend, CameraProfile, ControlSpec
from .duo import PROFILE as DUO_PROFILE
from .frames import CameraFrame, ReportedReading, decode_frame
from .registry import create_camera, register_backend
from .visible import VisibleFrame

__all__ = [
    "DUO_PROFILE",
    "CameraBackend",
    "CameraFrame",
    "CameraProfile",
    "ControlSpec",
    "ReportedReading",
    "VisibleFrame",
    "create_camera",
    "decode_frame",
    "register_backend",
]
