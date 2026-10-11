"""
Explicit backend registration; importing a profile never opens a camera.
"""

from collections.abc import Callable
from typing import Any

from .contracts import CameraBackend, CameraProfile, ControlSpec
from .duo import PROFILE as DUO_PROFILE
from .frames import CameraFrame, ReportedReading, decode_frame

_FACTORIES: dict[str, Callable[[], CameraBackend]] = {}


def register_backend(name: str, factory: Callable[[], CameraBackend]) -> None:
    """
    Register an application-owned backend without probing devices or running its factory.
    """
    if not name or name == "duo" or name in _FACTORIES or not callable(factory):
        raise ValueError("Invalid or duplicate camera backend")
    _FACTORIES[name] = factory


def create_camera(name: str = "duo", *, duo_factory: Callable[[], Any] | None = None) -> Any:
    """
    Construct the explicitly selected backend, preserving the Duo dependency boundary.
    """
    if name == "duo":
        if duo_factory is None:
            from ..camera import TC002CDuoCamera

            duo_factory = TC002CDuoCamera
        from ..macos_usb.client import use_helper

        duo_factory = use_helper(duo_factory)
        return duo_factory()
    if name not in _FACTORIES:
        raise ValueError(f"Unknown camera backend: {name}")
    camera = _FACTORIES[name]()
    if not isinstance(getattr(camera, "profile", None), CameraProfile):
        raise TypeError("Camera backend must provide a compatible device profile")
    if camera.profile.id == "duo":
        raise ValueError("The Duo profile identity is reserved for its audited backend")
    return camera


__all__ = [
    "DUO_PROFILE",
    "CameraBackend",
    "CameraFrame",
    "CameraProfile",
    "ControlSpec",
    "ReportedReading",
    "create_camera",
    "decode_frame",
    "register_backend",
]
