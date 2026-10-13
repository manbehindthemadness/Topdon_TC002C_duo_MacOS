"""
Explicit backend selection and registration without device probing.
"""

from collections.abc import Callable
from typing import Any

from .contracts import CameraBackend, CameraProfile

_FACTORIES: dict[str, Callable[[], CameraBackend]] = {}


def register_backend(name: str, factory: Callable[[], CameraBackend]) -> None:
    """
    Register an application-owned backend without probing devices or running its factory.
    """
    if not name or name in ("duo", "virtual-dual") or name in _FACTORIES or not callable(factory):
        raise ValueError("Invalid or duplicate camera backend")
    _FACTORIES[name] = factory


def create_camera(name: str = "duo", *, duo_factory: Callable[[], Any] | None = None) -> Any:
    """
    Construct the explicitly selected backend, preserving the Duo dependency boundary.
    """
    if name == "duo":
        if duo_factory is None:
            from .duo.camera import TC002CDuoCamera

            duo_factory = TC002CDuoCamera
        from .duo.macos_usb.client import use_helper

        duo_factory = use_helper(duo_factory)
        return duo_factory()
    if name == "virtual-dual":
        from .virtual_dual.camera import VirtualDualCamera

        return VirtualDualCamera()
    if name not in _FACTORIES:
        raise ValueError(f"Unknown camera backend: {name}")
    camera = _FACTORIES[name]()
    if not isinstance(getattr(camera, "profile", None), CameraProfile):
        raise TypeError("Camera backend must provide a compatible device profile")
    if camera.profile.id == "duo":
        raise ValueError("The Duo profile identity is reserved for its audited backend")
    return camera

