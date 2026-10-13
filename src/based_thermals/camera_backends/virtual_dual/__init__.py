"""
Synthetic dual-camera backend for offline development.
"""

from .camera import (
    PROFILE,
    VirtualDualCamera,
    synthetic_temperatures,
)

__all__ = [
    "PROFILE",
    "VirtualDualCamera",
    "synthetic_temperatures",
]
