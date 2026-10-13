"""
Public import facade for the synthetic dual-camera backend.
"""

from .virtual_dual import (
    PROFILE,
    VirtualDualCamera,
    synthetic_temperatures,
)

__all__ = [
    "PROFILE",
    "VirtualDualCamera",
    "synthetic_temperatures",
]
