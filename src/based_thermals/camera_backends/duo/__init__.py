"""
Audited TC002C Duo backend: transport, frame decoding and hardware controls.
"""

from .adapter import (
    PROFILE,
    control_specs,
    decode_frame,
    temperatures,
    validate_hardware,
)

__all__ = [
    "PROFILE",
    "control_specs",
    "decode_frame",
    "temperatures",
    "validate_hardware",
]
