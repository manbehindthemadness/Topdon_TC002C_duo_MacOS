"""
Device-profile settings isolation with preservation of the existing Duo preferences.
"""

import hashlib
from pathlib import Path

from .contracts import CameraProfile

DEVICE_KEYS = frozenset({
    "hardware", "spot_hardware", "pipeline", "spots", "ambient_input_celsius", "advanced_auto",
    "auto_calibrate", "fixed_range", "processing_preset", "camera_gamma", "camera_boost",
    "distance_calibration", "emissivity_calibration", "reflected_calibration",
})


def settings_path(base: Path, profile: CameraProfile | None) -> Path:
    """
    Keep legacy Duo data in place and scope other devices to their compatibility identity.
    """
    if profile is None or profile.id == "duo":
        return base.with_name("settings.json")
    digest = hashlib.sha256(profile.settings_key.encode()).hexdigest()[:24]
    return base.parent / "camera-profiles" / f"{digest}.json"
