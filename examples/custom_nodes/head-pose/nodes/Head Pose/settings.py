"""
Validate Head Pose controls before resolving models or creating inference sessions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Settings:
    """
    Retain validated display and inference parameters for one frame.
    """

    threshold: float
    max_heads: int
    rotation: int
    boxes: bool
    axes: bool
    angles: bool

    @classmethod
    def read(cls, config: dict[str, Any]) -> "Settings":
        """
        Reject malformed JSON values even when no heads or models are available.
        """
        threshold = config.get("score_threshold", 0.7)
        if (isinstance(threshold, bool) or not isinstance(threshold, (float, int))
                or not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("Head confidence must be between 0 and 1")
        maximum = config.get("max_heads", 1)
        if type(maximum) is not int or not 1 <= maximum <= 16:
            raise ValueError("Maximum heads must be an integer between 1 and 16")
        rotation = config.get("input_rotation", "0")
        if not isinstance(rotation, str) or rotation not in ("0", "90", "180", "270"):
            raise ValueError("Input rotation must be 0, 90, 180 or 270")
        flags = [config.get(key, True) for key in ("show_boxes", "show_axes", "show_angles")]
        if any(type(value) is not bool for value in flags):
            raise ValueError("Head overlay switches must be booleans")
        settings = cls(float(threshold), maximum, int(rotation) // 90, *flags)
        return settings
