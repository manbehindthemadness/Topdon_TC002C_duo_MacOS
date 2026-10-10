"""
Example custom display node with a relative helper import.
"""

from typing import Any

import numpy as np

from .contrast import adjust_contrast


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Adjust display contrast using the package's JSON configuration.
    """
    result = adjust_contrast(image, float(config.get("contrast", 1.2)))
    return result
