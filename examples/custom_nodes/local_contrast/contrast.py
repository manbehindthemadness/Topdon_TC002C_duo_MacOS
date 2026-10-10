"""
Image operation separated from the package entry point.
"""

import numpy as np


def adjust_contrast(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Scale contrast around the midpoint of the display intensity range.
    """
    result = (image - 127.5) * amount + 127.5
    return result
