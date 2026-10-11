"""
Retain a terminal canvas transform's source until its receiving resolution is known.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..image_transform import render_transform
from .coverage import resize_coverage


@dataclass
class CanvasTransform:
    """
    Keep display pixels and their original footprint together for one direct resampling.
    """

    image: np.ndarray
    params: dict[str, Any]
    coverage: np.ndarray | None

    def render(self, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Sample pixels directly while preserving the declared canvas's occupied pixel boundaries.
        """
        image = render_transform(self.image, self.params, size)
        coverage = resize_coverage(self.coverage, (size[1], size[0]))
        return image, coverage
