"""
Retain a canvas transform's source until its receiving resolution is known.
"""

from dataclasses import dataclass
from math import ceil
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
    source_detail_size: tuple[int, int] | None = None

    def detail_size(self) -> tuple[int, int]:
        """
        Give a shrunk image at least one destination sample per retained source pixel.
        """
        height, width = self.image.shape[:2]
        if self.source_detail_size is not None:
            width, height = self.source_detail_size
        size = (ceil(width * max(1, 100 / self.params["width"])),
                ceil(height * max(1, 100 / self.params["height"])))
        return size

    def render(self, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Sample pixels directly while preserving the declared canvas's occupied pixel boundaries.
        """
        image = render_transform(self.image, self.params, size)
        coverage = resize_coverage(self.coverage, (size[1], size[0]))
        return image, coverage
