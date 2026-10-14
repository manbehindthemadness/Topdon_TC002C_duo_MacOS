"""Bounded geometry caches and explicitly frame-scoped measurement readings."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .paths import boundary

if TYPE_CHECKING:
    from ..desktop_app.spots import SampleSpots
    from .display_masks import ShapeMasks
    from .overlays import MeasurementLabel, Rectangle
    from .sampling import AreaReading

MASK_CACHE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class CachedBoundary:
    """
    Keep display coordinates with the exact native geometry and scale that produced them.
    """

    key: tuple
    points: NDArray[np.int32]


def geometry_key(region: dict, center: tuple[int, int]) -> tuple:
    """
    Identify native geometry without display color, labels, or layer preferences.
    """
    return region["kind"], region["radius"], tuple(map(tuple, region["points"])), center


class MeasurementDisplayCache:
    """
    Retain one boundary/mask per saved identity and one current label layout.
    """

    def __init__(self) -> None:
        """
        Start with no retained display geometry or active measurement frame.
        """
        self.boundaries: dict[int, CachedBoundary] = {}
        self.masks: OrderedDict[int, tuple[tuple, ShapeMasks]] = OrderedDict()
        self.mask_bytes = 0
        self.layout_key: tuple | None = None
        self.layout: tuple[tuple[MeasurementLabel, ...], tuple[Rectangle, ...]] | None = None
        self._frame = False
        self._readings_key: tuple | None = None
        self._temperatures: np.ndarray | None = None
        self._readings: dict[int, AreaReading] | None = None

    def begin_frame(self) -> None:
        """
        Start a new sampling scope, even if the renderer reuses its temperature array.
        """
        self.end_frame()
        self._frame = True

    def end_frame(self) -> None:
        """
        Release frame-specific readings so unrelated callers always sample fresh data.
        """
        self._frame = False
        self._readings_key = None
        self._temperatures = None
        self._readings = None

    def readings(self, spots: SampleSpots, temperatures: np.ndarray) -> dict[int, AreaReading]:
        """
        Share readings only within an explicit frame and unchanged measurement geometry.
        """
        if not self._frame:
            return spots.readings(temperatures)
        key = tuple((number, geometry_key(spots.region(number), center),
                     spots.region(number)["exclude"]) for number, center in spots.active)
        if self._temperatures is not temperatures or key != self._readings_key:
            self._readings = spots.readings(temperatures)
            self._temperatures = temperatures
            self._readings_key = key
        assert self._readings is not None
        return self._readings

    def prune(self, numbers: list[int]) -> None:
        """
        Release deleted identities rather than retaining old geometry across clears.
        """
        saved = set(numbers)
        for cache in (self.boundaries, self.masks):
            for number in tuple(cache):
                if number not in saved:
                    if cache is self.masks:
                        self.mask_bytes -= self.masks[number][1].byte_size
                    del cache[number]

    def store_masks(self, number: int, key: tuple, masks: ShapeMasks) -> None:
        """
        Retain local rasters within a fixed byte budget, evicting least-recently used shapes.
        """
        previous = self.masks.pop(number, None)
        if previous is not None:
            self.mask_bytes -= previous[1].byte_size
        if masks.byte_size > MASK_CACHE_BYTES:
            return
        while self.masks and self.mask_bytes + masks.byte_size > MASK_CACHE_BYTES:
            _number, (_key, expired) = self.masks.popitem(last=False)
            self.mask_bytes -= expired.byte_size
        self.masks[number] = key, masks
        self.mask_bytes += masks.byte_size

    def boundary(self, number: int, region: dict, center: tuple[int, int], scale: int) -> NDArray[np.int32]:
        """
        Reuse an oriented display boundary until its coordinates or scale change.
        """
        key = geometry_key(region, center), scale
        cached = self.boundaries.get(number)
        if cached is not None and cached.key == key:
            return cached.points
        points = np.rint(np.asarray(boundary(region, center)) * scale + scale // 2).astype(np.int32)
        self.boundaries[number] = CachedBoundary(key, points)
        return points

    @staticmethod
    def text_bounds(text: str, font_scale: float) -> tuple[int, int, int]:
        """
        Describe text geometry independently of changing temperature digits.
        """
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        return width, height, baseline
