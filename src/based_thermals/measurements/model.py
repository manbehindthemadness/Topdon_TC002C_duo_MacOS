"""
Measurement extensions to the existing stable spot identity and positioning model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from .sampling import AreaReading, kernel, sample
from .schema import default_region, transform_region, validate_center, validate_region

if TYPE_CHECKING:
    from ..desktop_app.spots import SampleSpots


class MeasurementRegions:
    """
    Add area geometry, label configuration and kernel sampling to spot identities.
    """

    regions: dict[int, dict]
    region_masks: dict[int, tuple[tuple, np.ndarray]]

    @property
    def display_active(self: SampleSpots) -> list[tuple[int, tuple[int, int]]]:
        """
        Order enabled display shapes by Z index, preserving existing order for ties.
        """
        ordered = sorted(self.active, key=lambda item: self.region(item[0]).get("z_index", 0))
        return ordered

    def region(self: SampleSpots, number: int) -> dict:
        """
        Return a fresh default for legacy points or the configured geometry.
        """
        return self.regions.get(number, default_region())

    def configure_region(
        self: SampleSpots,
        number: object,
        region: object,
        size: tuple[int, int],
        *,
        pixel: object | None = None,
    ) -> None:
        """
        Validate an entire shape before replacing its configuration.
        """
        if not isinstance(number, int) or isinstance(number, bool) or number not in self.numbers:
            raise ValueError("Choose an existing measurement")
        validated = validate_region(region, size)
        center = validate_center(pixel, size) if pixel is not None else None
        self.regions[number] = validated
        if center is not None:
            self.pixels[self.numbers.index(number)] = center

    def create_region(
        self: SampleSpots,
        pixel: object,
        region: object,
        size: tuple[int, int],
    ) -> int:
        """
        Create a measurement independently of the legacy point-placement toggle.
        """
        if len(self.numbers) >= 1000:
            raise ValueError("At most 1,000 measurements can be saved")
        center = validate_center(pixel, size)
        validated = validate_region(region, size)
        number = self.next_number
        self.next_number += 1
        self.numbers.append(number)
        self.pixels.append(center)
        self.regions[number] = validated
        return number

    def transform_regions(self: SampleSpots, transform: Any) -> None:
        """
        Transform vertices and handles with the same mapping as their spot anchors.
        """
        self.regions = {
            number: transform_region(region, transform) for number, region in self.regions.items()
        }

    def exclusion_mask(self: SampleSpots, shape: tuple[int, int]) -> np.ndarray:
        """
        Union enabled exclusion areas without producing readings or graph entries.
        """
        excluded = np.zeros(shape, bool)
        for number, center in self.active:
            region = self.region(number)
            if region["exclude"]:
                excluded |= self.measurement_mask(number, center, shape)
        return excluded

    def measurement_mask(
        self: SampleSpots,
        number: int,
        center: tuple[int, int],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Reuse rasterized kernels until shape, position or native orientation changes.
        """
        region = self.region(number)
        signature = (
            region["kind"],
            region["radius"],
            tuple(tuple(point) for point in region["points"]),
            center,
            shape,
        )
        cached = self.region_masks.get(number)
        if cached is None or cached[0] != signature:
            cached = signature, kernel(region, center, shape)
            self.region_masks[number] = cached
        return cached[1]

    def readings(self: SampleSpots, temperatures: np.ndarray) -> dict[int, AreaReading]:
        """
        Sample enabled measurements in the native grid after exclusion subtraction.
        """
        shape = temperatures.shape[:2]
        excluded = self.exclusion_mask(shape)
        readings = {}
        for number, center in self.active:
            region = self.region(number)
            if not region["exclude"]:
                if region["kind"] == "spot" and region["radius"] == 0:
                    x, y = center
                    value = float(temperatures[y, x]) if not excluded[y, x] else float("nan")
                    if np.isfinite(value):
                        readings[number] = AreaReading(value, value, value, 1)
                    else:
                        readings[number] = AreaReading(float("nan"), float("nan"), float("nan"), 0)
                else:
                    readings[number] = sample(
                        temperatures, self.measurement_mask(number, center, shape), excluded
                    )
        return readings

    def graph_readings(self: SampleSpots, temperatures: np.ndarray) -> tuple:
        """
        Retain scalar legacy point histories and give every area three scalar curves.
        """
        samples = []
        for number, reading in self.display_cache.readings(self, temperatures).items():
            region = self.region(number)
            values = (
                reading.average
                if region["kind"] == "spot" and region["radius"] == 0
                else reading.graph_values()
            )
            samples.append(((self.generation, number - 1), values))
        return tuple(samples)

    def region_states(self: SampleSpots) -> list[dict]:
        """
        Return JSON-safe editable geometry and stable identity for the popup.
        """
        return [
            {
                "number": number,
                "enabled": number not in self.disabled,
                "name": self.name(number),
                "pixel": list(center),
                "region": deepcopy(self.region(number)),
            }
            for number, center in zip(self.numbers, self.pixels, strict=True)
        ]
