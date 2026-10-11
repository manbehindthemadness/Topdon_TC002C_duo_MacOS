"""
Scalar point and three-value area graph/CSV series contracts.
"""

from __future__ import annotations

MeasurementValue = float | tuple[float, float, float]


def series_values(value: MeasurementValue) -> tuple[float, ...]:
    """
    Normalize a point temperature or minimum/average/maximum area values.
    """
    values = value if isinstance(value, tuple) else (value,)
    return tuple(float(item) for item in values)


def series_labels(value: MeasurementValue) -> tuple[str, ...]:
    """
    Label the three area curves while retaining legacy scalar point graphs.
    """
    return ("Min", "Avg", "Max") if isinstance(value, tuple) else ("Temp",)
