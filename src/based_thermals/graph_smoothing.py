"""Display-only sample averaging and shape-preserving chart interpolation."""

import numpy as np


def rolling_average(values: np.ndarray, samples: int) -> np.ndarray:
    """
    Average each series' trailing retained samples, restarting after every gap.

    Use available samples at the start of a run and leave invalid samples invalid.
    Never modify the original history or mix separate measurement series.
    """
    if samples == 1:
        return values
    result = np.full_like(values, np.nan, dtype=np.float64)
    stops = np.arange(1, len(values) + 1)
    window_starts = np.maximum(0, stops - samples)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        cumulative = np.r_[0.0, np.cumsum(np.where(finite, values[:, column], 0.0))]
        if finite.all():
            result[:, column] = (cumulative[stops] - cumulative[window_starts]) / (
                stops - window_starts
            )
            continue
        last_gap = np.maximum.accumulate(np.where(finite, 0, stops))
        starts = np.maximum(last_gap, stops - samples)
        valid = np.flatnonzero(finite)
        result[valid, column] = (cumulative[stops[valid]] - cumulative[starts[valid]]) / (
            stops[valid] - starts[valid]
        )
    return result


def interpolate_samples(
    stamps: np.ndarray,
    values: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add N smooth intermediate points per interval within one finite curve run.

    Weighted harmonic tangents preserve monotonic segments and their extrema.
    Input is already bounded by chart pixels; repeated timestamps bypass smoothing.
    """
    if samples == 0 or len(stamps) < 2:
        return stamps, values
    intervals = np.diff(stamps)
    if np.any(intervals <= 0):
        return stamps, values
    secants = np.diff(values) / intervals
    tangents = np.r_[secants[0], np.zeros(len(values) - 2), secants[-1]]
    matching = (secants[:-1] * secants[1:]) > 0
    first_weight = 2 * intervals[1:] + intervals[:-1]
    second_weight = intervals[1:] + 2 * intervals[:-1]
    indices = np.flatnonzero(matching)
    tangents[indices + 1] = (first_weight[indices] + second_weight[indices]) / (
        first_weight[indices] / secants[indices] + second_weight[indices] / secants[indices + 1]
    )
    fraction = np.arange(samples + 1) / (samples + 1)
    square, cube = fraction**2, fraction**3
    curve = (
        (2 * cube - 3 * square + 1) * values[:-1, None]
        + (cube - 2 * square + fraction) * (intervals * tangents[:-1])[:, None]
        + (-2 * cube + 3 * square) * values[1:, None]
        + (cube - square) * (intervals * tangents[1:])[:, None]
    )
    times = stamps[:-1, None] + intervals[:, None] * fraction
    return np.r_[times.ravel(), stamps[-1]], np.r_[curve.ravel(), values[-1]]
