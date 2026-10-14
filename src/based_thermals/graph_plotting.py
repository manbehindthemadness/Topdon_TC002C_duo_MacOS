"""Bounded chart history compression, visible selection and pixel geometry."""

from collections import deque
from collections.abc import Sequence

import numpy as np

from .graph_settings import MAX_HISTORY_SAMPLES

HistorySample = tuple[float, tuple[float, ...]]
History = Sequence[HistorySample] | deque[HistorySample]


def visible_history(
    history: History,
    now: float,
    duration: float | None,
    preceding_samples: int = 0,
) -> tuple[HistorySample, ...]:
    """
    Select chronological visible samples without traversing older history.

    A missing duration selects the entire retained session. Histories are ordered
    by timestamp; retain optional preceding samples to warm up a rolling average.
    """
    if not history:
        return ()
    cutoff = None if duration is None else now - duration
    if history[-1][0] <= now and (cutoff is None or history[0][0] >= cutoff):
        return history if isinstance(history, tuple) else tuple(history)
    samples = []
    for sample in reversed(history):
        if sample[0] > now:
            continue
        if cutoff is not None and sample[0] < cutoff:
            if preceding_samples == 0:
                break
            preceding_samples -= 1
        samples.append(sample)
    samples.reverse()
    return tuple(samples)


def compress_history(history: deque, limit: int = MAX_HISTORY_SAMPLES) -> None:
    """
    Keep recent samples verbatim; retain extrema and gap markers in older buckets.
    """
    if len(history) <= limit:
        return
    samples = list(history)
    split = len(samples) - limit // 2
    older = []
    for offset in range(0, split, 64):
        bucket = samples[offset : min(offset + 64, split)]
        values = np.asarray([value for _, value in bucket], dtype=float)
        selected = {0, len(bucket) - 1}
        for column in values.T:
            finite = np.flatnonzero(np.isfinite(column))
            if finite.size:
                selected.update(
                    (
                        int(finite[np.argmin(column.take(finite))]),
                        int(finite[np.argmax(column.take(finite))]),
                    )
                )
        gaps = np.flatnonzero(~np.isfinite(values).all(axis=1))
        if gaps.size:
            selected.update((int(gaps[0]), int(gaps[-1])))
        previous = None
        for index in sorted(selected):
            if previous is not None:
                between = gaps[(gaps > previous) & (gaps < index)]
                if between.size:
                    older.append((bucket[int(between[0])][0], (float("nan"),) * len(bucket[0][1])))
            older.append(bucket[index])
            previous = index
    history.clear()
    history.extend(older)
    history.extend(samples[split:])


def pixel_curve_samples(x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Retain first/last/extrema per pixel column and splits at measurement gaps.

    Return chronological sample indices and curve split offsets. At most four
    finite samples remain per occupied pixel column. Gap detection uses the
    original samples, including invalid readings discarded by pixel reduction.
    """
    finite = np.isfinite(values)
    indices = np.flatnonzero(finite)
    if not indices.size:
        return indices, np.empty(0, dtype=np.intp)
    columns = x[indices]
    starts = np.r_[0, np.flatnonzero(np.diff(columns) != 0) + 1]
    ends = np.r_[starts[1:], len(indices)]
    lengths = ends - starts
    valid_values = values[indices]
    minimum = np.minimum.reduceat(valid_values, starts)
    maximum = np.maximum.reduceat(valid_values, starts)
    offsets = np.arange(len(indices))
    min_offsets = np.minimum.reduceat(
        np.where(valid_values == np.repeat(minimum, lengths), offsets, len(indices)),
        starts,
    )
    max_offsets = np.minimum.reduceat(
        np.where(valid_values == np.repeat(maximum, lengths), offsets, len(indices)),
        starts,
    )
    selected = np.zeros(len(values), dtype=bool)
    for positions in (starts, ends - 1, min_offsets, max_offsets):
        selected[indices[positions]] = True
    retained = np.flatnonzero(selected)
    gaps = np.cumsum(~finite)
    boundaries = np.flatnonzero(np.diff(gaps[retained]) > 0) + 1
    return retained, boundaries
