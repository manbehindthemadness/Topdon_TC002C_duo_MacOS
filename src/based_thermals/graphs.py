"""Bounded temperature history and graph rendering on a dedicated worker thread."""

import csv
import math
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from .graph_controls import (
    draw_graph_logging_control,
    graph_config_rect,
    graph_interval_rect,
    graph_log_button_rect,
    graph_reset_rect,
)
from .graph_plotting import History, compress_history, pixel_curve_samples, visible_history
from .graph_settings import (
    GRAPH_DEFAULTS,
    GRAPH_INTERVAL,
    MAX_HISTORY_SAMPLES,
    validate_graph_settings,
)
from .graph_smoothing import interpolate_samples, rolling_average
from .measurements.series import MeasurementValue, series_labels, series_values

__all__ = [
    "COLORS",
    "GRAPH_DEFAULTS",
    "GRAPH_INTERVAL",
    "HISTORY_SECONDS",
    "MAX_GRAPH_INTERVAL",
    "MAX_HISTORY_SAMPLES",
    "MIN_GRAPH_INTERVAL",
    "GraphIntervalEditor",
    "GraphSnapshot",
    "GraphWorker",
    "History",
    "compress_history",
    "draw_graph_logging_control",
    "graph_config_rect",
    "graph_interval_rect",
    "graph_log_button_rect",
    "graph_reset_rect",
    "history_duration",
    "render_graphs",
    "spot_title",
    "validate_graph_interval",
    "validate_graph_settings",
]

MIN_GRAPH_INTERVAL = 0.1
MAX_GRAPH_INTERVAL = 60.0
HISTORY_SECONDS = 60
COLORS = ((230, 160, 80), (90, 210, 110), (90, 110, 245), (220, 190, 240))


def validate_graph_interval(value: object) -> float:
    """
    Require a finite numeric interval within the supported graph refresh range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Graph update interval must be a number")
    value = float(value)
    if not math.isfinite(value) or not MIN_GRAPH_INTERVAL <= value <= MAX_GRAPH_INTERVAL:
        raise ValueError("Graph update interval must be between 0.1 and 60 seconds")
    return value


@dataclass
class GraphIntervalEditor:
    """
    Track pending edits in the legacy graph interval field.
    """

    value: float = GRAPH_INTERVAL
    text: str | None = None
    replace_text: bool = False

    def begin(self) -> None:
        """
        Start editing the current graph interval and select its previous value.
        """
        self.text = f"{self.value:g}"
        self.replace_text = True

    def key(self, key: int) -> float | None:
        """
        Edit the legacy interval field only while an edit is active.
        """
        current_text = self.text
        if current_text is None:
            return None
        if key == 27:
            self.text = None
        elif key in (10, 13):
            value = validate_graph_interval(float(current_text))
            self.text = None
            return value
        elif key in (8, 127):
            self.text = "" if self.replace_text else current_text[:-1]
            self.replace_text = False
        elif key in (*range(ord("0"), ord("9") + 1), ord(".")):
            edited_text = ("" if self.replace_text else current_text) + chr(key)
            self.text = edited_text[:8]
            self.replace_text = False
        return None


@dataclass(frozen=True)
class GraphSnapshot:
    """
    Hold immutable frame readings, chart dimensions and display preferences.
    """

    stats: tuple[float, float, float, float]
    spots: tuple[tuple[tuple[int, int], MeasurementValue], ...]
    size: tuple[int, int]  # width, height of the reserved graph area
    unit: str = "C"
    spot_names: tuple[tuple[tuple[int, int], str], ...] = ()
    measurements_valid: bool = True
    history_range: float | None = None
    averaging_samples: int = 1
    interpolation_samples: int = 0


def spot_title(snapshot: GraphSnapshot, key: tuple[int, int]) -> str:
    """
    Label stable identities as measurements for areas and spots for scalar points.
    """
    value = dict(snapshot.spots).get(key)
    prefix = "Measurement" if isinstance(value, tuple) else "Spot"
    default = f"{prefix} {key[1] + 1}"
    name = dict(snapshot.spot_names).get(key, default)
    return default if name == default else f"{name} ({default})"


def history_duration(seconds: float) -> str:
    """
    Format the graph history span for the viewer and captured headings.
    """
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


class GraphWorker:
    """
    Sample temperature history independently of USB capture and GUI rendering.
    """

    def __init__(self) -> None:
        """
        Initialize bounded history, logging state and the background graph worker.
        """
        self._condition = threading.Condition()
        self._latest = None
        self._image = None
        self._closed = False
        self._log_file = None
        self._log_writer = None
        self._log_path = None
        self._log_started = 0.0
        self._log_error = None
        self._master = deque()
        self._spots = {}
        self._settings = GRAPH_DEFAULTS.copy()
        self._interval = GRAPH_INTERVAL
        self._reschedule = False
        self._redraw = False
        self._history_generation = 0
        self._history_paused = False
        self._thread = threading.Thread(target=self._run, name="temperature-graphs", daemon=True)
        self._thread.start()

    def set_interval(self, value: object) -> None:
        """
        Validate a new sampling interval and wake the worker to reschedule.
        """
        value = validate_graph_interval(value)
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before changing the graph update interval")
            self._interval = value
            self._reschedule = True
            self._condition.notify()

    def configure(self, settings: object) -> None:
        """
        Apply chart preferences and redraw retained data without adding samples.
        """
        settings = validate_graph_settings(settings)
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before changing graph settings")
            self._settings = settings
            self._history_generation += 1
            if self._latest is not None:
                self._latest = self._configured_snapshot(self._latest)
                self._redraw = True
                self._condition.notify()
        self._compact_histories()

    def clear_history(self) -> None:
        """
        Clear session history and invalidate in-flight renders and compaction.
        """
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before clearing chart data")
            self._master.clear()
            self._spots.clear()
            self._history_paused = False
            self._history_generation += 1
            self._image = None
            self._redraw = True
            self._condition.notify()

    def _configured_snapshot(self, snapshot: GraphSnapshot) -> GraphSnapshot:
        """
        Attach the worker's display preferences to an immutable render snapshot.
        """
        return replace(
            snapshot,
            history_range=self._history_range(),
            averaging_samples=self._settings["averaging_samples"],
            interpolation_samples=self._settings["interpolation_samples"],
        )

    def _history_range(self) -> float | None:
        """
        Return the visible fixed duration, or None for the retained session.
        """
        if self._settings["range_mode"] != "fixed":
            return None
        duration = float(self._settings["range_seconds"])
        return duration

    def clear_spot_history(self, key: tuple[int, int]) -> None:
        """
        Clear one measurement's graph without deleting its shape or other histories.
        """
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before clearing measurement history")
            self._spots.pop(key, None)
            self._history_generation += 1
            self._image = None
            self._redraw = True
            self._condition.notify()

    def _compact(self, history: deque, settings: dict | None = None) -> None:
        """
        Compact private history with a captured preference snapshot.
        """
        settings = self._settings if settings is None else settings
        limit = settings["history_points"]
        if settings["compression"]:
            compress_history(history, limit)
        else:
            while len(history) > limit:
                history.popleft()

    def _compact_histories(self) -> None:
        """
        Compact copies outside the UI lock, then publish only current generations.

        Copy one history per short lock acquisition. Preserve samples appended by
        the worker during a concurrent settings change and reject reset histories.
        """
        with self._condition:
            generation = self._history_generation
            settings = self._settings.copy()
            histories = [(None, self._master), *self._spots.items()]
        for key, original in histories:
            with self._condition:
                current = self._master if key is None else self._spots.get(key)
                if self._closed or generation != self._history_generation:
                    return
                if current is not original or len(original) <= settings["history_points"]:
                    continue
                copied_count = len(original)
                compacted = deque(original)
            self._compact(compacted, settings)
            with self._condition:
                current = self._master if key is None else self._spots.get(key)
                if self._closed or generation != self._history_generation:
                    return
                if current is not original:
                    continue
                compacted.extend(islice(original, copied_count, None))
                # A settings change can race sampling; never lose the newer tail.
                if len(compacted) > settings["history_points"]:
                    if settings["compression"]:
                        continue
                    while len(compacted) > settings["history_points"]:
                        compacted.popleft()
                if key is None:
                    self._master = compacted
                else:
                    self._spots[key] = compacted

    @property
    def logging(self) -> bool:
        """
        Report whether CSV logging currently owns the settings lock.
        """
        with self._condition:
            return self._log_file is not None

    def start_logging(self, path: Path) -> Path:
        """
        Open a CSV log after resolving its suffix without replacing an unselected file.
        """
        with self._condition:
            if self._closed or self._log_file is not None:
                raise ValueError("Graph logging is already active or the worker is closed")
            selected = Path(path)
            path = selected if selected.suffix.lower() == ".csv" else selected.with_suffix(".csv")
            if path != selected and path.exists():
                raise FileExistsError(f"Choose {path.name} explicitly to replace that existing CSV")
            stream = path.open("w", newline="", encoding="utf-8")
            try:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "timestamp_utc",
                        "elapsed_seconds",
                        "series_id",
                        "series",
                        "temperature_celsius",
                        "temperature_display",
                        "display_unit",
                    )
                )
                stream.flush()
            except OSError:
                stream.close()
                raise
            self._log_file, self._log_writer = stream, writer
            self._log_path = path
            self._log_started = time.monotonic()
            self._log_error = None
            return path

    def stop_logging(self) -> Path | None:
        """
        Finalize the current CSV log and return its path, if logging was active.
        """
        with self._condition:
            if self._log_file is None:
                return None
            stream, self._log_file = self._log_file, None
            self._log_writer = None
            stream.close()
            return self._log_path

    def take_logging_error(self) -> str | None:
        """
        Consume a pending logging failure so the viewer can report it once.
        """
        with self._condition:
            error, self._log_error = self._log_error, None
            return error

    def _write_log(self, snapshot: GraphSnapshot, now: float) -> None:
        """
        Write each point or area statistic as its own Celsius/display CSV series.
        """
        if self._log_file is None or not snapshot.measurements_valid:
            return
        stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
        elapsed = f"{now - self._log_started:.3f}"
        series = [
            (f"scene.{name.lower()}", name, value)
            for name, value in zip(("Minimum", "Average", "Maximum", "Center"), snapshot.stats)
        ]
        for key, value in snapshot.spots:
            identifier, title = f"spot.{key[0]}.{key[1] + 1}", spot_title(snapshot, key)
            for label, reading in zip(series_labels(value), series_values(value), strict=True):
                series.append(
                    (identifier, title, reading)
                    if label == "Temp"
                    else (f"{identifier}.{label.lower()}", f"{title} {label}", reading)
                )
        try:
            for identifier, label, value in series:
                displayed = value * 1.8 + 32 if snapshot.unit == "F" else value
                self._log_writer.writerow(
                    (
                        stamp,
                        elapsed,
                        identifier,
                        label,
                        f"{value:.6f}" if np.isfinite(value) else "",
                        f"{displayed:.6f}" if np.isfinite(displayed) else "",
                        snapshot.unit,
                    )
                )
            self._log_file.flush()
        except OSError as exc:
            self._log_error = f"Temperature logging stopped: {exc}"
            try:
                self.stop_logging()
            except OSError:
                pass

    def submit(self, snapshot: GraphSnapshot) -> None:
        """
        Queue the latest frame and discard histories whose curve count changed.
        """
        with self._condition:
            first = self._latest is None
            for key, value in snapshot.spots:
                history: History | None = self._spots.get(key)
                if history and len(history[-1][1]) != len(series_values(value)):
                    self._spots.pop(key)
                    self._history_generation += 1
                    self._redraw = True
            self._latest = self._configured_snapshot(snapshot)
            if first or self._redraw:
                self._condition.notify()

    def pause(self) -> None:
        """
        Pause hidden charts and insert a history gap on the next valid submission.
        """
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before hiding the graphs")
            if self._latest is not None:
                self._history_paused = True
                self._latest = None
                self._condition.notify()

    def image(self, size: tuple[int, int], *, resize: bool = False) -> np.ndarray:
        """
        Return the cached pane, optionally resizing capture output without resampling history.
        """
        with self._condition:
            result = self._image
        width, height = size
        if result is None:
            return np.zeros((height, width, 3), np.uint8)
        if result.shape[:2] != (height, width):
            if resize:
                return cv2.resize(result, (width, height), interpolation=cv2.INTER_AREA)
            return np.zeros((height, width, 3), np.uint8)
        return result

    def close(self) -> None:
        """
        Join the worker before closing the current CSV log.
        """
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()
        self.stop_logging()

    def _sample(self, snapshot: GraphSnapshot, now: float, *, compact: bool = True) -> None:
        """
        Add point/area statistics or correctly sized gaps to active histories.
        """
        values = snapshot.stats if snapshot.measurements_valid else (float("nan"),) * 4
        self._master.append((now, values))
        if compact:
            self._compact(self._master)
        current = dict(snapshot.spots)
        self._spots = {key: history for key, history in self._spots.items() if key in current}
        for key, value in snapshot.spots:
            history = self._spots.setdefault(key, deque())
            values = series_values(value)
            if history and len(history[-1][1]) != len(values):
                history.clear()
            history.append(
                (now, values if snapshot.measurements_valid else (float("nan"),) * len(values))
            )
            if compact:
                self._compact(history)

    def _run(self) -> None:
        """
        Sample complete history while rendering snapshots of the visible interval.
        """
        deadline = 0.0
        while True:
            with self._condition:
                while not self._closed:
                    if self._reschedule:
                        deadline = self._master[-1][0] + self._interval if self._master else 0
                        self._reschedule = False
                    remaining = deadline - time.monotonic()
                    if self._latest is not None and (remaining <= 0 or self._redraw):
                        break
                    self._condition.wait(timeout=max(0, remaining) if self._latest else None)
                if self._closed:
                    return
                snapshot = self._latest
                now = time.monotonic()
                sample_due = remaining <= 0 or self._history_paused
                self._redraw = False
                if self._history_paused:
                    self._master.append((now, (float("nan"),) * 4))
                    for history in self._spots.values():
                        columns = len(history[-1][1]) if history else 1
                        history.append((now, (float("nan"),) * columns))
                    self._history_paused = False
                if sample_due:
                    self._sample(snapshot, now, compact=False)
                    self._write_log(snapshot, now)
                generation = self._history_generation
                master = visible_history(
                    self._master,
                    now,
                    snapshot.history_range,
                    snapshot.averaging_samples - 1,
                )
                spots = {
                    key: visible_history(
                        history,
                        now,
                        snapshot.history_range,
                        snapshot.averaging_samples - 1,
                    )
                    for key, history in self._spots.items()
                }
            self._compact_histories()
            result = render_graphs(snapshot, master, spots, now)
            with self._condition:
                if (
                    self._latest is not None
                    and not self._closed
                    and generation == self._history_generation
                ):
                    self._image = result
                    self._condition.notify_all()
            # No queued frames or catch-up bursts if rendering takes longer than the interval.
            if sample_due:
                deadline = max(now + self._interval, time.monotonic())


def render_graphs(
    snapshot: GraphSnapshot,
    master: History,
    spots: dict[tuple[int, int], History],
    now: float,
) -> np.ndarray:
    """
    Render scalar point curves or minimum/average/maximum area curves alongside scene stats.
    """
    width, height = snapshot.size
    canvas = np.full((height, width, 3), (23, 25, 31), np.uint8)
    duration = snapshot.history_range or (
        max(HISTORY_SECONDS, now - master[0][0]) if master else HISTORY_SECONDS
    )
    cv2.putText(
        canvas,
        f"History | {history_duration(duration)}",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        min(0.45, max(0.15, (graph_reset_rect(width)[0] - 16) / 155)),
        (215, 220, 230),
        1,
        cv2.LINE_AA,
    )
    charts: list[tuple[str, History, tuple[str, ...]]] = [
        ("Master stats", master, ("Min", "Avg", "Max", "Center"))
    ]
    charts.extend(
        (spot_title(snapshot, key), spots.get(key, ()), series_labels(value))
        for key, value in snapshot.spots
    )
    top = min(34, height)
    for index, (title, history, labels) in enumerate(charts):
        y0 = top + round((height - top) * index / len(charts))
        y1 = top + round((height - top) * (index + 1) / len(charts))
        _draw_chart(
            canvas[y0:y1],
            title,
            history,
            labels,
            snapshot.unit,
            now,
            duration,
            averaging_samples=snapshot.averaging_samples,
            interpolation_samples=snapshot.interpolation_samples,
        )
    return canvas


def _draw_chart(
    canvas: np.ndarray,
    title: str,
    history: History,
    labels: Sequence[str],
    unit: str,
    now: float,
    duration: float = HISTORY_SECONDS,
    averaging_samples: int = 1,
    interpolation_samples: int = 0,
) -> None:
    """
    Draw scalar histories with separate curves across unavailable measurements.
    """
    height, width = canvas.shape[:2]
    if height < 8 or width < 16:
        return
    cv2.line(canvas, (0, 0), (width - 1, 0), (65, 70, 82), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.48, max(0.2, height / 250))
    cv2.putText(
        canvas,
        f"{title} ({unit})",
        (10, min(20, height - 2)),
        font,
        scale,
        (235, 237, 245),
        1,
        cv2.LINE_AA,
    )
    samples = visible_history(history, now, duration, averaging_samples - 1)
    if not samples:
        return
    stamps = np.array([stamp for stamp, _ in samples])
    values = np.array([value for _, value in samples], dtype=np.float64)
    values = rolling_average(values, averaging_samples)
    visible = stamps >= now - duration
    stamps, values = stamps[visible], values[visible]
    if not len(stamps):
        return
    if unit == "F":
        values = values * 1.8 + 32
    for column, label in enumerate(labels):
        value = values[-1, column]
        text = f"{label} {value:.1f}" if np.isfinite(value) else f"{label} --"
        cv2.putText(
            canvas,
            text,
            (10 + column * max(1, (width - 20) // len(labels)), min(40, height - 2)),
            font,
            scale,
            COLORS[column],
            1,
            cv2.LINE_AA,
        )
    left, right, top, bottom = 62, width - 14, 53, height - 24
    finite = values[np.isfinite(values)]
    if bottom <= top or right <= left or not finite.size:
        return
    low, high = float(finite.min()), float(finite.max())
    padding = max(0.5, (high - low) * 0.1)
    low, high = low - padding, high + padding
    for fraction in (0, 0.5, 1):
        y = round(top + (bottom - top) * fraction)
        cv2.line(canvas, (left, y), (right, y), (46, 50, 60), 1)
        cv2.putText(
            canvas,
            f"{high - (high - low) * fraction:.1f}",
            (5, y + 4),
            font,
            0.32,
            (155, 160, 175),
            1,
            cv2.LINE_AA,
        )
    _draw_history_curves(
        canvas,
        stamps,
        values,
        (left, right, top, bottom),
        low,
        high,
        now,
        duration,
        interpolation_samples,
    )
    cv2.putText(
        canvas, f"-{history_duration(duration)}", (left, height - 7), font, 0.32, (155, 160, 175), 1
    )
    cv2.putText(canvas, "now", (max(left, right - 25), height - 7), font, 0.32, (155, 160, 175), 1)


def _draw_history_curves(
    canvas: np.ndarray,
    stamps: np.ndarray,
    values: np.ndarray,
    bounds: tuple[int, int, int, int],
    low: float,
    high: float,
    now: float,
    duration: float,
    interpolation_samples: int = 0,
) -> None:
    """
    Draw pixel-bounded curves with bulk coordinates and preserved extrema.

    Non-finite samples split curves; isolated valid samples retain a point marker.
    """
    left, right, top, bottom = bounds
    x = np.clip(
        np.rint(left + (stamps - now + duration) / duration * (right - left)),
        left,
        right,
    ).astype(np.int32)
    for column in range(values.shape[1]):
        indices, boundaries = pixel_curve_samples(x, values[:, column])
        if not indices.size:
            continue
        if interpolation_samples == 0:
            y = np.clip(
                np.rint(bottom - (values[indices, column] - low) / (high - low) * (bottom - top)),
                top,
                bottom,
            ).astype(np.int32)
            points = np.column_stack((x[indices], y))
            runs = np.split(points, boundaries)
            curves = [run for run in runs if len(run) > 1]
            if curves:
                cv2.polylines(canvas, curves, False, COLORS[column], 1, cv2.LINE_AA)
            isolated = points[np.r_[0, boundaries][np.diff(np.r_[0, boundaries, len(points)]) == 1]]
            for point in isolated:
                cv2.circle(
                    canvas, (int(point[0]), int(point[1])), 1, COLORS[column], -1, cv2.LINE_AA
                )
            continue
        runs = []
        for run_indices in np.split(indices, boundaries):
            times, curve_values = interpolate_samples(
                stamps[run_indices],
                values[run_indices, column],
                interpolation_samples,
            )
            curve_x = np.clip(
                np.rint(left + (times - now + duration) / duration * (right - left)),
                left,
                right,
            ).astype(np.int32)
            retained, _splits = pixel_curve_samples(curve_x, curve_values)
            y = np.clip(
                np.rint(bottom - (curve_values[retained] - low) / (high - low) * (bottom - top)),
                top,
                bottom,
            ).astype(np.int32)
            runs.append(np.column_stack((curve_x[retained], y)))
        curves = [run for run in runs if len(run) > 1]
        if curves:
            cv2.polylines(canvas, curves, False, COLORS[column], 1, cv2.LINE_AA)
        for run in runs:
            if len(run) == 1:
                point = (int(run[0, 0]), int(run[0, 1]))
                cv2.circle(canvas, point, 1, COLORS[column], -1, cv2.LINE_AA)
