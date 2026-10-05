"""Bounded temperature history and graph rendering on a dedicated worker thread."""

import csv
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .graph_settings import GRAPH_DEFAULTS, validate_graph_settings

GRAPH_INTERVAL = 0.5
MIN_GRAPH_INTERVAL = 0.1
MAX_GRAPH_INTERVAL = 60.0
HISTORY_SECONDS = 60
MAX_HISTORY_SAMPLES = 4096
COLORS = ((230, 160, 80), (90, 210, 110), (90, 110, 245), (220, 190, 240))


def validate_graph_interval(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Graph update interval must be a number")
    value = float(value)
    if not math.isfinite(value) or not MIN_GRAPH_INTERVAL <= value <= MAX_GRAPH_INTERVAL:
        raise ValueError("Graph update interval must be between 0.1 and 60 seconds")
    return value


@dataclass
class GraphIntervalEditor:
    value: float = GRAPH_INTERVAL
    text: str | None = None
    replace_text: bool = False

    def begin(self):
        self.text = f"{self.value:g}"
        self.replace_text = True

    def key(self, key):
        if key == 27:
            self.text = None
        elif key in (10, 13):
            value = validate_graph_interval(float(self.text))
            self.text = None
            return value
        elif key in (8, 127):
            self.text = "" if self.replace_text else self.text[:-1]
            self.replace_text = False
        elif key in (*range(ord("0"), ord("9") + 1), ord(".")):
            self.text = ("" if self.replace_text else self.text) + chr(key)
            self.text = self.text[:8]
            self.replace_text = False
        return None


@dataclass(frozen=True)
class GraphSnapshot:
    stats: tuple[float, float, float, float]
    spots: tuple[tuple[tuple[int, int], float], ...]
    size: tuple[int, int]  # width, height of the reserved graph area
    unit: str = "C"
    spot_names: tuple[tuple[tuple[int, int], str], ...] = ()
    measurements_valid: bool = True
    history_range: float | None = None


def spot_title(snapshot, key):
    default = f"Spot {key[1] + 1}"
    name = dict(snapshot.spot_names).get(key, default)
    return default if name == default else f"{name} ({default})"


def compress_history(history, limit=MAX_HISTORY_SAMPLES):
    """Keep recent samples verbatim; retain extrema and gap markers in older buckets."""
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
                    (int(finite[np.argmin(column[finite])]), int(finite[np.argmax(column[finite])]))
                )
        gaps = np.flatnonzero(~np.isfinite(values).all(axis=1))
        if gaps.size:
            selected.update((int(gaps[0]), int(gaps[-1])))
        previous = None
        for index in sorted(selected):
            if previous is not None:
                between = gaps[(gaps > previous) & (gaps < index)]
                if between.size:
                    older.append((bucket[int(between[0])][0], (float("nan"),) * values.shape[1]))
            older.append(bucket[index])
            previous = index
    history.clear()
    history.extend(older)
    history.extend(samples[split:])


def history_duration(seconds):
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


class GraphWorker:
    """Sample scalar measurements at a configurable interval; never touch USB or GUI windows."""

    def __init__(self):
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

    def set_interval(self, value):
        value = validate_graph_interval(value)
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before changing the graph update interval")
            self._interval = value
            self._reschedule = True
            self._condition.notify()

    def configure(self, settings):
        settings = validate_graph_settings(settings)
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before changing graph settings")
            self._settings = settings
            for history in (self._master, *self._spots.values()):
                self._compact(history)
            if self._latest is not None:
                self._latest = replace(self._latest, history_range=self._history_range())
                self._redraw = True
                self._condition.notify()

    def clear_history(self):
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

    def _history_range(self):
        return self._settings["range_seconds"] if self._settings["range_mode"] == "fixed" else None

    def _compact(self, history):
        limit = self._settings["history_points"]
        if self._settings["compression"]:
            compress_history(history, limit)
        else:
            while len(history) > limit:
                history.popleft()

    @property
    def logging(self) -> bool:
        with self._condition:
            return self._log_file is not None

    def start_logging(self, path: Path) -> Path:
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
        with self._condition:
            if self._log_file is None:
                return None
            stream, self._log_file = self._log_file, None
            self._log_writer = None
            stream.close()
            return self._log_path

    def take_logging_error(self) -> str | None:
        with self._condition:
            error, self._log_error = self._log_error, None
            return error

    def _write_log(self, snapshot, now):
        if self._log_file is None or not snapshot.measurements_valid:
            return
        stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
        elapsed = f"{now - self._log_started:.3f}"
        series = [
            (f"scene.{name.lower()}", name, value)
            for name, value in zip(("Minimum", "Average", "Maximum", "Center"), snapshot.stats)
        ]
        series.extend(
            (f"spot.{key[0]}.{key[1] + 1}", spot_title(snapshot, key), value)
            for key, value in snapshot.spots
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
        with self._condition:
            first = self._latest is None
            self._latest = replace(snapshot, history_range=self._history_range())
            if first:
                self._condition.notify()

    def pause(self) -> None:
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before hiding the graphs")
            if self._latest is not None:
                self._history_paused = True
                self._latest = None
                self._condition.notify()

    def image(self, size: tuple[int, int], *, resize: bool = False) -> np.ndarray:
        """Return the cached pane; captures can resize it without resampling history."""
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
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()
        self.stop_logging()

    def _sample(self, snapshot, now):
        values = snapshot.stats if snapshot.measurements_valid else (float("nan"),) * 4
        self._master.append((now, values))
        self._compact(self._master)
        current = dict(snapshot.spots)
        self._spots = {key: history for key, history in self._spots.items() if key in current}
        for key, value in snapshot.spots:
            history = self._spots.setdefault(key, deque())
            history.append((now, (value if snapshot.measurements_valid else float("nan"),)))
            self._compact(history)

    def _run(self):
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
                        history.append((now, (float("nan"),)))
                    self._history_paused = False
                if sample_due:
                    self._sample(snapshot, now)
                    self._write_log(snapshot, now)
                generation = self._history_generation
                master = tuple(self._master)
                spots = {key: tuple(history) for key, history in self._spots.items()}
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


def render_graphs(snapshot, master, spots, now):
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
    charts = [("Master stats", master, ("Min", "Avg", "Max", "Center"))]
    charts.extend(
        (spot_title(snapshot, key), spots.get(key, ()), ("Temp",)) for key, _value in snapshot.spots
    )
    top = min(34, height)
    for index, (title, history, labels) in enumerate(charts):
        y0 = top + round((height - top) * index / len(charts))
        y1 = top + round((height - top) * (index + 1) / len(charts))
        _draw_chart(canvas[y0:y1], title, history, labels, snapshot.unit, now, duration)
    return canvas


def _draw_chart(canvas, title, history, labels, unit, now, duration=HISTORY_SECONDS):
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
    samples = [(stamp, values) for stamp, values in history if stamp >= now - duration]
    if not samples:
        return
    stamps = np.array([stamp for stamp, _ in samples])
    values = np.array([value for _, value in samples], dtype=np.float64)
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
    for column in range(values.shape[1]):
        run = []
        for stamp, value in zip(stamps, values[:, column]):
            if not np.isfinite(value):
                if len(run) > 1:
                    cv2.polylines(
                        canvas, [np.array(run, np.int32)], False, COLORS[column], 1, cv2.LINE_AA
                    )
                run = []
                continue
            point = (
                round(left + (stamp - now + duration) / duration * (right - left)),
                round(bottom - (value - low) / (high - low) * (bottom - top)),
            )
            run.append(point)
            cv2.circle(canvas, point, 1, COLORS[column], -1, cv2.LINE_AA)
        if len(run) > 1:
            cv2.polylines(canvas, [np.array(run, np.int32)], False, COLORS[column], 1, cv2.LINE_AA)
    cv2.putText(
        canvas, f"-{history_duration(duration)}", (left, height - 7), font, 0.32, (155, 160, 175), 1
    )
    cv2.putText(canvas, "now", (max(left, right - 25), height - 7), font, 0.32, (155, 160, 175), 1)


def graph_log_button_rect(width):
    button_width = min(150, max(90, width // 3))
    return max(0, width - button_width - 6), 4, width - 6, 28


def graph_interval_rect(width):
    left = graph_log_button_rect(width)[0]
    return max(0, left - 88), 4, max(0, left - 8), 28


def graph_config_rect(width):
    left = graph_interval_rect(width)[0]
    return max(0, left - 152), 4, max(0, left - 84), 28


def graph_reset_rect(width):
    left = graph_config_rect(width)[0]
    return max(0, left - 68), 4, max(0, left - 4), 28


def draw_graph_logging_control(
    canvas, logging=False, pending=False, interval=GRAPH_INTERVAL, edit_text=None
):
    """Overlay the live button on the main thread without mutating the worker's cached image."""
    x0, y0, x1, y1 = graph_log_button_rect(canvas.shape[1])
    label = "Cancel logging" if pending else "Stop logging" if logging else "Log to CSV"
    color = (65, 65, 150) if logging else (47, 51, 61)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_width = cv2.getTextSize(label, font, 0.4, 1)[0][0]
    scale = 0.4 * min(1, max(1, x1 - x0 - 10) / text_width)
    cv2.putText(canvas, label, (x0 + 5, y0 + 16), font, scale, (235, 238, 245), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = graph_config_rect(canvas.shape[1])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (47, 51, 61), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
    cv2.putText(canvas, "Config", (x0 + 5, y0 + 16), font, 0.4, (235, 238, 245), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = graph_reset_rect(canvas.shape[1])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (47, 51, 61), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (105, 112, 128), 1)
    cv2.putText(
        canvas,
        "Reset",
        (x0 + 5, y0 + 16),
        font,
        0.4,
        (125, 130, 140) if logging or pending else (235, 238, 245),
        1,
        cv2.LINE_AA,
    )
    x0, y0, x1, y1 = graph_interval_rect(canvas.shape[1])
    border = (90, 190, 240) if edit_text is not None else (105, 112, 128)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (35, 38, 46), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 1)
    value = f"{interval:g}" if edit_text is None else edit_text + "|"
    cv2.putText(
        canvas,
        value,
        (x0 + 5, y0 + 16),
        font,
        0.4,
        (125, 130, 140) if logging or pending else (235, 238, 245),
        1,
        cv2.LINE_AA,
    )
    if x0 >= 85:
        cv2.putText(
            canvas, "Update (s)", (x0 - 80, y0 + 16), font, 0.4, (185, 190, 205), 1, cv2.LINE_AA
        )
