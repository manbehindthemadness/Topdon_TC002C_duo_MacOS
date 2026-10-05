"""Bounded temperature history and graph rendering on a dedicated worker thread."""

import csv
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

GRAPH_INTERVAL = 0.5
HISTORY_SECONDS = 60
HISTORY_SAMPLES = round(HISTORY_SECONDS / GRAPH_INTERVAL) + 1
COLORS = ((230, 160, 80), (90, 210, 110), (90, 110, 245), (220, 190, 240))


@dataclass(frozen=True)
class GraphSnapshot:
    stats: tuple[float, float, float, float]
    spots: tuple[tuple[tuple[int, int], float], ...]
    size: tuple[int, int]  # width, height of the reserved graph area
    unit: str = "C"
    spot_names: tuple[tuple[tuple[int, int], str], ...] = ()
    measurements_valid: bool = True


def spot_title(snapshot, key):
    default = f"Spot {key[1] + 1}"
    name = dict(snapshot.spot_names).get(key, default)
    return default if name == default else f"{name} ({default})"


class GraphWorker:
    """Sample the latest scalar measurements every 0.5 s; never touch USB or GUI windows."""

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
        self._master = deque(maxlen=HISTORY_SAMPLES)
        self._spots = {}
        self._thread = threading.Thread(target=self._run, name="temperature-graphs", daemon=True)
        self._thread.start()

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
            self._latest = snapshot
            if first:
                self._condition.notify()

    def pause(self) -> None:
        with self._condition:
            if self._log_file is not None:
                raise ValueError("Stop logging before hiding the graphs")
            if self._latest is not None:
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
        current = dict(snapshot.spots)
        self._spots = {key: history for key, history in self._spots.items() if key in current}
        for key, value in snapshot.spots:
            history = self._spots.setdefault(key, deque(maxlen=HISTORY_SAMPLES))
            history.append((now, (value if snapshot.measurements_valid else float("nan"),)))

    def _run(self):
        deadline = 0.0
        while True:
            with self._condition:
                while not self._closed:
                    remaining = deadline - time.monotonic()
                    if self._latest is not None and remaining <= 0:
                        break
                    self._condition.wait(timeout=max(0, remaining) if self._latest else None)
                if self._closed:
                    return
                snapshot = self._latest
                now = time.monotonic()
                self._sample(snapshot, now)
                self._write_log(snapshot, now)
            result = render_graphs(snapshot, self._master, self._spots, now)
            with self._condition:
                if self._latest is not None and not self._closed:
                    self._image = result
                    self._condition.notify_all()
            # No queued frames or catch-up bursts if rendering takes longer than the interval.
            deadline = max(now + GRAPH_INTERVAL, time.monotonic())


def render_graphs(snapshot, master, spots, now):
    width, height = snapshot.size
    canvas = np.full((height, width, 3), (23, 25, 31), np.uint8)
    cv2.putText(
        canvas,
        "History | 60 s | 0.5 s updates",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        min(0.45, max(0.15, (graph_log_button_rect(width)[0] - 16) / 370)),
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
        _draw_chart(canvas[y0:y1], title, history, labels, snapshot.unit, now)
    return canvas


def _draw_chart(canvas, title, history, labels, unit, now):
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
    samples = [(stamp, values) for stamp, values in history if stamp >= now - HISTORY_SECONDS]
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
        previous_stamp = None
        for stamp, value in zip(stamps, values[:, column]):
            if previous_stamp is not None and stamp - previous_stamp > GRAPH_INTERVAL * 2:
                if len(run) > 1:
                    cv2.polylines(
                        canvas, [np.array(run, np.int32)], False, COLORS[column], 1, cv2.LINE_AA
                    )
                run = []
            previous_stamp = stamp
            if not np.isfinite(value):
                if len(run) > 1:
                    cv2.polylines(
                        canvas, [np.array(run, np.int32)], False, COLORS[column], 1, cv2.LINE_AA
                    )
                run = []
                continue
            point = (
                round(left + (stamp - now + HISTORY_SECONDS) / HISTORY_SECONDS * (right - left)),
                round(bottom - (value - low) / (high - low) * (bottom - top)),
            )
            run.append(point)
            cv2.circle(canvas, point, 1, COLORS[column], -1, cv2.LINE_AA)
        if len(run) > 1:
            cv2.polylines(canvas, [np.array(run, np.int32)], False, COLORS[column], 1, cv2.LINE_AA)
    cv2.putText(canvas, "-60 s", (left, height - 7), font, 0.32, (155, 160, 175), 1)
    cv2.putText(canvas, "now", (max(left, right - 25), height - 7), font, 0.32, (155, 160, 175), 1)


def graph_log_button_rect(width):
    button_width = min(150, max(90, width // 3))
    return max(0, width - button_width - 6), 4, width - 6, 28


def draw_graph_logging_control(canvas, logging=False, pending=False):
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
