"""Approximate face-on square ranging in native thermal-image coordinates."""

import math
from dataclasses import asdict, dataclass
from itertools import combinations

import cv2
import numpy as np

from .camera import SENSOR_HEIGHT, SENSOR_WIDTH


def _edge_squares(thermal):
    """Recover square outlines from two pairs of perpendicular parallel edges."""
    height, width = thermal.shape
    low, high = np.percentile(thermal, (2, 98))
    image = np.uint8(np.clip((thermal - low) * 255 / max(high - low, 1), 0, 255))
    edges = cv2.Canny(image, 25, 70)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=12, maxLineGap=4)
    if lines is None:
        return []
    lines = lines.reshape(-1, 4).astype(float)
    lengths = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1)
    lines = lines[np.argsort(lengths)[::-1][:40]]
    directions = lines[:, 2:] - lines[:, :2]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    pairs = []
    for first, second in combinations(range(len(lines)), 2):
        direction = directions[first]
        if abs(np.dot(direction, directions[second])) < 0.996:
            continue
        normal = np.array([-direction[1], direction[0]])
        distance = abs(np.dot(lines[second, :2] - lines[first, :2], normal))
        if 12 <= distance <= min(height, width):
            pairs.append((first, second, distance))
    distances = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    polygons = []
    for pair_a, pair_b in combinations(pairs, 2):
        if abs(np.dot(directions[pair_a[0]], directions[pair_b[0]])) > 0.15:
            continue
        if max(pair_a[2], pair_b[2]) / min(pair_a[2], pair_b[2]) > 1.15:
            continue
        points = []
        for a, b in (
            (pair_a[0], pair_b[0]),
            (pair_a[0], pair_b[1]),
            (pair_a[1], pair_b[1]),
            (pair_a[1], pair_b[0]),
        ):
            matrix = np.column_stack((directions[a], -directions[b]))
            parameter = np.linalg.solve(matrix, lines[b, :2] - lines[a, :2])[0]
            points.append(lines[a, :2] + parameter * directions[a])
        points = np.array(points)
        try:
            square_edge_pixels(points, (width, height))
        except ValueError:
            continue
        # Infinite line intersections need real edge support across each side.
        supported = True
        for first, last in zip(points, np.roll(points, -1, axis=0), strict=True):
            samples = np.rint(first + np.linspace(0.1, 0.9, 16)[:, None] * (last - first)).astype(
                int
            )
            if np.mean(distances[samples[:, 1], samples[:, 0]] <= 1.5) < 0.8:
                supported = False
                break
        if supported:
            polygons.append(points.reshape(4, 1, 2))
    return polygons


def detect_hand_square(temperatures):
    """Find a cooler square with a warm surrounding hand; ambiguous scenes return None.

    All geometry uses the native Celsius plane. Temperature limits are initial
    hand-background heuristics, not a human detector or emissivity measurement.
    """
    thermal = np.asarray(temperatures, dtype=np.float32)
    if thermal.ndim != 2 or not np.isfinite(thermal).all():
        return None
    height, width = thermal.shape
    if min(height, width) < 16:
        return None
    smooth = cv2.GaussianBlur(thermal, (3, 3), 0.6)
    low, high = np.percentile(smooth, (2, 98))
    if high - low < 1:
        return None
    # Sweep temperatures for closed contours and recover fragmented outlines
    # from straight edges when the note has unevenly warmed against the hand.
    polygons = []
    for threshold in np.linspace(low + 0.5, high - 0.5, 16):
        mask = np.uint8(smooth > threshold) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 144 or area > thermal.size * 0.45:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            if abs(area - cv2.contourArea(polygon)) / area > 0.08:
                continue
            polygons.append(polygon)
    polygons.extend(_edge_squares(smooth))
    candidates = []
    for polygon in polygons:
        points = polygon[:, 0].astype(float)
        try:
            edge = square_edge_pixels(points, (width, height))
        except ValueError:
            continue
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        if lengths.max() / lengths.min() > 1.08:
            continue
        if (
            (points < 4).any()
            or (points[:, 0] >= width - 4).any()
            or (points[:, 1] >= height - 4).any()
        ):
            continue
        points = cv2.cornerSubPix(
            smooth,
            points.astype(np.float32).reshape(4, 1, 2),
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )[:, 0].astype(float)
        try:
            edge = square_edge_pixels(points, (width, height))
        except ValueError:
            continue
        # Sample inside and outside each edge, away from rounded corners.
        center = points.mean(axis=0)
        side_support, inside, outside = [], [], []
        margin = max(2, edge * 0.1)
        for first, last in zip(points, np.roll(points, -1, axis=0), strict=True):
            along = last - first
            normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
            if np.dot(normal, center - (first + last) / 2) < 0:
                normal = -normal
            samples = first + np.linspace(0.2, 0.8, 12)[:, None] * along
            inner = np.rint(samples + normal * margin).astype(int)
            outer = np.rint(samples - normal * max(2, edge * 0.04)).astype(int)
            if (outer < 0).any() or (outer[:, 0] >= width).any() or (outer[:, 1] >= height).any():
                break
            inner_values = smooth[inner[:, 1], inner[:, 0]]
            outer_values = smooth[outer[:, 1], outer[:, 0]]
            warm = (outer_values >= 24) & (outer_values <= 40)
            support = warm & (outer_values - inner_values >= 0.75)
            side_support.append(float(support.mean()))
            inside.extend(inner_values)
            outside.extend(outer_values)
        if len(side_support) != 4 or sum(value >= 0.6 for value in side_support) < 3:
            continue
        if np.mean(side_support) < 0.65:
            continue
        contrast = float(np.median(outside) - np.median(inside))
        if contrast < 0.75:
            continue
        score = float(np.mean(side_support)) * min(contrast, 8)
        # Canonical cyclic order keeps repeated frames' corners aligned.
        if cv2.contourArea(points.astype(np.float32), oriented=True) < 0:
            points = points[::-1]
        points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
        duplicate = next(
            (
                i
                for i, (_, previous) in enumerate(candidates)
                if np.linalg.norm(previous.mean(axis=0) - center) < max(3, edge * 0.1)
                and abs(square_edge_pixels(previous, (width, height)) - edge) < max(3, edge * 0.1)
            ),
            None,
        )
        if duplicate is None:
            candidates.append((score, points))
        elif score > candidates[duplicate][0]:
            candidates[duplicate] = (score, points)
    if len(candidates) != 1:
        return None
    return candidates[0][1]


def square_edge_pixels(corners, size):
    points = np.asarray(corners, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Select four square corners in clockwise or counterclockwise order")
    width, height = size
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("All four corners must lie inside the thermal image")
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 12:
        raise ValueError("Square is too small: each edge must span at least 12 native pixels")
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (np.all(turns > 0) or np.all(turns < 0)):
        raise ValueError("Corners must follow the square boundary without crossing")
    if lengths.max() / lengths.min() > 1.15:
        raise ValueError("Square looks tilted or uneven; face it toward the camera and reselect")
    adjacent_cosines = np.sum(edges * np.roll(edges, -1, axis=0), axis=1) / (
        lengths * np.roll(lengths, -1)
    )
    if np.abs(adjacent_cosines).max() > 0.15:
        raise ValueError("Corners do not form a face-on square; check alignment and selection")
    return float(lengths.mean())


@dataclass(frozen=True)
class DistanceReference:
    side_m: float
    distance_m: float
    edge_pixels: float

    def __post_init__(self):
        for value in (self.side_m, self.distance_m, self.edge_pixels):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Calibration values must be finite numbers")
        if not 0.0001 <= self.side_m <= 10:
            raise ValueError("Enter the square side length between 0.0001 and 10 meters")
        if not 0.3 <= self.distance_m <= 99:
            raise ValueError("Reference distance must be between 0.3 and 99 meters")
        if not 12 <= self.edge_pixels <= min(SENSOR_WIDTH, SENSOR_HEIGHT):
            raise ValueError("Invalid reference square size in native pixels")

    @property
    def focal_pixels(self):
        return self.edge_pixels * self.distance_m / self.side_m

    def estimate(self, corners, size):
        return self.focal_pixels * self.side_m / square_edge_pixels(corners, size)

    def as_dict(self):
        return {"version": 1, "native_size": [SENSOR_WIDTH, SENSOR_HEIGHT], **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value.get("version") != 1
            or value.get("native_size") != [SENSOR_WIDTH, SENSOR_HEIGHT]
        ):
            raise ValueError("Unsupported distance calibration profile")
        try:
            return cls(value["side_m"], value["distance_m"], value["edge_pixels"])
        except KeyError as exc:
            raise ValueError("Incomplete distance calibration profile") from exc


class DistanceCalibrator:
    STABLE_FRAMES = 8

    def __init__(self, reference=None):
        self.reference = DistanceReference.from_dict(reference) if reference else None
        self.corners = []
        self.selecting = None
        self.edge_pixels = None
        self.estimated_m = None
        self._detections = []
        self.message = "No reference yet. Hold a cooler Post-it in front of your palm."

    def begin(self, mode):
        if mode not in ("reference", "measure"):
            raise ValueError("Unknown square selection mode")
        if mode == "measure" and self.reference is None:
            raise ValueError("Save a reference calibration first")
        self.corners = []
        self.edge_pixels = self.estimated_m = None
        self.selecting = mode
        self._detections = []
        self.message = (
            "Looking for a cooler square against your warm palm. Keep it flat and steady."
        )

    def cancel(self):
        self.corners = []
        self.edge_pixels = self.estimated_m = None
        self.selecting = None
        self._detections = []
        self.message = (
            "Selection cancelled. Saved reference retained."
            if self.reference
            else "Selection cancelled."
        )

    def update(self, temperatures):
        if self.selecting is None:
            return False
        detected = detect_hand_square(temperatures)
        if detected is None:
            self.corners = []
            self._detections = []
            self.message = (
                "No unique square detected. Show the cooler note with warm skin around its edges."
            )
            return False
        if self._detections:
            previous = self._detections[-1]
            detected = min(
                (np.roll(detected, offset, axis=0) for offset in range(4)),
                key=lambda points: np.linalg.norm(points - previous),
            )
            if (
                np.max(np.linalg.norm(detected - previous, axis=1)) > 2
                or np.max(np.linalg.norm(detected - self._detections[0], axis=1)) > 2
            ):
                self._detections = []
        self._detections.append(detected)
        self.corners = detected.tolist()
        self.message = (
            f"Square detected. Hold steady ({len(self._detections)}/{self.STABLE_FRAMES})."
        )
        if len(self._detections) < self.STABLE_FRAMES:
            return False
        self.corners = np.median(self._detections, axis=0).tolist()
        size = temperatures.shape[::-1]
        try:
            self.edge_pixels = square_edge_pixels(self.corners, size)
        except ValueError:
            self._detections = []
            self.corners = []
            self.message = "Outline changed. Hold the note flat and steady for another detection."
            return False
        mode, self.selecting = self.selecting, None
        self._detections = []
        if mode == "measure":
            self.estimated_m = self.reference.estimate(self.corners, size)
            self.message = "Estimate ready. Validate against a measured distance before applying."
        else:
            self.message = (
                "Square captured. Enter its side length and measured reference distance, then save."
            )
        return True

    def save_reference(self, side_m, distance_m):
        if self.selecting or self.edge_pixels is None:
            raise ValueError("Detect a steady face-on reference square first")
        self.reference = DistanceReference(side_m, distance_m, self.edge_pixels)
        self.estimated_m = None
        self.corners = []
        self.message = (
            "Reference saved. Move the same square, keep it face-on, then Measure square."
        )

    def clear(self):
        self.cancel()
        self.reference = None
        self.message = "Reference cleared. Camera distance setting is unchanged."

    def state(self):
        return {
            "reference": self.reference.as_dict() if self.reference else None,
            "selecting": self.selecting,
            "corners": len(self.corners),
            "ready": self.edge_pixels is not None and self.selecting is None,
            "estimated_m": self.estimated_m,
            "status": self.message,
            "stable_frames": len(self._detections),
        }
