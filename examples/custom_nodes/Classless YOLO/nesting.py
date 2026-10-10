"""
Select innermost detections or retain parent/child box hierarchies.
"""

from .detector import Detection


def area(box: Detection) -> float:
    """
    Compute normalized box area for strict containment and painting order.
    """
    y1, x1, y2, x2, _ = box
    return (y2 - y1) * (x2 - x1)


def overlap(first: Detection, second: Detection) -> float:
    """
    Compute intersection area without treating crossing edges as full containment.
    """
    height = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    width = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return height * width


def contains(parent: Detection, child: Detection, coverage: float = 0.85) -> bool:
    """
    Require meaningful size difference and configured coverage, ignoring peer jitter.
    """
    child_area = area(child)
    return (child_area > 0 and area(parent) > child_area * 1.05
            and overlap(parent, child) / child_area >= coverage)


def suppress_duplicates(
    detections: list[Detection], threshold: float, coverage: float
) -> list[Detection]:
    """
    Keep the strongest overlapping peer; preserve distinct parent/child relationships.
    """
    kept: list[Detection] = []
    for box in sorted(detections, key=lambda candidate: candidate[4], reverse=True):
        duplicate = False
        for other in kept:
            intersection = overlap(box, other)
            union = area(box) + area(other) - intersection
            nested = contains(box, other, coverage) or contains(other, box, coverage)
            if not nested and union > 0 and intersection / union >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def select_detections(
    detections: list[Detection], mode: str, maximum: int,
    coverage: float = 0.85, duplicate_iou: float | None = 0.5,
) -> list[Detection]:
    """
    Filter parents and duplicate peers before limiting and painting outside-in.
    """
    selected = detections
    if mode == "Inside-out":
        selected = [box for box in detections
                    if not any(contains(box, child, coverage) for child in detections)]
    if duplicate_iou is not None:
        selected = suppress_duplicates(selected, duplicate_iou, coverage)
    selected = sorted(selected, key=lambda box: box[4], reverse=True)[:maximum]
    return sorted(selected, key=area, reverse=True)
