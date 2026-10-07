"""CPU feature selection on display images, with bounded detection resolution."""

import cv2
import numpy as np

COLORS = {
    "white": (255, 255, 255),
    "cyan": (255, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "black": (0, 0, 0),
}
MAX_CANDIDATES = 2048


def feature_fields(kind, p):
    fields = set(p)
    if p["output"] in ("mask", "cutout", "mean"):
        fields.discard("color")
    if p["region"] == "full":
        fields.discard("region_size")
    if kind == "edges":
        if p["method"] != "canny":
            fields -= {"auto_threshold", "lower", "upper", "l2"}
        elif p["auto_threshold"]:
            fields -= {"lower", "upper", "strength"}
        else:
            fields.discard("strength")
    else:
        if p["segmentation"] != "threshold":
            fields.discard("threshold")
        if p["segmentation"] != "canny":
            fields -= {"lower", "upper"}
        else:
            fields.discard("invert")
        if p["segmentation"] != "adaptive":
            fields -= {"block_size", "adaptive_c"}
        if p["output"] in ("mask", "cutout", "mean"):
            fields.discard("thickness")
    return fields


def validate_feature(kind, p):
    uses_canny = (
        p["method"] == "canny" and not p["auto_threshold"]
        if kind == "edges"
        else p["segmentation"] == "canny"
    )
    if uses_canny and p["lower"] >= p["upper"]:
        raise ValueError("Edge lower threshold must be below upper threshold")
    if kind == "contours":
        if p["min_area"] > p["max_area"]:
            raise ValueError("Minimum region area must not exceed maximum")
        if p["min_aspect"] > p["max_aspect"]:
            raise ValueError("Minimum aspect ratio must not exceed maximum")


def _prepare(image, p):
    height, width = image.shape[:2]
    factor = min(1.0, p["resolution"] / max(height, width))
    small = (
        cv2.resize(
            image,
            (max(1, round(width * factor)), max(1, round(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
        if factor < 1
        else image
    )
    small = np.clip(small, 0, 255).round().astype(np.uint8)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if p["blur"]:
        gray = cv2.GaussianBlur(gray, (int(p["blur"]), int(p["blur"])), 0)
    roi = np.full(gray.shape, 255, np.uint8)
    if p["region"] == "center":
        h, w = gray.shape
        rh, rw = (
            max(1, round(h * p["region_size"] / 100)),
            max(1, round(w * p["region_size"] / 100)),
        )
        roi[:] = 0
        y, x = (h - rh) // 2, (w - rw) // 2
        roi[y : y + rh, x : x + rw] = 255
    return small, gray, roi


def _canny(gray, low, high, l2=True):
    return cv2.Canny(gray, low, high, apertureSize=3, L2gradient=l2)


def _edge_mask(gray, p):
    gx = (
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        if p["method"] != "scharr"
        else cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    )
    gy = (
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        if p["method"] != "scharr"
        else cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    )
    magnitude = cv2.magnitude(gx, gy)
    if p["method"] == "canny":
        low, high = p["lower"], p["upper"]
        if p["auto_threshold"]:
            nonzero = magnitude[magnitude > 0]
            high = max(1.0, 0.75 * float(np.percentile(nonzero, 75))) if nonzero.size else 1.0
            low = high * 0.4
        mask = _canny(gray, low, high, p["l2"])
    else:
        scaled = magnitude / (16 if p["method"] == "scharr" else 4)
        mask = np.where(scaled >= max(0.001, p["strength"]), 255, 0).astype(np.uint8)
    if p["direction"] == "horizontal":
        mask[np.abs(gy) < 1.5 * np.abs(gx)] = 0
    elif p["direction"] == "vertical":
        mask[np.abs(gx) < 1.5 * np.abs(gy)] = 0
    return mask


def _select_edges(mask, p):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ids = np.arange(1, count)
    ids = ids[stats[ids, cv2.CC_STAT_AREA] >= p["min_pixels"]]
    if p["exclude_border"]:
        s = stats[ids]
        h, w = mask.shape
        ids = ids[(s[:, 0] > 0) & (s[:, 1] > 0) & (s[:, 0] + s[:, 2] < w) & (s[:, 1] + s[:, 3] < h)]
    if p["rank"] == "center":
        h, w = mask.shape
        score = np.sum((centers[ids] - [(w - 1) / 2, (h - 1) / 2]) ** 2, axis=1)
    else:
        score = -stats[ids, cv2.CC_STAT_AREA]
    ids = ids[np.argsort(score, kind="stable")[: int(p["max_count"])]]
    lookup = np.zeros(count, np.uint8)
    lookup[ids] = 255
    selected = lookup[labels]
    if p["thickness"] > 1:
        selected = cv2.dilate(
            selected,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (int(p["thickness"]), int(p["thickness"]))
            ),
        )
    return selected


def _segment(gray, p):
    mode = p["segmentation"]
    if mode == "canny":
        mask = _canny(gray, p["lower"], p["upper"])
    elif mode == "adaptive":
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if p["invert"] else cv2.THRESH_BINARY,
            int(p["block_size"]),
            p["adaptive_c"],
        )
    else:
        code = cv2.THRESH_BINARY_INV if p["invert"] else cv2.THRESH_BINARY
        if mode == "otsu":
            code |= cv2.THRESH_OTSU
        mask = cv2.threshold(gray, p["threshold"], 255, code)[1]
    if p["close_kernel"]:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (int(p["close_kernel"]), int(p["close_kernel"]))
            ),
        )
    return mask


def _region_mean(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [(contour - [x, y]).astype(np.int32)], -1, 255, cv2.FILLED)
    return cv2.mean(image[y : y + h, x : x + w], mask)[:3]


def _contours(small, gray, mask, p):
    detected, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = sorted(((cv2.contourArea(c), c) for c in detected), key=lambda pair: -pair[0])[
        :MAX_CANDIDATES
    ]
    h, w = gray.shape
    selected = []
    for area, contour in candidates:
        if not p["min_area"] <= 100 * area / (w * h) <= p["max_area"] or area <= 0:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if p["exclude_border"] and (x <= 0 or y <= 0 or x + cw >= w or y + ch >= h):
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) <= 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if not p["min_aspect"] <= aspect <= p["max_aspect"]:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(contour, True)
        if not hull_area or not perimeter:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area / hull_area < p["solidity"] or circularity < p["circularity"]:
            continue
        polygon = cv2.approxPolyDP(contour, p["simplify"] * perimeter / 100, True)
        shape = p["shape"]
        if shape == "rectangle":
            if len(polygon) != 4 or not cv2.isContourConvex(polygon) or area / (rw * rh) < 0.75:
                continue
            vertices = polygon[:, 0, :].astype(np.float32)
            a = np.roll(vertices, 1, axis=0) - vertices
            b = np.roll(vertices, -1, axis=0) - vertices
            cosines = np.abs(np.sum(a * b, axis=1)) / np.maximum(
                1e-6, np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            )
            if np.max(cosines) > 0.35:
                continue
        if shape == "circle" and (len(polygon) < 6 or circularity < 0.75 or aspect > 1.3):
            continue
        if shape == "convex" and not cv2.isContourConvex(polygon):
            continue
        moments = cv2.moments(contour)
        cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        mean = (
            _region_mean(small, contour)
            if p["rank"] in ("brightest", "darkest") or p["output"] == "mean"
            else (0, 0, 0)
        )
        brightness = 0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2]
        score = {
            "largest": -area,
            "smallest": area,
            "brightest": -brightness,
            "darkest": brightness,
            "center": (cx - (w - 1) / 2) ** 2 + (cy - (h - 1) / 2) ** 2,
        }[p["rank"]]
        geometry = {
            "contour": contour,
            "polygon": polygon,
            "hull": hull,
            "box": cv2.boxPoints(rect).round().astype(np.int32).reshape(-1, 1, 2),
        }[p["geometry"]]
        selected.append((score, geometry, mean))
    return sorted(selected, key=lambda candidate: candidate[0])[: int(p["max_count"])]


def _render(image, mask, p, decorated=None):
    if p["output"] == "mask":
        filtered = np.repeat(mask[..., None], 3, axis=2).astype(np.float32)
    elif p["output"] == "cutout":
        filtered = image * (mask[..., None] > 0)
    else:
        filtered = image.copy() if decorated is None else decorated
        if decorated is None:
            filtered[mask > 0] = COLORS[p["color"]]
    if p["mix"] == 1:
        return filtered.astype(np.float32, copy=False)
    return cv2.addWeighted(
        image.astype(np.float32), 1 - p["mix"], filtered.astype(np.float32), p["mix"], 0
    )


def apply_features(image, kind, p):
    if not p["mix"]:
        return image
    small, gray, roi = _prepare(image, p)
    height, width = image.shape[:2]
    if kind == "edges":
        mask = _select_edges(cv2.bitwise_and(_edge_mask(gray, p), roi), p)
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return _render(image, mask, p)
    if kind != "contours":
        raise ValueError("Unknown feature processor")
    selected = _contours(small, gray, cv2.bitwise_and(_segment(gray, p), roi), p)
    mask = np.zeros((height, width), np.uint8)
    decorated = image.copy() if p["output"] in ("overlay", "fill", "mean") else None
    scale = np.array([width / gray.shape[1], height / gray.shape[0]])
    for _, contour, mean in selected:
        points = ((contour + 0.5) * scale - 0.5).round().astype(np.int32)
        points[..., 0] = points[..., 0].clip(0, width - 1)
        points[..., 1] = points[..., 1].clip(0, height - 1)
        if p["output"] in ("mask", "cutout"):
            cv2.drawContours(mask, [points], -1, 255, cv2.FILLED)
        if p["output"] == "mean":
            cv2.drawContours(decorated, [points], -1, mean, cv2.FILLED)
        elif p["output"] in ("overlay", "fill"):
            cv2.drawContours(
                decorated,
                [points],
                -1,
                COLORS[p["color"]],
                cv2.FILLED if p["output"] == "fill" else int(p["thickness"]),
                lineType=cv2.LINE_AA,
            )
    return _render(image, mask, p, decorated)
