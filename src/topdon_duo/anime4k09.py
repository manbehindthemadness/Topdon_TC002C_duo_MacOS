"""Portable NumPy/OpenCV port of Anime4KCPP's Anime4K09 OpenCL edge passes.

Derived from Anime4KCPP v2.5.0 Anime4KCore/kernel/Anime4KCPPKernel.cl.
Copyright (c) 2020 TianZer; MIT license in models/LICENSE-ANIME4K09.txt.
Phone parameters: 2x linear resize, 3 passes, gradient strength 0.5, color push 0.
"""

import cv2
import numpy as np


def _neighbors(plane: np.ndarray) -> tuple[np.ndarray, ...]:
    h, w = plane.shape[:2]
    padding = ((1, 1), (1, 1)) + (((0, 0),) if plane.ndim == 3 else ())
    padded = np.pad(plane, padding, mode="edge")
    return tuple(padded[y : y + h, x : x + w] for y in range(3) for x in range(3))


def _gray(color: np.ndarray) -> np.ndarray:
    return cv2.transform(color, np.array([[0.114, 0.587, 0.299]], dtype=np.float32))


def _quantize(value: np.ndarray) -> np.ndarray:
    # The upstream byte-image OpenCL path uses UNORM8 intermediate buffers.
    return np.clip(value, 0, 255).round()


def upscale_anime4k09(image: np.ndarray, passes: int = 3, strength: float = 0.5) -> np.ndarray:
    grayscale = image.ndim == 2
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if grayscale else image
    color = cv2.resize(color.astype(np.float32), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = _quantize(_gray(color))
    color = _quantize(color)
    for _ in range(passes):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
        gradient = _quantize(255 - np.minimum(cv2.magnitude(gx, gy), 255))
        g = _neighbors(gradient)
        c = _neighbors(color)
        result = color.copy()
        remaining = np.ones(gray.shape, dtype=bool)
        # Ordered, first-match directional tests from the upstream pushGradient.
        for light, dark, straight in (
            ((0, 1, 2), (6, 7, 8), True),
            ((6, 7, 8), (0, 1, 2), True),
            ((1, 2, 5), (3, 4, 7), False),
            ((3, 6, 7), (1, 4, 5), False),
            ((2, 5, 8), (0, 3, 6), True),
            ((0, 3, 6), (2, 5, 8), True),
            ((5, 8, 7), (1, 4, 3), False),
            ((3, 0, 1), (7, 4, 5), False),
        ):
            minimum = np.minimum(np.minimum(g[light[0]], g[light[1]]), g[light[2]])
            maximum = np.maximum(np.maximum(g[dark[0]], g[dark[1]]), g[dark[2]])
            condition = ((minimum > g[4]) & (g[4] > maximum)) if straight else minimum > maximum
            selected = remaining & condition
            if np.any(selected):
                total = cv2.add(cv2.add(c[light[0]], c[light[1]]), c[light[2]])
                updated = cv2.addWeighted(color, 1 - strength, total, strength / 3, 0)
                cv2.copyTo(updated, selected.astype(np.uint8), result)
                remaining[selected] = False
        gray = _quantize(_gray(result))
        color = _quantize(result)
    result = color.astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if grayscale else result
