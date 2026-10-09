"""
Desktop command-line parsing and defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..camera import (
    DEFAULT_USB_QUEUE_DEPTH,
)
from .constants import (
    TIMELAPSE_MAX_FPM,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse args.
    """
    parser = argparse.ArgumentParser(description="Native TOPDON TC002C Duo viewer")
    parser.add_argument("--ambient", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=None)
    parser.add_argument(
        "--scale",
        type=int,
        choices=range(1, 7),
        default=4,
        help="native sensor display scale (default: 4, 1024x768 landscape image)",
    )
    parser.add_argument(
        "--reset-window-size",
        action="store_true",
        help="ignore the remembered viewer size on this launch; preserve image aspect ratio",
    )
    parser.add_argument(
        "--image-source",
        choices=("preview", "raw", "analyze"),
        default=None,
        help="camera preview (default, when available) or raw thermal visualization",
    )
    parser.add_argument(
        "--timelapse-fpm",
        type=int,
        default=None,
        help="timelapse frames per minute (default: remembered rate, or 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="initial directory for the Save dialog",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--diagnostics", type=Path, help="Write UI timings and stall thread stacks")
    parser.add_argument(
        "--usb-queue-depth",
        type=int,
        choices=(0, *range(2, 129)),
        default=(
            DEFAULT_USB_QUEUE_DEPTH
            if sys.platform == "darwin" or sys.platform.startswith("linux")
            else 0
        ),
        help="Queued USB requests (2-128; 0: synchronous; default: 32 on macOS/Linux)",
    )
    args = parser.parse_args(argv)
    if args.timelapse_fpm is not None and not 1 <= args.timelapse_fpm <= TIMELAPSE_MAX_FPM:
        parser.error(f"--timelapse-fpm must be between 1 and {TIMELAPSE_MAX_FPM}")
    return args
