"""
Desktop command-line parsing and defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import APP_NAME
from ..camera import (
    DEFAULT_USB_QUEUE_DEPTH,
)
from .constants import (
    TIMELAPSE_MAX_FPM,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse desktop options without coupling the application identity to one camera backend.
    """
    parser = argparse.ArgumentParser(description=f"{APP_NAME} native thermal-camera viewer")
    parser.add_argument("--camera", default="duo", help="Camera backend: duo, virtual-dual, or a registered backend (default: duo)")
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
        choices=("preview", "raw", "analyze", "visible"),
        default=None,
        help="thermal upsample (default), raw thermal visualization, or experimental visible light",
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
    if args.timelapse_fpm is not None and (
        args.timelapse_fpm < 1 or args.camera == "duo" and args.timelapse_fpm > TIMELAPSE_MAX_FPM
    ):
        parser.error(f"--timelapse-fpm must be between 1 and {TIMELAPSE_MAX_FPM} for the Duo")
    return args
