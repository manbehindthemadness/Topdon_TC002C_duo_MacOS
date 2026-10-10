"""
Compare capture without desktop processing or hardware-settings control traffic.

Close all camera viewers before running this experiment. It opens the real Duo,
performs the application's normal UVC negotiation, and drains complete frames.
It does not apply camera settings. The optional --preview mode adds an OpenCV
image window; --component exercises desktop processing with no AI nodes.
Component construction can perform the application's startup GPU capability probe.
Use the existing project environment and USB permissions; no network is needed.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from topdon_duo.camera import CameraError, TC002CDuoCamera, decode_duo_frame
from topdon_duo.frame_pump import CameraFramePump


class CaptureFinished(Exception):
    """
    End the bounded capture from the stream observer, including during frame loss.
    """


def main() -> None:
    """
    Drain the camera for a bounded interval and persist per-second stream reports.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--preview", action="store_true", help="Show a minimal OpenCV preview")
    parser.add_argument(
        "--component", choices=("none", "renderer", "worker", "both"), default="none",
        help="Add selected real desktop processing components to the preview",
    )
    parser.add_argument(
        "--stop-on-stall", action="store_true", help="Stop after five seconds without valid frames",
    )
    parser.add_argument(
        "--frame-pump", action="store_true",
        help="Use the desktop acquisition thread without UI or hardware-settings traffic",
    )
    parser.add_argument("--usb-queue-depth", type=int, choices=(0, *range(2, 129)), default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/usr/src/codex/scratch/jetson-feed-loss/capture-only.jsonl"),
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.preview and not args.frame_pump:
        parser.error("--preview requires --frame-pump")
    if args.component != "none" and not args.preview:
        parser.error("--component requires --preview")
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = process.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(arg.decode(errors="replace")).name == "topdon-duo-desktop"
               for arg in command if arg):
            parser.error("Close the desktop viewer before opening the camera")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    camera = TC002CDuoCamera()
    camera.usb_queue_depth = args.usb_queue_depth
    components = None
    if args.component != "none":
        from capture_components import CaptureComponents

        components = CaptureComponents(args.component)
    started = time.monotonic()
    last_count = 0
    last_complete_at = None
    if args.preview:
        import cv2

        cv2.namedWindow("Capture diagnostic", cv2.WINDOW_NORMAL)
    with args.output.open("x", encoding="utf-8") as output:
        def observe(report: dict[str, Any]) -> None:
            """
            Save counters and terminate even if no complete frames are delivered.
            """
            nonlocal last_count, last_complete_at
            now = time.monotonic()
            elapsed = now - started
            if report["frames"] > last_count:
                last_complete_at = now
                last_count = report["frames"]
            stalled = last_complete_at is not None and now - last_complete_at >= 5
            row = {
                "time": datetime.now(UTC).isoformat(), "elapsed": elapsed,
                "frame_pump": args.frame_pump, "preview": args.preview,
                "component": args.component,
                "stalled": stalled, **report,
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(
                f"{elapsed:.0f}s frames={report['frames']} "
                f"rejected={report['rejected']} timeouts={report['timeouts']}",
                flush=True,
            )
            if elapsed >= args.seconds or (args.stop_on_stall and stalled):
                raise CaptureFinished

        camera.stream_observer = observe
        try:
            with camera:
                source = CameraFramePump(camera) if args.frame_pump else camera.frames()
                with closing(source):
                    for _frame in source:
                        if args.preview:
                            if components is not None:
                                image = components.process(_frame)
                                if image is not None:
                                    cv2.imshow("Capture diagnostic", image)
                            elif _frame is not None:
                                _, _, preview = decode_duo_frame(_frame)
                                image = cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)
                                cv2.imshow("Capture diagnostic", image)
                            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                                break
        except (CaptureFinished, KeyboardInterrupt):
            pass
        except CameraError as exc:
            if not isinstance(exc.__cause__, CaptureFinished):
                raise
        finally:
            if components is not None:
                components.close()
            if args.preview:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
