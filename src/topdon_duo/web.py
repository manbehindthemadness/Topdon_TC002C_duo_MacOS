"""Flask live viewer and command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string

from . import __version__
from .camera import CameraError, TC002CDuoCamera, platform_warning
from .render import ThermalRenderer

LOG = logging.getLogger(__name__)

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TOPDON TC002C Duo</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #090b10; color: #edf1f7; display: grid;
           min-height: 100vh; place-items: center; }
    main { width: min(96vw, 900px); }
    header { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
    h1 { font-size: clamp(1.25rem, 3vw, 2rem); margin: 0.75rem 0; }
    #status { color: #90e0aa; font-size: .9rem; }
    img { display: block; width: 100%; height: auto; border-radius: 12px;
          background: #11151d; box-shadow: 0 16px 48px #0009; }
    footer { color: #8c96a8; margin: .8rem 0; font-size: .85rem; }
  </style>
</head>
<body><main>
  <header><h1>TOPDON TC002C Duo</h1><span id="status">Connecting...</span></header>
  <img src="/stream.mjpg" alt="Live thermal camera stream">
  <footer>Native 256x196 radiometric pane · 25 fps camera · uncalibrated temperatures</footer>
  <script>
    const status = document.querySelector('#status');
    setInterval(async () => {
      try {
        const r = await fetch('/api/status', {cache: 'no-store'});
        const s = await r.json();
        status.textContent = s.error || (s.stats ?
          `Center ${s.stats.center.toFixed(1)} C · ${s.frames} frames` : 'Starting...');
        status.style.color = s.error ? '#ff8d8d' : '#90e0aa';
      } catch (_) { status.textContent = 'Viewer unavailable'; }
    }, 1000);
  </script>
</main></body></html>"""


class LiveStream:
    def __init__(self, camera: TC002CDuoCamera | None = None) -> None:
        self.camera = camera or TC002CDuoCamera()
        self.renderer = ThermalRenderer()
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.stats: dict[str, float] | None = None
        self.frames = 0
        self.error: str | None = None
        self.running = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.camera.open()
        self.running.set()
        self.thread = threading.Thread(target=self._capture, name="tc002c-capture", daemon=True)
        self.thread.start()

    def _capture(self) -> None:
        try:
            for frame in self.camera.frames():
                if not self.running.is_set():
                    break
                image, stats = self.renderer.render(frame)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.stats = stats.as_dict()
                    self.frames += 1
                    self.condition.notify_all()
        except CameraError as exc:
            LOG.error("Capture stopped: %s", exc)
            self.error = str(exc)
            with self.condition:
                self.condition.notify_all()
        finally:
            self.camera.close()

    def mjpeg(self):
        seen = -1
        while self.running.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda current=seen: self.frames != current or self.error is not None,
                    timeout=2.0,
                )
                if self.error and self.jpeg is None:
                    return
                if self.jpeg is None or self.frames == seen:
                    continue
                seen = self.frames
                jpeg = self.jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def status(self) -> dict[str, object]:
        return {"frames": self.frames, "stats": self.stats, "error": self.error}

    def stop(self) -> None:
        self.running.clear()
        self.camera.close()
        with self.condition:
            self.condition.notify_all()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.5)


def create_app(stream: LiveStream) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/stream.mjpg")
    def video_stream():
        return Response(stream.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        return jsonify(stream.status())

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TOPDON TC002C Duo live viewer for macOS")
    parser.add_argument("--host", default="127.0.0.1", help="web bind address")
    parser.add_argument("--port", type=int, default=5001, help="web port")
    parser.add_argument("--diagnose", action="store_true", help="list the USB device and exit")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if warning := platform_warning():
        LOG.warning(warning)
    if args.diagnose:
        print(json.dumps(TC002CDuoCamera.diagnostics(), indent=2))
        return 0

    stream = LiveStream()
    app = create_app(stream)
    try:
        stream.start()
    except CameraError as exc:
        stream.error = str(exc)
        stream.running.set()
        LOG.error("Unable to start capture: %s", exc)
        LOG.info("The web status page will remain available for diagnostics")

    def stop(_signum=None, _frame=None):
        stream.stop()

    signal.signal(signal.SIGTERM, stop)
    try:
        LOG.info("Open http://%s:%d", args.host, args.port)
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
