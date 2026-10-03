# TOPDON TC002C Duo viewer for macOS

An experimental live thermal viewer for the TOPDON TC002C Duo on Apple Silicon
macOS. It talks directly to the camera's USB Video Class bulk endpoint because
AVFoundation/OpenCV can select one of the device's malformed UVC descriptors and
crash before capture starts.

The viewer negotiates the camera's radiometric UVC mode (frame index 10), extracts
its native `256x192` 16-bit temperature plane, and serves a false-colour MJPEG
stream at <http://127.0.0.1:5001>.

## Requirements

- macOS on Apple Silicon
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A TOPDON TC002C Duo that settles as USB device `2bdf:0102`

## Install and run

```bash
git clone https://github.com/manbehindthemadness/Topdon_TC002C_duo_MacOS.git
cd Topdon_TC002C_duo_MacOS
uv sync

# macOS owns the UVC interfaces, so direct USB capture must run elevated.
sudo .venv/bin/topdon-duo
```

The Duo's temperature offset tracks its internal sensor temperature. By default,
the viewer anchors the cold background to 22 °C. Set this to your measured room
temperature for better absolute readings:

```bash
sudo .venv/bin/topdon-duo --ambient 21.9
```

Use the **Rotate 90° clockwise** button while viewing, or set the initial camera
orientation on startup:

```bash
sudo .venv/bin/topdon-duo --ambient 21.9 --rotate 90
```

Then open <http://127.0.0.1:5001>. Keep the default loopback host unless you
intentionally want to expose the stream to your local network:

```bash
sudo .venv/bin/topdon-duo --host 0.0.0.0 --port 5001
```

Diagnostics that do not claim the camera can run without `sudo`:

```bash
uv run topdon-duo --diagnose
```

Press `Ctrl-C` to stop. The USB interfaces are released and the macOS drivers
are reattached when possible.

## Temperature caveat

The radiometric mode has a validated gain of 1/64 °C per raw count, but its
per-frame offset depends on the camera's internal temperature. The viewer tracks
that drift by anchoring the second percentile to `--ambient`. It is useful for
thermal contrast and approximate readings, but is not measurement-grade.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

## Acknowledgements

This project began as a macOS adaptation of
[tna76874/topdon](https://github.com/tna76874/topdon), which in turn credits
PyThermalCamera and P2Pro-Viewer. The direct UVC handling is based on the
published TC001N/`2bdf:0102` descriptor and streaming analysis by Samuel Loury.
The original BSD 2-Clause license is retained.
