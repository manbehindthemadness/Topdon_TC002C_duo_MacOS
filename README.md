# TOPDON TC002C Duo viewer for macOS

An experimental live thermal viewer for the TOPDON TC002C Duo on Apple Silicon
macOS. It talks directly to the camera's USB Video Class bulk endpoint because
AVFoundation/OpenCV can select one of the device's malformed UVC descriptors and
crash before capture starts.

The viewer negotiates the camera's native `256x392 @ 25 fps` YUY2 stream, splits
its 196-row image and radiometric panes, converts the 16-bit radiometric values,
and serves a false-colour MJPEG stream at <http://127.0.0.1:5001>.

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

The camera supplies raw values that convert as `raw / 64 - 273.15`. The display
is useful for thermal contrast and relative readings. Absolute temperatures are
not calibrated by this project and should not be treated as measurement-grade.

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
