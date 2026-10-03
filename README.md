# TOPDON TC002C Duo for macOS

An experimental native thermal-camera viewer for the TOPDON TC002C Duo on
Apple Silicon Macs. It reads the camera directly over USB, extracts the native
`256x192` 16-bit radiometric plane, and provides both an OpenCV desktop app and
a browser-based MJPEG viewer.

## Features

- Live false-colour thermal video at the camera's 25 fps rate
- Per-pixel temperature inspection
- Resizable and rotatable desktop view
- Adjustable ambient-temperature calibration
- Celsius and Fahrenheit display modes
- Native macOS Save dialog
- PNG preview plus lossless NPZ radiometric data and JSON metadata
- Optional local web viewer

## Requirements

- Apple Silicon Mac running macOS
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- TOPDON TC002C Duo (`2bdf:0102`)

## Quick start

```bash
git clone https://github.com/manbehindthemadness/Topdon_TC002C_duo_MacOS.git
cd Topdon_TC002C_duo_MacOS
uv sync

# Direct USB capture needs elevated access on macOS.
sudo .venv/bin/topdon-duo-desktop --ambient 21.9 --rotate 90
```

Set `--ambient` to the measured room temperature for more useful absolute
readings. Use `--rotate` with `0`, `90`, `180`, or `270` to choose the starting
orientation.

## Desktop controls

The controls are available from the toolbar as well as the keyboard:

| Action | Control |
| --- | --- |
| Inspect a pixel | Move the mouse over the image |
| Adjust ambient temperature | Toolbar, `[` / `]`, slider, or mouse wheel when supported |
| Save a capture | **Save** or `S` |
| Rotate clockwise | **Rotate** or `O` |
| Toggle Celsius/Fahrenheit | **Unit** or `F` |
| Show control help | **Help** or Space |
| Quit | **Quit**, `Q`, or Escape |

Saving opens the native macOS Save dialog without pausing camera capture. A
single chosen filename produces three matching files:

- `.png` — the false-colour image
- `.npz` — lossless `uint16` raw counts and `float32` Celsius temperatures
- `.json` — capture time, statistics, orientation, ambient setting, and selected
  pixel information

Radiometric data remains in Celsius even when the viewer is displaying
Fahrenheit.

## Web viewer

Start the local server:

```bash
sudo .venv/bin/topdon-duo --ambient 21.9 --rotate 90
```

Then open <http://127.0.0.1:5001>. To make it available on your local network:

```bash
sudo .venv/bin/topdon-duo --host 0.0.0.0 --port 5001
```

Only use `0.0.0.0` when you intend to expose the viewer to other devices on the
network.

## Diagnostics

Camera discovery can be checked without claiming its USB interfaces:

```bash
uv run topdon-duo --diagnose
```

If capture cannot open the camera, disconnect other apps using it, reconnect the
device, wait a moment for macOS to enumerate it, and run the viewer with `sudo`.

## Temperature accuracy

The radiometric mode uses a validated gain of 1/64 °C per raw count. Its
per-frame offset varies with the camera's internal temperature, so this project
anchors the cold-background percentile to `--ambient`. The result is useful for
thermal contrast and approximate readings, but it is not measurement-grade.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

## Acknowledgements

This project began as a macOS adaptation of
[tna76874/topdon](https://github.com/tna76874/topdon), which credits
PyThermalCamera and P2Pro-Viewer. The direct UVC handling builds on published
TC001N/`2bdf:0102` descriptor and streaming analysis by Samuel Loury. The
original BSD 2-Clause license is retained.
