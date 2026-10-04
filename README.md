# TOPDON TC002C Duo for macOS and Ubuntu

An experimental native thermal-camera viewer for the TOPDON TC002C Duo on
Apple Silicon Macs and Ubuntu Linux. It reads the camera directly over USB, extracts the native
`256x192` 16-bit radiometric plane, and provides both an OpenCV desktop app and
a browser-based MJPEG viewer.

## Features

- Live false-colour thermal video at the camera's 25 fps rate
- Per-pixel temperature inspection
- Resizable and rotatable desktop view
- Adjustable ambient-temperature calibration
- Celsius and Fahrenheit display modes
- Native macOS Save dialog and GTK Save dialog on Ubuntu
- PNG preview plus lossless NPZ radiometric data and JSON metadata
- Optional local web viewer

## Requirements

- Apple Silicon Mac running macOS, or Ubuntu 24.04+ with a graphical desktop
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- TOPDON TC002C Duo (`2bdf:0102`)

## Quick start

### Ubuntu

The original macOS setup remains available below. On Ubuntu, install the runtime
libraries and GTK save dialog, then create a local Python environment:

```bash
cd /usr/src/Topdon_TC002C_duo_MacOS
sudo apt install python3-venv libusb-1.0-0 libgl1 libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 zenity
python3 -m venv .venv
.venv/bin/python -m pip install uv
.venv/bin/uv sync --dev
```

Allow the active local desktop user to access this camera without running the
viewer as root:

```bash
sudo install -m 644 packaging/udev/70-topdon-duo.rules /etc/udev/rules.d/70-topdon-duo.rules
sudo udevadm control --reload-rules
```

Unplug and reconnect the camera, then run:

```bash
.venv/bin/topdon-duo --diagnose
.venv/bin/topdon-duo-desktop --ambient 21.9 --rotate 90
# Alternatively, start the browser viewer:
.venv/bin/topdon-duo --ambient 21.9 --rotate 90
```

Linux's `uvcvideo` driver is temporarily detached only from this camera's two
interfaces while capture runs, then reattached when the viewer closes. Close
other apps using the camera before starting. The udev rule applies to the active
local desktop session; remote/headless users need USB permissions configured
separately. Live capture has been checked on Ubuntu 24.04. Linux assembly uses
the negotiated full frame size and rejects incomplete frames; the macOS
assembler and radiometric decoding remain unchanged.

### macOS

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

The viewer uses the camera's processed preview when it is populated, including
its full 512×384 plane on supported Linux frames. Temperatures and sample spots
still use the native 256×192 measurement plane. Empty previews fall back to the
raw thermal visualization. Open **View** or press `V` to choose the camera preview
or raw image, or start with `--image-source raw`. The popup also offers left/right and top/bottom
mirroring, image filters, antialiasing, and color palettes. Changes apply live;
mirroring keeps existing spots on the same physical sensor pixels. Filters and
color changes affect the picture while temperatures retain the measurement data.
The selected view also applies to saved images and recordings; this does not establish whether TISR is active.

## Desktop controls

The controls are available from the toolbar as well as the keyboard:

| Action | Control |
| --- | --- |
| Inspect a pixel | Move the mouse over the image |
| Place fixed sample spots | **Add spots** or `P`, then click image locations |
| Clear all sample spots | **Clear spots** or `P` again |
| Adjust ambient temperature | Toolbar, `[` / `]`, slider, or mouse wheel when supported |
| Save image data | **Capture** → **Save image data**, or `S` |
| Open image, video and timelapse controls | **Capture** or `C` |
| Rotate clockwise | **Rotate** or `O` |
| Toggle Celsius/Fahrenheit | **Unit** or `F` |
| Open display settings | **View** or `V` |
| Show control help | **Help** or Space |
| Quit | **Quit**, `Q`, or Escape |

Sample spots use crosshairs with a one-pixel stroke and live temperature
readings in the selected unit. They remain on the same thermal pixels when the
window is resized or the view is rotated. Placement stays enabled until you
press **Clear spots** or `P` again.
The center temperature appears in the statistics without a fixed center crosshair.
Crosshairs invert the thermal pixels beneath them, with a one-pixel black or white
outline. Temperature labels use white text with a two-pixel black outline to stay
readable across hot/cold boundaries and busy backgrounds. Nearby labels move
automatically to avoid each other and the crosshairs, with a connecting line
for every reading. The mouse readout also avoids fixed spot labels.

The mouse sampler uses the same thin crosshair and hides its marker and
temperature reading when the pointer leaves the image.

Saving opens the native macOS Save dialog or Ubuntu's Zenity Save dialog without pausing camera capture. A
single chosen filename produces three matching files:

- `.png` — the false-colour image
- `.npz` — lossless `uint16` raw counts and `float32` Celsius temperatures
- `.json` — capture time, statistics, orientation, ambient setting, and selected
  pixel information

Radiometric data remains in Celsius even when the viewer is displaying
Fahrenheit.

Open **Capture** to access image, video and timelapse controls in a separate popup
window. **Save image data** saves the PNG, raw/Celsius arrays, and JSON metadata
described above. **Record video** or **Record timelapse** opens a Save dialog for an `.mp4`
filename. Recording starts after you select the filename; press **Stop video**
or **Stop timelapse** in the popup to finish. While choosing a filename, the
button becomes **Cancel**. The popup and main toolbar show recording status.
Closing the popup keeps recording active; reopen **Capture** to stop it.
Quitting the viewer finalizes an active recording.

Set the timelapse rate in the popup's **Timelapse frames per minute** number
field before starting. It defaults to **60** (one frame each second) and supports
1–1500 frames per minute. You can also set
the initial rate with `--timelapse-fpm 120`. Both modes play back at 25 fps; the
default timelapse therefore plays 25 times faster than real time.

Recordings include fixed sample spots and their live temperature readings.
The popup's **Capture cursor** checkbox is unchecked by default; checking it includes the moving
sampler crosshair and temperature reading whenever the pointer is over the
image. You can toggle it during a recording. The toolbar and help panel stay out
of the video. Rotating during recording fits the image into the original video
dimensions with black borders, preserving its aspect ratio.

## Web viewer

Start the local server:

```bash
# macOS: prefix with sudo. Ubuntu with the udev rule: run as your regular user.
.venv/bin/topdon-duo --ambient 21.9 --rotate 90
```

Then open <http://127.0.0.1:5001>. To make it available on your local network:

```bash
.venv/bin/topdon-duo --host 0.0.0.0 --port 5001
```

Only use `0.0.0.0` when you intend to expose the viewer to other devices on the
network.

## Diagnostics

Camera discovery can be checked without claiming its USB interfaces:

```bash
uv run topdon-duo --diagnose
```

If capture cannot open the camera, disconnect other apps using it, reconnect the
device, and wait a moment for it to enumerate. On macOS, run the viewer with
`sudo`. On Ubuntu, install the udev rule above and reconnect the camera. Discovery
can still list interfaces without write access; unavailable USB strings can
indicate missing permissions.

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
