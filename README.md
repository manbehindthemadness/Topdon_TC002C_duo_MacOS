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
raw thermal visualization. Open **Camera** or press `V` to choose the camera preview
or raw image, or start with `--image-source raw`. The popup also offers left/right and top/bottom
mirroring, temperature units, image filters, antialiasing, and color palettes. Changes apply live;
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
| Open camera controls | **Camera** or `V` |
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

The main thermal image window remembers its resized dimensions across viewer
restarts. The **Camera** popup also remembers its window size across closes and viewer restarts,
and groups display, AI enhancement, and camera adjustments.
Each control has a title and editable value; only numeric ranges have sliders.
Choice and on/off controls use dropdowns. On/off settings are grouped below the display
and camera adjustments. All available inputs stay enabled. The single
**Advanced / Auto** checkbox is reserved for future automatic control behavior. Hardware rows include ambient and reflected temperature,
distance, emissivity, humidity, optical transmission, center overlay, brightness,
contrast, noise reduction mode and levels, detail enhancement, and camera palettes.
Use **Restore camera settings** to restore all original camera values. Quitting also restores camera overrides.
Original payloads are saved under `$XDG_STATE_HOME/topdon-duo/camera-baselines`
(default `~/.local/state/topdon-duo/camera-baselines`) before controls are enabled.

**Camera → Upsampling algorithm** offers **Anime4K09 2×** (the algorithm selected
by the inspected Android phone app), plus **ACNet 2×** with no, light, medium,
or strong denoising. It defaults to Off. **Enhancement input size** selects
native sensor size (the default, 256×192 → 512×384) or the full preview.
**Enhancement amount** blends ACNet output with ordinary interpolation, or
adjusts Anime4K09's gradient strength; 0 gives the original image and 1 gives
full enhancement. **Anime4K09 passes** offers 1–5 passes; the phone uses 3.

Anime4K09 uses a portable CPU port of the public OpenCL rules with the phone's
parameters. ACNet uses four bundled ONNX models through the existing OpenCV CPU
engine. Neither needs downloads, compilation, or extra runtime dependencies.
Models load when selected and are reused across frames.
The Camera status shows processing time or a fallback error.
Enhancement also appears in saved images and recordings, while the temperature
grid, spot positions, and temperature readings keep their original resolution.
The existing display scale controls the final image size; enhancement runs at 2×
before that final resize. Native mode downsamples enlarged previews before
enhancement. Full preview mode retains their complete input resolution.
Processing may reduce the live frame rate, especially for the full 512×384 preview.
The portable Anime4K09 port has been checked against the public kernel rules;
identical results to a phone GPU have not been established.

When a hardware temperature-correction row is edited, measurements use the
camera count conversion (`raw / 64 - 50`) instead of the software ambient anchor;
readings remain approximate. Camera image controls preserve its preview intensity,
and camera palettes use its YUYV color output. The display gradient still applies
to raw images and grayscale previews. Unsupported or unvalidated hardware switches
are omitted.

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

By default the viewer scales raw counts by 1/64 and anchors the cold-background
percentile to `--ambient` in software. The toolbar ambient adjustment does not
send that setting to the camera; use the **Camera** popup's hardware ambient row
for that. These readings are approximate and are not measurement-grade.

Live hardware experiments found that ambient temperature, distance and emissivity
can be set on the camera, and that mode-8 counts converted with `raw / 64 - 50`
usually agree closely with the camera's telemetry. Editing a hardware correction
row selects this conversion; restoring camera settings returns to the software
ambient anchor.

## Hardware research

[Hardware investigation notes](docs/hardware-research.txt) record over 60 live
control trials, the verified USB protocol, preserved settings, six captured
stream formats, processing controls, and unresolved features. The official
Android app's TC002C Duo TISR switch was traced to host-side enhancement using
Anime4KCPP's rule-based Anime4K09 algorithm. ACNet CNN models are also bundled,
but use a separate constructor. The generic SDK's ignored super-resolution field
is a separate interface. The app's iOS implementation has not been inspected.
[Enhancement source and model notes](docs/anime-enhancement.txt) record the
obtained upstream code, verified ACNet weights, and a successful local test.

Large captures, original configuration payloads and extracted research material
remain local under the gitignored `diagnostics/telemetry-research/` directory.

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
