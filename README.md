# TOPDON TC002C Duo for macOS and Ubuntu

An experimental native thermal-camera viewer for the TOPDON TC002C Duo on
Apple Silicon Macs and Ubuntu Linux. It reads the camera directly over USB, extracts the native
`256x192` 16-bit radiometric plane, and provides both an OpenCV desktop app and
a browser-based MJPEG viewer.

## Features

- Live false-colour thermal video at the camera's 25 fps rate
- Per-pixel temperature inspection
- Resizable and rotatable desktop view
- Hardware ambient-temperature correction
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
.venv/bin/topdon-duo-desktop --rotate 90
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
sudo .venv/bin/topdon-duo-desktop --rotate 90
```

Set ambient temperature in **Camera → Ambient temperature**; this writes to the
camera and its corrected counts drive desktop measurements. Use `--rotate` with `0`, `90`, `180`, or `270` to choose the starting
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
| Adjust ambient temperature | Camera popup hardware ambient row |
| Save image data | **Capture** → **Save image data**, or `S` |
| Open image, video and timelapse controls | **Capture** or `C` |
| Rotate clockwise | **Rotate** or `O` |
| Toggle metric/imperial units (°C + meters / °F + feet) | **Units** or `F` |
| Open camera controls | **Camera** or `V` |
| Show/hide graph area | **Show graph** / **Hide graph** or `G` |
| Start/stop temperature logging | **Log to CSV** / **Stop logging** at the top of the graph area, or `L` |
| Show control help | **Help** or Space |
| Quit | **Quit**, `Q`, or Escape |

**Show graph** doubles the main window width, keeping its height and displaying
thermal history on the right. **Hide graph** halves the width again. The toggle
is remembered across restarts. The master graph plots minimum, average, maximum
and center temperatures; each placed spot gets its own graph in placement order.
All graphs share the available height evenly and show a rolling 60-second history
in the selected Celsius/Fahrenheit unit. Sampling and rendering run on a separate
worker thread every 0.5 seconds, using the latest camera measurements without
queuing video frames. Hiding the graph area pauses sampling and rendering;
reopening resumes, with gaps rather than connecting lines across long pauses.
Spot history survives rotation and mirroring, and clears when the spots are
cleared. Histories are session-only. Captures and recordings continue to contain
the thermal view only.

**Log to CSV** opens a Save dialog before logging begins. **Stop logging** closes
and saves the file; **Cancel logging** cancels an unfinished Save dialog. Hiding
the graph area is disabled while logging or choosing its file. During logging,
Camera inputs, reset/restore buttons and Advanced / Auto are disabled. Main-window
unit, rotation and spot changes are also blocked, including shortcuts and queued
settings requests. Stopping logging unlocks the available controls again.
Quitting the app
finalizes active logs, and logging does not resume automatically on restart.
CSV files open in spreadsheet applications such as Excel or LibreOffice. Each
half-second sample has one row for each master statistic and each active spot,
with columns `timestamp_utc`, `elapsed_seconds`, `series_id`, `series`,
`temperature_celsius`, `temperature_display`, and `display_unit`. Spot IDs remain
stable through rotation and mirroring, and change when spots are cleared and
replaced. Both canonical Celsius and selected-unit values are recorded.
Rows are flushed after each sample; write failures stop logging and report an
error without stopping the live graphs.

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
Display preferences, selected temperature unit, rotation, Advanced / Auto state
and successfully applied hardware overrides are saved as `settings.json` beside
`main-window.json` in the application configuration directory. Changes save as
they are applied and on exit. Saved overrides are reapplied after reading and
preserving the camera baseline on the next launch; the physical baseline is
still restored on exit. Restore camera settings also clears saved hardware
overrides; Reset display settings saves the display defaults. Explicit
`--rotate` and `--image-source` options override remembered values. Invalid saved
fields are ignored independently so valid preferences can still load.
Each control has a title and editable value; only numeric ranges have sliders.
The **Units** toggle and Camera's **Measurement units** selection switch together:
metric uses Celsius and meters; imperial uses Fahrenheit and feet. Ambient,
reflected-temperature and distance inputs convert their values, ranges and
steps accordingly. Distance sliders retain the camera's 1 cm precision, and
pending edits survive unit changes. Hardware writes and saved hardware overrides
remain in Celsius and meters, so switching units does not change calibration.
Choice and on/off controls use dropdowns. On/off settings use two columns within Display controls, above
AI enhancement and camera adjustments. All available inputs stay enabled. The single
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

Desktop measurements always use the camera count conversion (`raw / 64 - 50`),
including before edits and after restoring hardware settings;
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

The desktop uses camera-corrected counts converted with `raw / 64 - 50` for
statistics, cursor readings, sample spots and saved measurements. The **Camera**
popup's ambient row is the sole desktop ambient control. Its effective hardware
value also appears in the readout and capture metadata; if configuration cannot
be read, ambient is shown as unavailable. The old desktop `--ambient` argument
is accepted for compatibility but ignored with a warning.

Live hardware experiments found that ambient temperature, distance and emissivity
can be set on the camera, and mode-8 counts usually agree closely with its
telemetry. Restoring settings keeps the camera conversion and original hardware
ambient value. Readings remain approximate and are not measurement-grade.

The web viewer retains its software cold-background anchor and `--ambient` option.

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

## Automatic controls and future calibrations

Planned Auto behavior: adapt brightness, contrast, general/spatial/temporal noise
reduction and detail enhancement using image histograms, estimated noise, motion
and edge strength. These mappings still need validation on this camera.
Ambient temperature stays manual and is excluded from automatic calculation.
Distance, emissivity, reflected temperature and optical transmission also stay
manual until a suitable reference-based calibration is available. Palettes,
mirrors and overlays remain user preferences. Weather humidity is a possible
optional outdoor estimate, with its source and age shown; it is not a measured
indoor value. Advanced / Auto is currently a placeholder.

| Future calibration | Required reference or data | Proposed method | Validation / limits |
| --- | --- | --- | --- |
| Brightness and contrast | Fixed scenes with narrow and wide temperature spans | Map histogram percentiles and clipping to hardware settings; smooth changes over time | Check clipping and stability; distinguish existing camera gain adjustment from host adjustment |
| General and spatial denoising | Repeated frames of flat regions and fine-detail targets | Estimate noise and measure the effect of each strength | Compare noise reduction against lost detail; scene texture must not count as noise |
| Temporal denoising | Static scenes and controlled target motion | Map temporal noise and motion estimates to filtering strength | Check trails and response delay; allow hardware settling before comparing frames |
| Detail enhancement | Edge targets captured at several noise levels | Select strength using edge clarity and noise estimates | Check halos and amplified noise; enhanced pixels do not establish new measurement resolution |
| Distance / camera geometry | Known-size target, measured distances and target orientation | Calibrate projection and estimate distance from target size in native pixels | Validate at held-out distances; arbitrary objects and unknown orientation do not give reliable absolute distance |
| Emissivity | Known-emissivity reference on the same surface or an independent surface-temperature measurement; supplied environmental corrections | Fit emissivity to the reference temperature, then restore original settings | Verify thermal equilibrium and useful temperature contrast; result applies to that material, surface and measurement conditions |
| Reflected apparent temperature | Suitable reflector placed near the target | Guided reflector measurement with a documented temporary camera configuration | Validate the procedure against this camera's supported distance and correction settings; restore originals afterward |
| Optical transmission | Known IR-transmitting optics and a stable reference target | Compare reference measurements with and without the optics under controlled conditions | Confirm the hardware field's meaning and compensation model before fitting; a single temperature ratio is insufficient |

For later trials, retain original hardware payloads, reference conditions, camera
and firmware identification, raw frames, applied settings, settling time and
validation results. Record estimates separately from measured references and
preserve the camera's corrected temperature conversion throughout display tuning.
The measurement reference procedures are informed by
[FLIR's thermographic measurement guidance](https://support.flir.com/docdownload/assets/web/4jau/en-us/T505000.xml.html);
they still require validation for this camera.

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
