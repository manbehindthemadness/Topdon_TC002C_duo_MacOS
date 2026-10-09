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

### NVIDIA Jetson / JetPack

The Linux viewer supports optional NVIDIA CUDA inference for ACNet and ONNX
visual models. Follow [JetPack setup and GPU verification](docs/nvidia-jetpack.md)
to install a compatible GPU runtime, then select **NVIDIA CUDA** in the node
execution controls. Radiometric measurements keep their native CPU path.

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
| Stop/start spot placement | **Add spots** or `P` again; existing spots remain |
| Reposition an existing spot | Left-click its crosshair and drag; Add spots may be on or off |
| Manage sample spots | Right-click the thermal image for per-spot checkboxes and Clear buttons |
| Clear all sample spots | Right-click → **Clear all spots** (top of the menu) |
| Adjust ambient temperature | Camera popup hardware ambient row |
| Save image data | **Capture** → **Save image data**, or `S` |
| Open image, video and timelapse controls | **Capture** or `C` |
| Rotate clockwise | **Rotate** or `O` |
| Toggle metric/imperial units (°C + centimeters / °F + inches) | **Units** or `F` |
| Open camera controls | **Camera** or `V` |
| Calibrate the camera once | Right-click the thermal image → **Calibrate now** |
| Show/hide graph area | **Show graph** / **Hide graph** or `G` |
| Start/stop temperature logging | **Log to CSV** / **Stop logging** at the top of the graph area, or `L` |
| Show control help | **Help** or Space |
| Quit | **Quit**, `Q`, or Escape |

### Visual AI models

Install the optional weights once (not while capturing):

```bash
uv sync
uv run topdon-duo-models espcn mewzoom
uv run topdon-duo-models mewzoom-v0-2x
# Optional newer TrunkNet models (keeps V0 available):
uv run topdon-duo-models mewzoom-v1-2x mewzoom-v1-4x
# Lightweight general-purpose 4x upscaler:
uv run topdon-duo-models realesr-general-x4v3
# One-time conversion of the authors' denoising weights:
uv run scripts/export_visual_denoisers.py dncnn-25 ffdnet-gray
# Then launch the camera viewer (install commands do not launch it):
uv run topdon-duo-desktop
```

In **Camera → Pipeline**, insert **AI enhancement** in a
software tab. Start with **Native sensor** input and choose **ESPCN 3×**,
**MewZoom V0 2× (legacy)**, **MewZoom V0 4× (legacy)**, **MewZoom V1 2× (TrunkNet)**, or
**MewZoom V1 4× (TrunkNet)**, or **Real-ESRGAN general 4× v3**.
These models appear alongside ACNet and Anime4K09; do not
stack upscalers initially. ONNX upscalers use one fixed-scale pass, an enhancement blend,
and default to CPU execution. Capable systems also expose Apple Core ML with a
compute-device preference, or NVIDIA CUDA with the optional GPU runtime.
Saved GPU preferences are retained; an unavailable GPU backend tries the other
GPU before CPU. Explicit CPU selections remain CPU.
The **Compute devices** selector displays **CPU only** and is disabled while
execution is set to CPU. Selecting Apple Core ML restores the saved device choice.

- [ESPCN](https://huggingface.co/onnxmodelzoo/super-resolution-10): luminance-only
  3× enhancement, about 240 KB, Apache-2.0. The published fixed 224×224 export
  uses overlapping tiles to preserve the camera feed's rectangular geometry.
- [MewZoom V0 2X](https://huggingface.co/andrewdalpino/MewZoom-V0-2X): full-RGB
  2× enhancement, about 7.65 MB ONNX download, Apache-2.0. This is the non-control variant.
- [MewZoom V0 4X](https://huggingface.co/andrewdalpino/MewZoom-V0-4X): full-RGB
  4× enhancement, about 57.3 MB, Apache-2.0. This is the non-control variant;
  its enhancement amount blends against bicubic, not independent noise/blur controls.
- [MewZoom V1 2X](https://huggingface.co/andrewdalpino/MewZoom-V1-2X): full-RGB
  TrunkNet, 5.3M parameters, about 23 MB ONNX download, Apache-2.0.
- [MewZoom V1 4X](https://huggingface.co/andrewdalpino/MewZoom-V1-4X): full-RGB
  TrunkNet, 21M parameters, Apache-2.0. This is larger than V0 4X, so it is
  not automatically faster. The larger UNet variants are not included.
- [Real-ESRGAN general x4v3](https://github.com/xinntao/Real-ESRGAN): compact
  full-RGB 4× model, BSD-3-Clause. Uses the checksum-pinned float32
  [SkillSafe ONNX conversion](https://huggingface.co/skillsafe-ai/realesr-general-x4v3)
  (4.64 MB), with its documented upstream checkpoint and conversion recipe.
  Enhancement amount is a bicubic blend, not Real-ESRGAN's two-weight denoise control.

For same-size denoising, insert another **AI enhancement** node before an
upscaler and select DnCNN or FFDNet. They process luminance and preserve chroma
and current image dimensions; input-size and pass controls are hidden:

- **DnCNN luminance (fixed noise 25)** uses KAIR's 17-layer `dncnn_25` weights.
  Adjust the denoising blend; its trained noise level is fixed.
- **FFDNet luminance (adjustable noise)** uses KAIR's 15-layer `ffdnet_gray`
  weights. Start at sigma 15; the 0–75 sigma control describes 8-bit image noise,
  not degrees or sensor calibration. Odd dimensions are padded and cropped back.

The old separate ONNX super-resolution, ONNX denoising and Apple Core ML ACNet
nodes are removed from Add/Insert menus. Use **AI enhancement** for those models
and execution controls. Existing pipelines and presets still load their legacy
nodes unchanged; model caches are reused and no reinstallation is needed.

The exporter downloads weights from the authors' [KAIR release](https://github.com/cszn/KAIR/releases/tag/v1.0)
over verified HTTPS, loads tensor-only checkpoints with `weights_only=True`,
checks ONNX validity and CPU/PyTorch parity at two rectangular sizes, then
installs weights, an offline checksum, and a provenance JSON in the model cache.
PyTorch and ONNX export dependencies live in a separate `uv` script environment;
the viewer does not require them. Both models are MIT licensed (Kai Zhang;
notice in `scripts/LICENSE-KAIR.txt`). Apple execution is optional and experimental;
provider availability is not a performance guarantee. Preferences survive on CPU-only systems.

### Pipeline presets

Dragging a node shows a placement bar at valid drop positions in each pipeline
stack. Source and Output remain fixed, and locked stacks cannot be reordered.

The dropdown displays the loaded preset name and keeps it selected while you
edit. **Update pipeline** re-saves that preset, including pending input edits.
With no preset loaded, the button reads **Save pipeline** and asks for a name;
saving selects the new preset. **Save current pipeline as…** in the dropdown
saves under another name. Updating a bundled preset creates a user override.
Reopening Camera settings recognizes presets that match the active pipeline.
Choose **Create new** to start a blank, unsaved pipeline with only the required
source/output nodes. **Rename current pipeline…**, directly below Create new,
renames the loaded preset while preserving its saved settings and open edits.
**Delete current pipeline…** removes the loaded preset after
confirmation and keeps the open pipeline available to edit or save again. Deleted
bundled presets stay hidden across restarts without changing the shipped files.

**Import pipelines** and **Export pipelines** open checkbox lists for selecting
multiple pipelines. Export the current pipeline (including open edits), saved
presets, or both into one JSON file. Import accepts these bundles and existing
individual pipeline files, including several files at once. Name conflicts offer
Rename, Replace, or Skip; unique names are suggested to keep existing presets.
Imported pipelines are saved as presets, with an option to load the first selection.

Raw thermal display pixels (including raw blend/mask inputs and branches) use
the latest valid sensor frame, independently of temporal measurement averaging.
Temperature readouts, spots and graphs retain their averaged measurement plane;
calibration/invalid frames still hold the last valid image and readings.

**Yautja** preserves the black-hot preview, thermal-detail branch B, raw contrast
mask in C and camera-palette overlay in D, with single-pass Apple Core ML ACNet
and the saved hardware detail/noise configuration.

**Yautja GPU** retains that same multi-branch composition and hardware setup,
with native-input Real-ESRGAN 4× using Apple Core ML `ALL` devices instead of ACNet.

**Redneck Combat** is also bundled in the dropdown, preserving the original
Inferno preview with its Laplacian/contour thermal branch and masked blend.

**Redneck Combat GPU** preserves the later tuned threshold-mask blend, Scharr
edges, contours, antialiasing and sharpening in branch B, followed by native-input
Real-ESRGAN 4× in the AI enhancement node with Apple Core ML `ALL` devices.

The Pipeline presets dropdown includes **Detail Enhanced GPU Upscale**, captured
from the tuned preview/thermal-edge blend with Real-ESRGAN 4× in the current
AI enhancement node, Native input, and Apple Core ML (`ALL` compute devices).
Its complete A/B branch configuration is bundled with
the app. GPU preferences remain stored and fall back to CPU on non-Apple systems;
Real-ESRGAN downloads on first use if missing. Loading the preset replaces the
pipeline, not your temperature measurements or other viewer preferences.

### AI image styling

Select a style in **Camera → Pipeline → AI image styling** to download its
weights automatically on first use. Downloads run in the background; the live
feed and thermal measurements continue without the style effect until it is
ready. Progress appears in the viewer status and Pipeline panel. Verified weights
are cached for offline reuse. On a download failure, bypass/re-enable the node
to retry. Blend 0 or bypassed nodes do not trigger downloads.

This also applies to downloadable ONNX enhancement models. DnCNN/FFDNet still
require their documented one-time local export; the viewer does not install
Torch or launch an export automatically.

Manual pre-installation remains available if desired:

```bash
uv run topdon-duo-models style-mosaic style-candy
uv run topdon-duo-models style-rain-princess style-udnie style-pointillism style-line-art style-animegan-sketch
uv run topdon-duo-desktop
```

In **Camera → Pipeline**, add **AI image styling** and choose Mosaic, Candy,
Rain Princess, Udnie, Pointillism, Line Art, or AnimeGAN Portrait Sketch.
Start at the default 50% style blend. The separate node can go before
an upscaler, so styling does not interfere with your AI enhancement selections.
Blend 0 skips inference. The five painting styles use ONNX Model Zoo's verified
[Fast Neural Style exports](https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style)
(about 6.7 MB each; the upstream README lists BSD-3-Clause).

These exports have a fixed 224×224 RGB input in the **0–255** range, unlike
the enhancement models' normalized inputs. The node fits the entire frame into
that square with aspect-preserving padding, removes the padding from the styled
output, then restores the original image dimensions. This is a low-resolution
art effect, not an enhancement of sensor detail. CPU works on Linux and macOS;
Apple systems also expose optional Core ML execution, which needs local testing.
Linux NVIDIA systems expose CUDA with a compatible GPU runtime.
Saved GPU preferences are retained but ignored on CPU-only systems.

**Line Art** uses the [Informative Drawings ONNX conversion](https://github.com/josephrocca/image-to-line-art-js)
of the [MIT-licensed original project](https://github.com/carolineec/informative-drawings).
It works at a bounded 256×256 resolution, consumes normalized RGB 0–1, and
returns a single-channel drawing expanded to RGB for blending.

**AnimeGAN Portrait Sketch** uses the author's [public ONNX release](https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/releases/tag/1.0).
Its input/output are NHWC RGB in −1 to 1 at a 512×512 working resolution.
The [AnimeGANv3 terms](https://github.com/TachibanaYoshino/AnimeGANv3#scroll-license)
restrict these weights to **non-commercial use**; the installer and selector
identify this restriction. It was trained for portraits, not thermal scenes;
this node applies it to the full frame without face detection or alignment.
Both new model families use aspect-preserving padding and restore the incoming
image dimensions. All weights are checksum-verified; Udnie's publisher checksum
is resolved at installation and retained alongside the cached model.

AnimeGAN comic and 8-bit styles are demonstrated upstream, but no verified public
ONNX weight downloads were found for them, so they are not exposed as working
options. Reference-image style transfer is not integrated.

These are experimental natural-image models, not thermal-measurement models.
They may invent visual detail or flicker between frames. Hover temperatures,
spots, graphs, and saved radiometric data still use the existing sensor measurements.
Native 256×192 input produces 768×576 with ESPCN, 512×384 with either MewZoom 2X,
or 1024×768 with either MewZoom 4X or Real-ESRGAN.
Full 512×384 preview input is also supported; outputs must remain below the
4-megapixel budget. Apple provider availability does not prove all operations
execute on the GPU; each model needs local speed/quality testing.

Weights are checksum-verified and cached outside the repository (macOS:
`~/Library/Caches/topdon-duo/models`; Linux: `$XDG_CACHE_HOME/topdon-duo/models`
or `~/.cache/topdon-duo/models`). `TOPDON_MODEL_DIR` overrides the directory.
Inference never downloads weights or executes model-provided Python code.
V1 installation obtains the publisher's SHA256 from its Git LFS pointer over
verified HTTPS and saves that checksum beside the verified weights for offline
use. V0 and ESPCN retain their built-in checksum pins. Existing V0 presets and
weights are preserved; selecting V1 never silently replaces a saved V0 model.
For V1, Real-ESRGAN, and denoiser Apple inference, the runtime specializes the symbolic input dimensions
to the actual frame size before Core ML conversion, avoiding ORT's dynamic
layout-conversion axis error. Changing the input size creates a new Apple
session and may trigger compilation again. CPU inference remains dynamic and
unchanged; weights and measurements are never rewritten.
On macOS the downloader supplements Python's trust store with `/etc/ssl/cert.pem`
to support framework Python installations with missing default CA roots. TLS
certificate and hostname verification remain enabled; explicit `SSL_CERT_FILE`
and `SSL_CERT_DIR` settings are respected. Linux keeps its default trust configuration.
CPU, Apple, and NVIDIA inference run in persistent spawned helpers, containing
native crashes and keeping model work off the GUI and USB acquisition threads.

### Viewer size and graph layout

The default native camera canvas is **1024×768** in landscape (768×1024 when
rotated), with the toolbar above it. The camera retains its aspect ratio.
The viewer still remembers your resized window; use
`uv run topdon-duo-desktop --reset-window-size` once to use the new native size.
`--scale 3` selects the previous 768×576 camera canvas. Graphs are rendered at
their pane's displayed size, rather than enlarged from a fixed low-resolution plot.

**Show graph** doubles the main window width, keeping its height and displaying
thermal history on the right. **Hide graph** halves the width again. The toggle
is remembered across restarts. The master graph plots minimum, average, maximum
and center temperatures; each placed spot gets its own graph in placement order.
The graph pane grows with the window and uses the remaining width beside the
camera image. The camera image scales without changing its aspect ratio; unused
space below it stays black. Graph text stays readable as the pane grows.
All graphs share the available height evenly and use the same time axis in the
selected Celsius/Fahrenheit unit. The default range shows the most recent
10 minutes. Entire-session mode expands the axis as measurements accumulate. Older samples compress
automatically, retaining bucket minima/maxima and measurement gaps; the newest
half of the point budget remains uncompressed. The default budget is 4,096 points
per chart (2,048 recent samples).
CSV logging retains every valid sample at the selected interval. Sampling and rendering run on a separate
worker thread every 0.5 seconds by default, using the latest camera measurements without
queuing video frames. Hiding the graph area pauses sampling and rendering;
reopening resumes, with gaps rather than connecting lines across long pauses.
Spot history survives rotation and mirroring, and clears when the spots are
cleared. Histories are session-only; disabling a spot removes its graph history,
and enabling it starts fresh sampling. Captures and recordings can include the
graph pane with the Capture dialog’s **Include graphs** checkbox beside **Capture cursor**.
Camera, Capture, Graph configuration and measuring-spot menus stay above the
main viewer so they remain accessible in fullscreen.

Both checkboxes and the timelapse frames-per-minute setting save immediately
and restore on the next launch. An explicit `--timelapse-fpm` value overrides
the remembered rate.

Image, video, timelapse, CSV log dialogs each remember their
last accepted directory across launches. Canceling leaves the previous directory
unchanged. Missing directories fall back to the normal starting location; an
explicit `--output` directory overrides the remembered capture/log location.

**Update (s)**, immediately left of **Log to CSV**, sets the graph and CSV sampling
interval from 0.1 to 60 seconds. Click the number and type a value; press Enter to
apply or Escape to cancel. The interval is remembered across restarts and is
locked while logging or choosing a log file.

**Config**, beside **Update (s)**, opens Graph configuration. Choose **Entire
session** for an expanding time axis or **Recent time window** for a fixed range
in minutes. **Compress older history** keeps extrema and gaps in older samples;
turning it off keeps only the latest points when the budget is full. **Points per
chart** sets the history budget from 256 to 65,536. Increasing the budget allows
more future detail; it cannot recover discarded or already compressed samples.
These settings persist across launches and are locked during logging or file selection.
The **Close** button dismisses the configuration popup; changes apply as you make them.
**Reset**, beside **Config** in the graph header, clears chart histories. Sampling
continues at the selected interval; spot positions, graph settings and saved CSV
files are retained. Reset is disabled while logging or choosing a log file.
Changes redraw the graphs immediately without adding extra measurements to CSV.

**Log to CSV** opens a Save dialog before logging begins. **Stop logging** closes
and saves the file; **Cancel logging** cancels an unfinished Save dialog. Hiding
the graph area is disabled while logging or choosing its file. During logging,
Camera inputs and pipeline editing are disabled. Main-window
unit, rotation and spot changes are also blocked, including shortcuts and queued
settings requests. Stopping logging unlocks the available controls again.
Quitting the app
finalizes active logs, and logging does not resume automatically on restart.
CSV files open in spreadsheet applications such as Excel or LibreOffice. Each
sample has one row for each master statistic and each active spot,
with columns `timestamp_utc`, `elapsed_seconds`, `series_id`, `series`,
`temperature_celsius`, `temperature_display`, and `display_unit`. Spot IDs remain
stable through rotation and mirroring, and change when spots are cleared and
replaced. Both canonical Celsius and selected-unit values are recorded.
Rows are flushed after each sample; write failures stop logging and report an
error without stopping the live graphs.

Sample spots use crosshairs with a one-pixel stroke and live temperature
readings in the selected unit. They remain on the same thermal pixels when the
window is resized or the view is rotated. **Add spots** (or `P`) toggles placement;
turning it off preserves every existing spot. Numbered labels match the graph
numbers and the right-click menu. Left-click a visible spot's crosshair and drag
to fine-tune its location, with Add spots either on or off. Its number, region
name, and graph identity remain unchanged, and its temperature updates from the
new pixel. Dragging stays within the thermal image and is locked during logging
or calibration. A drag release never places a second spot.

Right-click the thermal image to open the spot menu. The **Clear all spots**
button at the top removes all spots; **Add spots** toggles placement. Each spot
has an enable checkbox, an editable region name, and a **Clear** button. Enter a
name such as “Left hand” or “Motor” and press Enter or leave the field to save.
Names label graph titles and the CSV `series` column; on-image labels keep their
spot numbers and temperatures. CSV `series_id` retains the stable spot identity.
Names stay with spots when rotated, mirrored, disabled, or re-enabled. Clearing
a name restores the default “Spot N”. Spot positions, names, enabled states, placement mode and numbering save
automatically and restore on launch. Locations are stored in sensor coordinates,
so they also restore correctly with rotation and mirroring. Chart histories are
session-only; saved spots start a fresh history when the app reopens.
Unchecking a spot hides its marker and stops its graph sampling without losing
its position; checking restores it.
Clearing one spot preserves the numbers and histories of the others. New spots
receive new numbers until Clear all resets numbering. Spot changes are locked
while temperature logging, a pending logging dialog, or calibration is active.
Disabled spots are omitted from graph captures and CSV logging.
The center temperature appears in the statistics without a fixed center crosshair.
Crosshairs invert the thermal pixels beneath them, with a one-pixel black or white
outline. Temperature labels use white text with a two-pixel black outline to stay
readable across hot/cold boundaries and busy backgrounds. Nearby labels move
automatically to avoid each other and the crosshairs, with a connecting line
for every reading. The mouse readout also avoids fixed spot labels.

The mouse sampler uses the same thin crosshair and hides its marker and
temperature reading when the pointer leaves the image. Its temperature label
also hides while hovering over or dragging an enabled spot, leaving that spot's
own reading visible. Starting logging cancels any active spot drag; spot controls
unlock when logging stops.

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

The Capture popup's **Include graphs** checkbox beside **Capture cursor** optionally appends
visible graphs to PNG, video and timelapse captures. It defaults to off. Show the
graphs before starting a recording; its graph layout stays fixed until recording
ends. Hiding graphs during a recording leaves that pane blank and pauses graph
updates. The checkbox is locked while a recording or its Save dialog is active.
PNG raw/Celsius arrays retain their native dimensions; graphs only affect the
viewable image. Graph captures omit application buttons and use the graph's
existing 0.5-second update cadence.

Recordings include fixed sample spots and their live temperature readings.
The popup's **Capture cursor** checkbox is unchecked by default; checking it includes the moving
sampler crosshair and temperature reading whenever the pointer is over the
image. You can toggle it during a recording. The toolbar and help panel stay out
of the video. Rotating during recording fits the image into the original video
dimensions with black borders, preserving its aspect ratio.

The main thermal image window remembers its resized dimensions across viewer
restarts. The **Camera** popup also remembers its window size across closes and viewer restarts,
and contains an expandable Camera hardware stack followed by software tabs A–D.
Tab A feeds the viewer; Combine nodes activate and blend other tabs on independent
threads. Combine inputs and optional masks can also use camera preview or raw thermal.
Image filter nodes include configurable CPU blurs, sharpening, edge detection,
CLAHE/equalization, thresholds, morphology, emboss and high-pass detail. Each
filter exposes its relevant settings and blend amount; saved older filters
retain their existing appearance. Edge features and Contour regions nodes add
CPU feature selection by direction, size, shape, brightness or position, with
overlay, isolation and mask outputs for selective Combine processing.
[Pipeline controls and examples](docs/processing-pipelines.md) explain node ordering,
static board imaging, persistence, and import/export.
Display preferences, selected temperature unit, rotation, Auto calibrate, pipeline nodes and their expanded/bypassed states
and successfully applied hardware overrides are saved as `settings.json` beside
`main-window.json` in the application configuration directory. Changes save as
they are applied and on exit. Saved overrides are reapplied after reading and
preserving the camera baseline on the next launch; the physical baseline is
still restored on exit. Restore camera settings also clears saved hardware
overrides and bypasses optional hardware nodes; Restore default pipeline resets the stacks. Explicit
`--rotate` and `--image-source` options override remembered values. Invalid saved
fields are ignored independently so valid preferences can still load.
Ambient temperature inputs retain the value you enter (for example, 72°F), while
the hardware write silently rounds to the control's Celsius step. The requested
input is remembered across launches; measurements use the actual camera setting.
Restoring camera settings clears this requested input along with hardware overrides.

**Auto calibrate** is at the top of Camera settings and defaults to off. Its
saved choice is applied on startup. After the first valid camera frame arrives,
the viewer requests one calibration, including when Auto calibrate is off.
Readings are held during the calibration as usual. Right-click the thermal image and choose
**Calibrate now** to request one calibration. Both controls are disabled while
logging or measuring a calibration reference. Automatic camera calibration is
re-enabled when the viewer exits; the saved switch choice is retained for the
next launch. Restore camera settings and Reset display settings keep this choice.
Before accessing controls, the viewer reads the camera's SDK 2.0 protocol version
to initialize command dispatch after a power cycle. If a legacy 512-byte layout
reappears, it repeats this handshake once and reselects the command; unexpected
versions or block sizes remain rejected before configuration writes.
Hardware setting changes show an operation notice at the top of Camera before
USB commands run, explaining that image and graph updates may pause briefly.
Controls are locked during the operation; completion or rejection stays visible
for eight seconds. Gamma uploads show their existing percentage progress there.
The viewer log records each camera operation and its completion time.

Each control has a title and editable value; only numeric ranges have sliders.
The **Units** toggle and Camera's **Measurement units** selection switch together:
metric uses Celsius and centimeters; imperial uses Fahrenheit and inches. Ambient,
reflected-temperature and distance inputs convert their values, ranges and
steps accordingly. Distance sliders retain the camera's 1 cm precision, and
pending edits survive unit changes. Hardware writes and saved hardware overrides
remain in Celsius and meters, so switching units does not change calibration.
Choice controls use dropdowns, and on/off settings use checkboxes. Numeric range
controls have sliders. Image source is pinned first in each software tab, and
tab A has a fixed Output node at the bottom;
optional hardware nodes are singletons. Software processing nodes run top to bottom
and can be repeated. Headers expand/collapse, grips drag, and right-click menus
add, insert, remove, or clear nodes. All inputs ignore wheel edits.

Auto calibrate, units, ambient/reflected temperature, optical transmission,
distance, emissivity and calibration tools remain separate. Optical transmission
appears directly below Reflected temperature. Fixed detail mode is a checkbox
inside Detail enhancement. It requires enabled detail enhancement, Balanced,
neutral camera gamma (50), and boost Off. Hardware controls restore their owned
fields when bypassed/removed, and original camera overrides restore on exit.
Snapshots remain under `$XDG_STATE_HOME/topdon-duo/camera-baselines`.

The old Analyze toggle is replaced by repeatable Temperature range and Sensor
interpolation nodes. A range fixes software temperature scaling, including when
Camera preview is selected (then rendering uses thermal data and explicitly
approximate camera-style colors). Camera preview effects are inactive in that
case. Bounds follow the selected temperature units and never change measurements.
[Static board analysis examples](docs/processing-pipelines.md#static-board-analysis)
show a useful starting pipeline.

Apple Core ML ACNet is available through **AI enhancement**; the old separate
node is retained only for existing pipelines. `uv sync` installs its prebuilt
runtime automatically on macOS only. Startup detection exposes Apple execution
controls when Metal and Core ML are available; saved pipelines remain unchanged. Launch with
`uv run topdon-duo-desktop`; no OpenCV rebuild is needed.
See [Apple node setup and limitations](docs/processing-pipelines.md#experimental-apple-core-ml-node).

AI enhancement nodes offer Anime4K09 and bundled ACNet ONNX models, with amount,
passes, ACNet denoising and Current/Native/Preview input sizes. They expose
Apple Core ML or NVIDIA CUDA execution when the corresponding runtime is detected.
GPU preferences stay in saved presets but are ignored on CPU-only systems,
where the normal node uses OpenCV ACNet instead. Models need no compilation
or additional weight downloads. Image processing runs independently from measurements
and graphs; slow chains report latency rather than blocking sampling. A four
megapixel intermediate-image limit reports oversize chains. Sharpening and AI
can change shapes; they do not add measured sensor pixels.


Desktop measurements always use the camera count conversion (`raw / 64 - 50`),
including before edits and after restoring hardware settings;
readings remain approximate. Camera image controls preserve its preview intensity,
and camera palettes use its YUYV color output. The two palette controls have
different roles:

- **App palette (raw / grayscale)** colors raw thermal data and grayscale previews
  in the program. It is inactive while camera color output is being displayed.
- **Camera palette (color preview)** sets colors inside the camera. Selecting one
  enables camera colors when **Image source → Camera preview** is selected. It
  is inactive when **Raw thermal image** is selected.

Use **Color source → App colors** or **Camera colors (preview)** to choose which
palette supplies the displayed colors. App colors use native thermal data when
a colored camera preview is active, so the camera's palette cannot alter the app
palette's appearance. Camera colors require **Image source → Camera preview**.
Selecting an app palette switches to App colors; selecting a camera palette
switches to Camera colors. The active source is shown below the app palette.
Only the palette for the selected Color source is editable: App colors disables
the camera palette, and Camera colors disables the app palette.
Both palette choices and the color source persist; temperature measurements
remain unchanged. Unsupported or unvalidated hardware switches
are omitted.

### Camera palette names

Palettes 10–22 have descriptive names based on sampled TC002C Duo output.
These are app labels rather than official vendor names; their camera IDs are
unchanged. White hot (1) and Black hot (2) retain their existing labels.

| Camera ID | Name | Sampled colors, cool → warm |
| --- | --- | --- |
| 10 | Violet iron | Dark blue, violet, magenta, orange, yellow |
| 11 | Classic rainbow | Dark violet, blue, green, orange, cream |
| 12 | Fire | Black, dark red, red, orange, yellow, cream |
| 13 | Sunset | Purple, magenta, red, orange, pale yellow |
| 14 | Soft iron | Dark blue, muted violet, magenta, orange, pale yellow |
| 15 | Amber | Black, brown, amber, yellow, pale gold |
| 16 | Vivid rainbow | Bright magenta, violet, blue, cyan, green, yellow, orange |
| 17 | Neon sunset | Blue, violet, bright magenta, red, orange, yellow |
| 18 | Silver | Black, gray, silver, white |
| 19 | Pastel rainbow | Dark violet, magenta, blue, green, yellow, peach, pale pink |
| 20 | Red hot | Black, dark red, bright red |
| 21 | Green hot | Black, dark green, bright green |
| 22 | Ocean heat | Dark blue, blue, cyan, green, yellow, pale highlights |

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

During camera shutter calibration (shown as “Auto calibrate”), frozen temperature frames hold the last
valid image and readings with a status message. Calibration sampling and CSV
logging skip those frames; graphs show a gap. Averaging restarts when live
measurements return. Frames containing invalid endpoint counts receive the
same treatment. Captures record measurement validity in JSON and NPZ; before
the first valid frame, temperature labels show `--` and JSON readings are null.

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

## Calibrate with household objects

Open **Camera** from the main window to find the calibration controls:

- **Post-it → distance:** use a standard 3 × 3 inch note (nominally 76 × 76 mm;
  measure yours). Hold it flat in front of a warm palm so its cooler square has
  visible edges. Enter its size and a tape-measured distance from the front lens,
  detect it, then save the reference. At another distance, measure the same note
  and apply the estimated distance to the camera.
- **Shiny spoon → reflected temperature:** show the reflector target, cover the
  entire center circle with shiny bare metal, and measure while holding steady.
  A stable reading saves automatically. Apply the saved reflected temperature to
  update the camera. This experimental method depends on spoon orientation;
  compare with a crumpled then flattened foil reflector before relying on it.
- **Known surface temperature → emissivity:** select a point and enter an
  independently measured surface temperature, fit emissivity, then apply it.

All three saved calibration references load automatically when the program
starts and appear in their Camera sections. Values you have **applied** to the
camera also restore automatically after connection. Saving a reference alone
keeps it available for later use; it does not apply a new camera setting. For
Post-it distance, the saved reference is reused for new measurements rather than
reusing an unfinished distance estimate. Reset display and Restore camera
settings preserve saved calibration references; Restore camera settings clears
applied hardware preferences until you apply them again. Repeat calibration when
the camera, target material, or relevant measurement conditions change.

## Distance calibration with a square

Camera → Distance calibration uses a **76 × 76 mm Post-it** by default. You can
edit the square's side length. Keep it flat, steady and facing the camera near
the image center. Hold the cooler note in front of your warm palm, with skin
visible around its edges. The sticky edge alone may not hold the whole note flat.
The note needs thermal contrast against the hand; if it warms to the same
temperature, use a fresh note or let it cool before trying again.

1. Measure the distance from the camera's front lens to the note. Enter that
   reference distance and the side length in centimeters or inches, according to
   the selected measurement units.
2. Click **Detect reference square** and hold steady. The detected outline appears
   on the main image and locks after eight consistent frames. Click **Save reference**.
3. Move the same square to another distance, keeping it face-on. Click
   **Measure square** and hold steady for another automatic detection.
4. Check the estimated distance against a tape measurement at another distance.
   **Apply distance to camera** writes the estimate to the hardware distance
setting, rounded to its 1 cm precision. Detecting and saving do not write it.

The reference persists across restarts; incomplete selections and estimates do
not. Recalibrate after changing the camera or lens. Clearing the calibration
leaves the hardware distance setting unchanged. Controls are locked while logging.

This is an approximate, single-reference range estimate: distance equals the
reference distance multiplied by reference pixel width divided by current pixel
width. It averages the four edges in native thermal coordinates, so display
scaling, AI upsampling, mirroring and rotation do not change the scale. A new
rotation or mirror change cancels the current detection. Crossed corners,
noticeably tilted or uneven squares, and edges below 12 native pixels are
rejected. Estimates outside the hardware's 0.3–99 m range cannot be applied.
A 76 mm target may become too small well before the camera's maximum range.

The initial shape detector sweeps temperature thresholds for square contours,
also joins straight edges when uneven warming breaks the contour, refines the
corners, and checks that warmer surroundings support at least three
edges. Its starting hand-background range is 24–40 °C, with at least 0.75 °C
contrast. It uses the native Celsius plane, independent of display palette,
units and AI enhancement. Missing or multiple qualifying squares keep detection
waiting; moving or losing the target restarts the stability check. These are
thermal heuristics, so other cool squares against warm backgrounds can qualify.
Check the displayed outline before saving or applying a measurement.

The method follows the [pinhole projection model documented by OpenCV](https://docs.opencv.org/5.0/main_modules/calib.html).
It does not correct lens distortion or solve target tilt; a single reference is
not a full camera calibration. The front-lens measurement also approximates the
optical origin. Verify accuracy at additional measured distances before relying
on it for temperature compensation.

## Emissivity calibration with a known temperature

Camera → Emissivity calibration fits the camera's global emissivity setting to
an independently measured surface temperature:

1. Click **Select reference point**. Measuring-spot, cursor and distance-detection
   markers disappear. Click the reference surface in the main thermal image;
   its own reference marker remains visible with a live temperature label.
2. The selected pixel's current temperature is copied into **Known surface
   temperature**, in the selected °C or °F units. Edit it to your independently
   measured temperature. Live updates preserve your edit. Selecting a new point
   imports its temperature again.
3. Click **Fit emissivity** and keep the camera and reference steady. The fit
   temporarily probes the camera's supported 0.01–1.00 emissivity range, waits
   one second after each write, and collects five stable readings. Other settings
   are locked during fitting, and temperature logging cannot start.
4. A successful fit saves the calibration separately, restores the prior
   emissivity, clears the reference marker and offers the fitted value.
   **Apply emissivity to camera** keeps it as a saved hardware preference and
   restores the markers. **Close calibration** ends point selection and restores
   the markers while retaining any saved fit. Closing during fitting also restores
   the previous setting.

**Reset display settings** and **Restore camera settings** preserve both the
distance reference and the saved emissivity calibration, including across
restarts. Restore returns hardware settings to their original values; use
**Apply saved emissivity** to reapply the retained fit when appropriate.

The fit reads the native camera-corrected plane rather than display smoothing,
palette pixels or AI output. It tests the observed response rather than assuming
a radiometric formula. It stops without a result for weak or inconsistent
response, an unreachable temperature, unstable readings, a timeout, or when the
camera's 0.01 emissivity steps cannot match within 0.3 °C. This matching tolerance
is not a claim about the camera's absolute temperature accuracy.

Use an independent surface-temperature reference and appropriate ambient,
reflected-temperature and distance settings. The fitted emissivity is specific
to the selected material and conditions, while applying it affects the entire
image. The procedure follows the known-temperature adjustment approach in
[FLIR's thermographic measurement guidance](https://support.flir.com/docdownload/assets/web/4jau/en-us/T505000.xml.html).
The workflow and feedback fit are covered by simulated-camera tests; a physical
trial with a measured reference is still required for this camera.

## Reflected-temperature calibration

Camera → Reflected-temperature calibration uses a small circle at the center
of the image. Show the target, cover the entire circle with a shiny bare-metal
spoon or foil reflector near the subject, and click **Measure stable temperature**.
Other measurement markers are hidden during calibration. The reading uses the
median of native temperature pixels inside the circle, with a one-second settling
period and at least two seconds of stability within 0.3 °C. It times out after
30 seconds if the reading does not settle.

Measurement temporarily sets emissivity to 1.00 and optical transmission to 100%,
then restores their previous values and override states on success, cancellation,
or failure. Other environmental corrections remain at their configured values.
A stable result is saved independently and clears the circle. **Apply saved
reflected temperature** writes it to the camera at its 0.1 °C precision. The saved
reference survives restarting, Reset display, and Restore camera settings; the
Camera panel displays it in the selected Celsius/Fahrenheit unit. Settings and
logging are locked during measurement.

A spoon is an experimental reflector: curvature makes its reading depend on
orientation, and it may reflect the operator or camera. Compare it against the
[FLIR foil reflector procedure](https://support.flir.com/docdownload/assets/web/2p5q/en-us/T505000.xml.html)
(crumpled then flattened aluminum foil) before relying on it. A steady reading
alone cannot establish that a material is reflective or that the calibration is
accurate. This camera's distance and atmospheric corrections still need physical
validation for this procedure.

## Automatic controls and future calibrations

Planned Auto behavior: adapt brightness, contrast, general/spatial/temporal noise
reduction and detail enhancement using image histograms, estimated noise, motion
and edge strength. These mappings still need validation on this camera.
Ambient temperature stays manual and is excluded from automatic calculation.
Distance uses automatic square detection with a measured reference. Emissivity
has the known-temperature fitting workflow above; both still need accuracy
validation against physical references. Reflected temperature has an experimental reflector workflow; optical
transmission remains manual until a suitable reference-based calibration is available. Palettes,
mirrors and overlays remain user preferences. Weather humidity is a possible
optional outdoor estimate, with its source and age shown; it is not a measured
indoor value. Automatic calculation remains a future feature.

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

While the desktop viewer holds a camera connection, it requests temporary display
idle inhibition to prevent monitor blanking from stalling the display loop. The
request is released on exit, including after a camera error. On GNOME this uses
`gnome-session-inhibit --inhibit idle`; other Linux desktops use the ScreenSaver
D-Bus interface, and macOS uses `caffeinate -d`. System power settings are unchanged.
If the desktop refuses the request, the viewer reports the failure.

After reconnecting, the camera may stream before its settings interface is ready.
If initial settings reads fail, the viewer retries them as frames arrive, at most
once per second for 30 seconds. Saved hardware values and calibration commands
are applied only after the complete original settings have been read and saved.
This lets unavailable controls recover without repeated manual restarts.

Desktop USB acquisition runs continuously on a dedicated thread, retaining the
newest complete frame for the UI. Rendering does not stop the camera reads or
build a queue of old images. If no new frame arrives for half a second, the image
and readouts are held while graphs mark measurements invalid and CSV logging
skips them. Window events remain responsive while capture waits for the camera.

On macOS and Linux, camera capture keeps 32 asynchronous USB reads queued so host
scheduling delays do not interrupt capture requests. Completed reads are
resubmitted before their packets are processed. Use `--usb-queue-depth 0` for
diagnostic synchronous capture, or choose a depth from 2 to 128.

## Development

Application-owned Python files, including tests, must stay at or below 800
physical lines. Split larger modules by responsibility into package folders;
`tests/test_repository_layout.py` enforces the limit during the normal test run.
Project skills stay version-controlled under `.agents/skills/`.

The public `desktop.py`, `pipeline.py`, `pipeline_editor.py`, and
`pipeline_processing.py` modules retain the entry points and imports used by
callers. Their implementation lives in these packages:

- `desktop_app/`: a session coordinator with settings, capture, hardware-control,
  and presentation controllers; shared typed session state; toolbar/graph layout,
  spot handling, overlays, dialogs, and capture helpers.
- `pipeline_model/`: node catalog, validation, migration, and branch dependencies.
- `pipeline_ui/`: the Qt editor, widgets, and runtime state updates.
- `processing/`: image operations, per-branch processing and model caches,
  dependency scheduling, and the asynchronous worker.

Controllers use public renderer and hardware APIs to restart measurement
averaging, inspect cached state, and report upload progress. Keep camera reads
and audited hardware setup order intact when changing these boundaries. Qt
popup processes continue to communicate through JSON separately from OpenCV.
Shared test fixtures live in `tests/support/`; recording, pipeline, and ONNX
tests are grouped by responsibility so each test module follows the same limit.

To investigate a Linux viewer stall without reopening its camera, monitor the
running viewer's PID:

```bash
.venv/bin/python tools/watch_ui.py PID --output diagnostics/ui-watch.jsonl
```

The monitor writes one JSON record per second with thread CPU activity, scheduler
waits, memory use, open CSV file sizes, and periodic X11 display power and window
geometry. It stops when the viewer exits or after one hour (`--duration` changes
the limit). CSV buffering and normal waiting can look like inactivity; these are
diagnostic clues rather than measurement counters. Kernel stack availability is
recorded at startup and may be restricted by Linux permissions. The monitor does
not acquire the camera, change power settings, or interrupt the viewer.

For precise diagnostics, launch the desktop viewer with
`--diagnostics diagnostics/viewer.jsonl`. This records received-frame counts,
measurement-validity transitions, and slow stages such as camera acquisition,
rendering and GUI event processing. A watchdog captures all Python thread stacks
in `diagnostics/viewer.stacks.log` when a stage runs for at least two seconds.
This requires starting the viewer with the option; it cannot be attached to an
existing session. A native call holding Python's interpreter lock can delay the
watchdog, so the external monitor remains useful alongside it.
The trace also records USB packet/byte counts, timeouts and frame rejection
reasons, including the most recent rejected length. This distinguishes missing
USB traffic from traffic that does not assemble into a complete thermal frame.

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

## Researching another thermal camera

The repository includes the Codex skill
[reverse-engineer-thermal-camera](.agents/skills/reverse-engineer-thermal-camera/SKILL.md)
for discovering another camera's capture format, radiometry, telemetry and safe
controls, then integrating verified capabilities into this viewer. It includes
our Duo case study, failed leads, restoration constraints, program integration
map, experiment/handoff guidance and an offline evidence-manifest helper.

Invoke `$reverse-engineer-thermal-camera` in a future Codex session working in
this repository. The folder uses the repository-local `.agents/skills` layout
supported by [Codex skill discovery](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).
The skill does not perform camera probes when loaded; live experiments require
the actual target's protocol evidence and the session's authorized scope.
