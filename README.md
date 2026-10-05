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

**Show graph** doubles the main window width, keeping its height and displaying
thermal history on the right. **Hide graph** halves the width again. The toggle
is remembered across restarts. The master graph plots minimum, average, maximum
and center temperatures; each placed spot gets its own graph in placement order.
All graphs share the available height evenly and show a rolling 60-second history
in the selected Celsius/Fahrenheit unit. Sampling and rendering run on a separate
worker thread every 0.5 seconds by default, using the latest camera measurements without
queuing video frames. Hiding the graph area pauses sampling and rendering;
reopening resumes, with gaps rather than connecting lines across long pauses.
Spot history survives rotation and mirroring, and clears when the spots are
cleared. Histories are session-only; disabling a spot removes its graph history,
and enabling it starts fresh sampling. Captures and recordings can include the
graph pane with the Capture dialog’s **Include graphs when visible** checkbox.

**Update (s)**, immediately left of **Log to CSV**, sets the graph and CSV sampling
interval from 0.1 to 60 seconds. Click the number and type a value; press Enter to
apply or Escape to cancel. The interval is remembered across restarts and is
locked while logging or choosing a log file.

**Log to CSV** opens a Save dialog before logging begins. **Stop logging** closes
and saves the file; **Cancel logging** cancels an unfinished Save dialog. Hiding
the graph area is disabled while logging or choosing its file. During logging,
Camera inputs, reset/restore buttons and Advanced / Auto are disabled. Main-window
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
a name restores the default “Spot N”. Names are session-only, like spot positions.
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

The Capture popup's **Include graphs when visible** checkbox optionally appends
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
and groups display, AI enhancement, and camera adjustments.
Display preferences, selected temperature unit, rotation, Auto calibrate and Advanced / Auto states
and successfully applied hardware overrides are saved as `settings.json` beside
`main-window.json` in the application configuration directory. Changes save as
they are applied and on exit. Saved overrides are reapplied after reading and
preserving the camera baseline on the next launch; the physical baseline is
still restored on exit. Restore camera settings also clears saved hardware
overrides; Reset display settings saves the display defaults. Explicit
`--rotate` and `--image-source` options override remembered values. Invalid saved
fields are ignored independently so valid preferences can still load.
**Auto calibrate** is at the top of Camera settings and defaults to off. Its
saved choice is applied on startup. After the first valid camera frame arrives,
the viewer requests one calibration, including when Auto calibrate is off.
Readings are held during the calibration as usual. Right-click the thermal image and choose
**Calibrate now** to request one calibration. Both controls are disabled while
logging or measuring a calibration reference. Automatic camera calibration is
re-enabled when the viewer exits; the saved switch choice is retained for the
next launch. Restore camera settings and Reset display settings keep this choice.

Each control has a title and editable value; only numeric ranges have sliders.
The **Units** toggle and Camera's **Measurement units** selection switch together:
metric uses Celsius and centimeters; imperial uses Fahrenheit and inches. Ambient,
reflected-temperature and distance inputs convert their values, ranges and
steps accordingly. Distance sliders retain the camera's 1 cm precision, and
pending edits survive unit changes. Hardware writes and saved hardware overrides
remain in Celsius and meters, so switching units does not change calibration.
Choice and on/off controls use dropdowns. On/off settings use two columns within Display controls, above
AI enhancement and camera adjustments. All available inputs stay enabled. The
**Advanced / Auto** checkbox is reserved for future automatic control behavior. Hardware rows include ambient and reflected temperature,
distance, emissivity, humidity, optical transmission, center overlay, brightness,
contrast, noise reduction mode and levels, detail enhancement, and camera palettes.
Use **Restore camera settings** to restore all original camera values. Quitting also restores camera overrides.
Original payloads are saved under `$XDG_STATE_HOME/topdon-duo/camera-baselines`
(default `~/.local/state/topdon-duo/camera-baselines`) before controls are enabled.

**Camera → Enhancement algorithm** offers **Anime4K09 2×** (the algorithm selected
by the inspected Android phone app), plus **ACNet 2×** with no, light, medium,
or strong denoising, experimental **TIDY thermal denoise**, and **DnCNN blind denoise**.
It defaults to Off. **Enhancement input size** selects
native sensor size (the default, 256×192 → 512×384) or the full preview.
**Enhancement amount** blends ACNet output with ordinary interpolation, or
adjusts Anime4K09's gradient strength; 0 gives the original image and 1 gives
full enhancement. TIDY and DnCNN also blend their output with the original input at the
selected amount. **Anime4K09 passes** offers 1–5 passes; the phone uses 3.

Anime4K09 uses a portable CPU port of the public OpenCL rules with the phone's
parameters. ACNet uses four bundled ONNX models through the existing OpenCV CPU
engine. Neither needs downloads, compilation, or extra runtime dependencies.
Models load when selected and are reused across frames.
The Camera status shows processing time or a fallback error.
Enhancement also appears in saved images and recordings, while the temperature
grid, spot positions, and temperature readings keep their original resolution.
The existing display scale controls the final image size; Anime4K09 and ACNet run at 2×,
while TIDY and DnCNN retain their input resolution, before that final resize. Native mode downsamples enlarged previews before
enhancement. Full preview mode retains their complete input resolution.
Processing may reduce the live frame rate, especially for the full 512×384 preview.
The portable Anime4K09 port has been checked against the public kernel rules;
identical results to a phone GPU have not been established.

**TIDY setup:** Select **TIDY — thermal denoise (experimental)**, then use
**TIDY ONNX model file → Browse…** to choose an exported `tidy.onnx`. The path and
enhancement settings persist across restarts. TIDY uses the existing OpenCV CPU
backend; no PyTorch or ONNX Runtime installation is needed to view an exported
model. A missing or incompatible file shows an error and keeps the original
image visible. Switch modes or choose another file to retry.

Obtain weights and export instructions from the
[official TIDY repository](https://github.com/williamrheeth/TIDY). Export is a
one-time development step requiring PyTorch and ONNX, using `python -m
tidy.deployment onnx --output /usr/src/TIDY/artifacts/tidy.onnx` from that source
tree. Verify the checkpoint against its supplied `weights/SHA256SUMS` before
exporting. The exported ONNX file is portable; it does not need recompilation
for Linux versus macOS. Keep the model outside this repository's bundled model
directory. Upstream supplies it under **CC BY-NC 4.0**, and the roughly 443 MiB
artifact is not bundled with this application.

TIDY processes normalized thermal intensity before the app palette. For a camera
color preview it denoises replicated luminance and preserves the original chroma.
It targets noise and stripes, and may also soften small details; it does not
increase sensor resolution or alter measured temperatures. Native sensor size
is recommended for interactive use. In a local Intel i5-7267U test, native
256×192 inference took about 0.5 seconds through OpenCV; ONNX Runtime took about
0.3 seconds, versus 1.3–1.6 seconds for a 512×384 input. These are inference-only
timings, not guaranteed live frame rates. OpenCV output matched the official
PyTorch execution to within approximately 0.000003 on the tested normalized
frame; macOS performance has not been measured.

**DnCNN experiment:** Select **DnCNN — blind denoise (experimental)** under
**Camera → Enhancement algorithm**. The bundled grayscale blind model uses the
same OpenCV CPU engine and needs no model-file selection or extra dependencies.
It estimates and subtracts noise from normalized thermal intensity before the
app palette; camera color previews retain their chroma. **Enhancement amount**
blends with the original intensity, and **Native sensor size** is recommended.
This model was trained for Gaussian image noise, rather than thermal stripe
artifacts, so changes can be subtle on already smooth images.

The ONNX graph was exported from the authors'
[KAIR grayscale blind checkpoint](https://github.com/cszn/KAIR/blob/master/main_test_dncnn.py),
including the residual subtraction. The roughly 2.5 MiB model is MIT-licensed;
attribution and the license accompany it. `tools/export_dncnn.py` reproduces the
export with developer-only PyTorch and ONNX dependencies. On the same saved
thermal frame used for TIDY, OpenCV inference took 0.49–0.50 seconds at native
size and 1.90–2.04 seconds at 512×384. The output matched the official PyTorch
model within 0.000003 on normalized pixels. This single-frame comparison has no
clean thermal reference and does not establish improved measurement accuracy;
the app's temperature measurements remain unchanged.

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
