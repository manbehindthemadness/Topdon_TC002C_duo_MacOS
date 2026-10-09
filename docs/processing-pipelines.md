# Camera processing pipelines

The public pipeline modules keep their import paths while delegating to
`pipeline_model/` (catalog and document validation), `pipeline_ui/` (editor,
widgets, and state updates), and `processing/` (image operations, branch
execution, scheduling, and worker delivery). Each branch owns its enhancement
model caches; the scheduler owns dependency order and branch executors. The
worker preserves revision checks and replaces pending frames with the latest
submission. Display processing does not modify measurement averages or sensor
coordinates.

Camera settings has a hardware stack, a row of tabs A–D, and a software stack
for the selected tab. Click a node header to expand its controls; drag the grip
to reorder. Node titles summarize their settings, such as **App colors: Inferno**
or **Image filter: Sharpen · 0.7**, and update as you edit. Temperature range
titles follow the selected measurement units. Right-click empty space to add nodes or clear the current stack.
Clear all nodes clears hardware and all four software tabs. Right-click a node
to insert above/below or remove it. Image source stays first in every tab and
cannot be removed or bypassed. A also has a fixed Output → viewer node at the
bottom. Hardware controls can appear once; processing and Combine nodes repeat.
While dragging, a highlighted placement bar marks the insertion boundary. The
bar appears only for valid moves within the same stack; Source and Output remain
fixed. It clears when the drag leaves the stack, is cancelled, or is dropped.

The pipeline presets dropdown beside Import, Export, and Restore defaults
offers bundled and user-saved presets. Choose **Save current pipeline as…** to
name and save your own preset; select its name to apply it later. Presets include
the hardware stack and all four software tabs, persist across restarts, and can
still be exported as JSON.
The bundled collection includes Detail Enhanced GPU Upscale, Detail Enhanced CPU
Upscale, Redneck Combat, Redneck Combat GPU, Yautja, Yautja GPU, Reaper Night GPU,
and Reaper Day GPU. The current saved Yautja GPU, Reaper Night/Day GPU, and Detail
Enhanced CPU Upscale configurations are captured as individual exports in `presets/`
and included in the package defaults, so they are available without local preferences.
**Create new** starts a blank, unsaved pipeline: the hardware stack is empty,
tab A contains only Source and Output, and B–D contain only Source. It returns
the editor to tab A and leaves saved presets unchanged.
**Rename current pipeline…**, directly beneath Create new, renames the loaded
preset and selects its new name. The dialog starts with the existing name and
asks before replacing another preset. Saved settings and open edits are preserved;
use Update pipeline to save those edits. Renaming a bundled preset hides its old
name and saves a user preset under the new name without changing bundled files.
**Delete current pipeline…** is available when a preset is loaded. It asks for
confirmation, removes the saved preset, and retains the open pipeline as unsaved
work. Deletions persist across restarts, including bundled presets; bundled files
are unchanged. The name can be saved again later.
The dropdown retains the loaded preset name while editing. **Update pipeline**
re-saves it, including pending input edits; updating a bundled preset creates a
user override. With no preset loaded, **Save pipeline** asks for a name and
selects the newly saved preset. Reopening Camera settings recognizes matching
presets independently of node expansion state. Restoring defaults clears the
association, then recognizes any matching preset. Loading an imported pipeline
selects its saved preset name.
Cancelling a save keeps the previous selection. Save as asks before replacing
an existing name; Update pipeline saves directly. Presets are locked
during logging and calibration measurements, like the other pipeline controls.

Hardware order only organizes the menu. Hardware commands run in an audited,
fixed order, and bypass/removal restores the affected original camera fields.
Software nodes run from top to bottom. Rotation still runs last, using the main
window's existing Rotate button. Spots continue to measure the same physical
sensor pixels through mirror changes and rotation.

Selecting ACNet starts with one pass (2× output). Each extra ACNet pass doubles
both dimensions again; Anime4K09's three default passes refine one 2× output.
For a 512×384 input, three ACNet passes exceed the 4-megapixel processing limit.
Use one or two passes, or choose Native sensor input for three passes. Saved
pipelines retain their explicitly configured pass counts.

## Available nodes

| Camera hardware (one each) | Software (repeatable) |
| --- | --- |
| | Image source (mandatory, first) |
| | Brightness: −100–100%, neutral 0 |
| Processing preset: Balanced, Shadow, Soft | Contrast: 0–3, neutral 1 |
| Brightness and contrast: 0–100 | Gamma: 0.1–3, neutral 1 |
| Gamma: 0–100, neutral 50 | App colors, including white/black hot |
| Boost: Off, Mode 1, Mode 2, Mode 3 | Configurable OpenCV image filters |
| Detail enhancement, amount, Fixed detail checkbox | Horizontal and vertical mirror |
| Camera colors and native palette | Adjustable antialiasing |
| Noise reduction mode and levels | From/To temperature range |
| Relative humidity | Nearest/linear/bicubic/Lanczos interpolation, 1×/2×/4× |
| | Anime4K09 or ACNet enhancement |
| | Combine: another tab, camera preview or raw thermal |
| | Output → viewer (mandatory, last, tab A only) |
| | Pipeline preview (repeatable, active only when expanded) |
| | Edge features: selected connected edges, overlays and masks |
| | Contour regions: selected shapes, outlines, fills and masks |

Auto calibrate, measurement units, ambient
and reflected temperatures, optical transmission, distance, emissivity, and the calibration tools
remain separate. Optical transmission is directly below Reflected temperature.
An active transmission node in older saved preferences migrates to this control;
old pipeline imports omit it without changing the standalone measurement setting.
Calibrate now remains in the image context menu. The former
Advanced / Auto placeholder and Analyze mode switch are removed.

Fixed detail requires enabled detail enhancement, Balanced processing,
gamma 50 and boost Off. Invalid combinations are rejected before writing.
Gamma uploads retain progress and cancellation. Camera controls that affect
only preview are inactive when no connected tab or combine input/mask uses
camera preview; their selected values remain saved for returning to Camera preview. Measurement controls work in
either source. Settings, drag operations and imports are locked during logging
and calibration measurements. Mouse-wheel scrolling never edits inputs.

## Parallel tabs and Combine nodes

Your existing software pipeline becomes tab A. B–D start with independent Image
source nodes; configure them just like A. Selecting a tab edits it, without
changing the viewer output. A always feeds the viewer. A branch runs when
an enabled Combine input or mask connects it, directly or through other tabs,
to A, or while an expanded Preview node inspects it in Camera. Idle branches retain their settings without processing frames.

To mix B into A, add a Combine node in A and select **Tab B** for Blend input.
The node blends B's final image with A's image at that point in the stack;
subsequent A nodes process the combined result. A Combine in B can similarly
use C or D. One tab's output can feed several nodes and is calculated once per
frame. Self-connections and circular chains are rejected, including mask chains.
Bypassing or removing a Combine disconnects that input and mask.

Every active tab has its own worker thread and model cache. Independent branches
run concurrently; consumers wait for their inputs. All branches and masks use
the same captured frame and radiometric snapshot. A frame's final image is
published only when its connected branches finish. Disconnected workers and
model caches are released. Graph and measurement sampling remain independent.

Combine supports Opacity, Weighted sum, Add, Subtract, Difference, Multiply,
Screen, Overlay, Lighten, Darken, bitwise AND/OR/XOR, and Luminance mask. Opacity
sets the overall strength for every mode. Weighted sum exposes separate current
and input weights plus a brightness offset. Luminance mask uses the blend input
as a threshold mask over the current image, with an invert option. These use
OpenCV arithmetic and blending primitives; see [OpenCV's image operations](https://docs.opencv.org/doc/doxygen/html/d0/d86/tutorial_py_image_arithmetics.html).

**Optional mix mask** can select the blend input, camera preview, raw thermal,
or any tab output. A soft luminance mask scales the blend strength from black
(no change) to white (full selected opacity). A binary mask exposes a threshold;
**Invert input**, beside the mask selector, reverses either type. The mask applies to every blend mode.
Selecting a tab as a mask activates that branch even if it isn't a blend input.
Blend input itself can also directly use camera preview or raw thermal.

Raw inputs and masks use their explicit From/To temperature bounds, independent
of scene statistics. These values are stored in Celsius and displayed in your
selected units. Camera preview inputs use the actual camera colors; tab inputs
use that tab's final processed colors. Inputs and masks resize to match the
current image using the node's selected interpolation. This matches dimensions,
not geometric registration: keep mirrors/orientation aligned, or use another
tab to prepare a spatially matching input/mask. Pixel blending changes displayed
colors only; temperatures and CSV measurements always use the sensor data.

## Preview nodes

Right-click the software stack and choose **Add node → Pipeline preview**.
Place it anywhere after Image source and before A's Output. Its image shows
exactly the processing stage at that position, before later nodes and the
main-window rotation. Multiple preview nodes let you compare stages. Right-click
the preview image to toggle fit-width zoom; previews start zoomed to fit width.
Click and drag to pan the zoomed image. Right-click again restores the full-image view. Zoom keeps the preview
height fixed and crops vertically; panning stops at the image edges. A header
counter shows cumulative processing milliseconds in that tab from image decoding
and source preparation through the nodes preceding the preview. Thumbnail
encoding, UI updates, queue delays and waiting for other tabs are excluded;
Combine blending itself is included, while the other tab has its own timing.
The counter updates with the thumbnail and freezes with the image on bypass.
Live updates
retain your zoom and pan. The node header retains its normal context menu.

Expanding a Preview node enables its live thumbnail; collapsing it disables
thumbnail generation. Bypass stops updates and keeps the last displayed frame,
including its current zoom and pan; re-enabling resumes live updates. Node locations and bypass states persist and export with the pipeline. Preview
nodes always start collapsed when Camera opens, including after reopening it.
New Preview nodes also start collapsed; expand one to begin inspecting that stage. Thumbnails preserve aspect ratio and
refresh at up to twice per second while the Camera window is open. They are
read-only taps: inserting, moving or clearing a preview does not change the
image output, temperature measurements, or camera hardware.

An expanded Preview node temporarily activates its tab and any Combine inputs
or masks it needs, even if the tab is disconnected from A. Preview-only branches
run at the thumbnail update rate and keep their model caches between updates.
Closing the last preview in an unconnected tab stops that branch and dependencies
needed solely for it. Closing Camera stops all preview-only branches. Tabs
connected to A continue viewer processing regardless of previews. A failure in
an unconnected preview reports inside the node without interrupting the viewer.

## Static board analysis

Start with Raw thermal image. Add a Temperature range with bounds suited to the
board, followed by Sensor interpolation (Bicubic, 2×), and optionally a Sharpen
filter with a small amount. Add App colors last if you want a palette. White hot
keeps the view grayscale. A fixed range prevents automatic software scaling as
hot/cold objects enter the scene. Without a range, raw display scaling retains
the existing 1st–99th percentile mapping.

The first active range defines the initial thermal normalization even if placed
later in the stack, so earlier image processing survives range adjustments.
Further range nodes remap the current image's display intensities, keeping
previous filtering. After nonlinear adjustments or color conversion, these
intensities no longer promise an exact temperature-to-color correspondence.
Temperatures remain independently measured from the native sensor grid.
Bounds are stored in Celsius and shown in the selected measurement units.

With Camera preview selected, adding a range uses the radiometric source with
an approximation of the selected camera palette. This is **thermal recoloring**,
not a freeze of the hardware preview. Preview hardware effects become inactive unless another connected branch or
combine input/mask still uses the real preview.
App colors lists its standard palettes first and the camera-style palettes last,
using their friendly names. White hot and Black hot each appear once; saved
duplicate selections migrate automatically. The camera-style gradients remain
software approximations of the manufacturer's lookup tables. Removing/bypassing all
range nodes returns to the camera preview.

Brightness, contrast and gamma change luminance while retaining chroma.
App color nodes recolor the previous image's luminance. Repeating filters and
antialiasing adds processing at each location. Interpolation resizes the current
image at its node; the final viewport resize preserves the sensor's aspect.

## CPU image filters

Add an **Image filter** node and choose its Filter. Only the relevant controls
appear. Filter nodes repeat, run in stack order, and save/export their individual
settings. **Blend amount** mixes the filtered image with the incoming image:
0 keeps the original; 1 applies the full result. None passes the image through.
These filters change display pixels, leaving sensor temperatures and CSV data intact.

| Filter | Configuration |
| --- | --- |
| Box blur | Kernel size, border handling, color/luminance channels |
| Edge-preserving denoise (bilateral) | Kernel, color sigma, spatial sigma, borders, channels |
| Median denoise | Kernel, channels |
| Smooth (Gaussian) | Kernel, Gaussian sigma, borders, channels |
| Sharpen | Kernel, Gaussian sigma, sharpening amount, borders, channels |
| Sobel edges | Kernel, X/Y/magnitude direction, response gain, borders |
| Scharr edges | X/Y/magnitude direction, response gain, borders |
| Laplacian edges | Kernel, response gain, borders |
| Canny edges | Aperture/kernel, lower/upper thresholds, accurate L2 gradient |
| Local contrast (CLAHE) | Clip limit, tiles per axis |
| Histogram equalization | Blend amount |
| Threshold / Otsu | Binary/inverted/truncate/to-zero/Otsu method, threshold when applicable, maximum output |
| Adaptive threshold | Mean/Gaussian method, block size, offset C, invert, maximum output |
| Morphology | Erode/dilate/open/close/gradient/top-hat/black-hat, kernel shape/size, iterations, borders, channels |
| Emboss | Direction, response gain, neutral offset, borders |
| High-pass detail | Kernel, Gaussian sigma, response gain, neutral offset, borders, channels |

Auto kernel uses 3×3 for most filters and 5×5 for bilateral. Gaussian-based
filters calculate their kernel from sigma when Auto is selected. Median and
bilateral kernels are limited to 9×9; derivative/Canny kernels to 7×7; others
allow up to 21×21. Unsupported kernel choices are disabled. Adaptive threshold
has its own odd block size, up to 31×31. Morphology allows 1–5 iterations.
Border choices are reflected pixels, repeated edge pixels, or constant zero.

Edges, thresholds and emboss produce grayscale images, which can then feed
App colors or a Combine mask. CLAHE and histogram equalization alter luminance
while preserving chroma. Other filters offer all-channel or luminance-only
processing. OpenCV's 8-bit operators round the incoming display values before
processing; the radiometric sensor values remain separate.

These use standard CPU OpenCV operations without extra models or contrib modules.
Cost increases with image dimensions, kernel size, repeated nodes and active tabs;
use denoising before upsampling when possible. See the official
[filter reference](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html),
[threshold reference](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html),
and [CLAHE reference](https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html).

## Feature processors

Right-click the software stack → **Add node → Edge features** or
**Contour regions**. These are repeatable processors with their own controls,
bypass states and saved/exported settings. Place a Preview node after them to
inspect the selection. They work on the incoming displayed image in either
camera-preview or raw-thermal pipelines, and never change temperatures or spots.

**Edge features** detects Canny, Sobel or Scharr edges. Automatic Canny thresholds
adapt to the smoothed image's nonzero gradient strengths; turn Auto off to set
lower/upper thresholds. Sobel and Scharr expose a minimum gradient strength
instead. Direction can select all edges, horizontal edges or vertical edges.
Minimum connected edge pixels suppresses speckles; Select by keeps the longest
connected components or those nearest the image center. Maximum selected features
limits the retained components. A connected component may contain several joined
edges; its pixel count measures its size rather than a geometric line length.
Edge thickness dilates the selected mask at detection resolution.

**Contour regions** segments with Otsu automatic threshold, a manual threshold,
adaptive threshold, or Canny closed boundaries. Select dark regions reverses
threshold-based segmentation. Adaptive block size and offset C control local
thresholding; zero offset is the default. Close gaps optionally joins small breaks
before detection. It detects external silhouettes, filling enclosed holes in
mask/fill outputs. An open boundary might not yield a useful filled region.

Filter regions by area as a percentage of the detection image, longest/shortest
side aspect ratio, solidity (area divided by convex-hull area), and circularity
(4π × area / perimeter²). Rectangle selection requires four approximately
right-angle corners and a well-filled rotated bounding box; round regions require
at least six polygon vertices, circularity ≥0.75 and side ratio ≤1.3. Convex
selection checks the simplified polygon. These are geometric heuristics, not
component recognition. Polygon simplification is a percentage of perimeter.
Aspect ratios use a rotated bounding box so rotating a rectangular target does
not make it appear square.

Rank matching regions by largest, smallest, brightest, darkest, or closest to
center, then keep up to Maximum selected features. Brightness uses the original
incoming display colors, not temperature: put this node before App colors, using
White hot raw input when you want hotter regions to look brighter. Rankings are
recalculated each frame and do not track object identities. Original contour,
simplified polygon, convex hull or rotated box chooses the rendered geometry;
selection metrics always describe the detected contour.

Both nodes offer a centered selection region and image-border exclusion. Edges
can overlay a chosen color, emit a binary mask, or retain input pixels at selected
edges against black. Contours can overlay outlines, tint filled regions, emit a
binary region mask, isolate selected regions against black, or flatten each region
to its average input color. Flattening uses color samples at detection resolution
and retains the image outside selected regions. A mask needs Blend amount = 1
for purely black/white output; smaller amounts blend in the original image.

Detection keeps the input aspect ratio and uses at most 256, 512 (default), or
1024 pixels along the longest side; smaller inputs stay native. Geometry and masks
map back to the current pipeline image size. Smoothing, close-gap kernels and
minimum connected edge pixels use detection pixels; area percentages remain
relative to the full detection image even for a centered search. Downsampling
can miss very small features. Selection is limited to 256 objects; contour metrics
inspect at most the 2048 largest candidate contours to bound noisy-frame work.

For board outlines, try Raw thermal → fixed Temperature range → light denoise →
Edge features (Auto Canny, minimum 10 connected pixels) → Preview. For component
regions, replace edges with Contour regions, select Rectangles / squares, then
adjust area and aspect-ratio limits while inspecting the preview.

For selective processing, build a contour pipeline in B with **Binary region
mask** output and Blend amount 1. In A, add a Combine and choose **Tab B output**
for Optional mix mask. White regions allow blending; black regions preserve A's
current image. Keep the mask's orientation aligned with A and the blend input.
A Preview in B lets you build the mask before connecting it; existing branch
activation and Camera-close rules still apply.

Implementation uses standard CPU OpenCV
[contour and connected-component operations](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)
and [Canny edge detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).

## Enhancement and performance

The normal **AI enhancement** node exposes execution and **Compute devices**
controls when startup detects Metal plus the Core ML runtime, for ACNet and ONNX
models. Anime4K09 is CPU-only and hides them. Changing the execution backend never
changes the selected model. With CPU execution, **Compute devices** displays
**CPU only** and is disabled. Selecting Apple Core ML enables the selector and
restores the saved device choice. Apple preferences remain stored but inactive
when switching to Anime4K09. Returning to ACNet restores these preferences.
The same input, denoising, amount and pass controls work with either ACNet backend.
Each pass doubles both dimensions. ACNet passes are clamped to the 4-megapixel
budget: up to two passes for 512×384 input, three for 256×192, fewer after earlier
upscaling. The spinbox and slider follow the input/model; Anime4K09 retains 1–5
refinement passes. Execution also clamps older/imported excessive ACNet requests.
If even one pass cannot fit, reduce preceding scales or choose Native input.

Execution defaults to CPU for new nodes and older presets. Backend and Apple
device preferences are always saved/exported, including on Ubuntu. When the
preferred GPU/runtime is unavailable, the normal node tries the other available
GPU, then CPU ACNet, without altering the saved preferences. Apple compute-device
controls appear when Core ML is the effective GPU backend. Returning the same
preset to a capable Mac restores the Apple choice. Actual Apple runtime errors
still appear in Camera status rather than
silently switching backends. The older standalone Apple node remains supported.

### NVIDIA CUDA execution

On a Linux NVIDIA system with a working GPU runtime, the execution selector
also offers **NVIDIA CUDA** for ACNet and ONNX models. CUDA uses GPU zero;
Apple compute-device choices remain saved and are hidden while CUDA is the
effective backend.
ACNet uses the original bundled ONNX weights with the same denoising levels,
passes, amount blending, and size limit as CPU execution. Anime4K09 remains CPU.
The startup probe contains native failures and confirms an actual small CUDA
inference before exposing controls. Unavailable saved CUDA settings try Apple
Core ML before CPU, and unavailable Apple settings try CUDA before CPU. Explicit
CPU selections remain CPU. The status badge shows the effective fallback; errors
from an active GPU session appear in Camera status. Ordinary filters
and radiometry retain their CPU implementation.

See [JetPack installation and verification](nvidia-jetpack.md), including the
CUDA 13 runtime replacement and `uv run --no-sync` workflow.

### Experimental Apple Core ML node

On macOS, `uv sync` automatically installs the prebuilt Apple runtime; Linux does
not install it. Launch with `uv run topdon-duo-desktop`. At startup a short,
isolated probe checks `MTLCreateSystemDefaultDevice` and the Core ML provider.
Apple execution controls in **AI enhancement** are exposed when both are
available; Camera controls show readiness or the reason it is unavailable. Detection does not
rewrite saved pipelines or presets, and model loading stays lazy until the node
is used. The older `--extra apple` commands remain supported. Add **AI enhancement**,
select **ACNet**, then choose **Apple Core ML** execution. No custom OpenCV build
is required. The separate experimental Apple node is no longer offered in Add/Insert
menus; the following describes its legacy runtime for existing saved pipelines.

This separate node performs one 2× ACNet pass, defaults to Native sensor input,
and offers denoising, strength, and CPU + GPU / CPU + Neural Engine / all-device
choices. It uses a persistent spawned helper process with its own lazily loaded
ONNX Runtime session and does not alter OpenCV backend settings. Native runtime
crashes become a Camera status error rather than terminating the viewer. Bypass
the node to continue with CPU processing. Compilation/inference has a 45-second
timeout. MLProgram requires macOS 12 or newer.

The Apple node materializes the model's default zero Conv and ConvTranspose padding
in memory to work around ONNX Runtime 1.30's missing-padding crash and missing
Core ML `pad` parameter during compilation. Bundled CPU model
files are untouched; regression tests compare the transformed model against the
original model and upstream ACNet reference output.

Core ML chooses the actual hardware; enabling its provider does not prove GPU
execution or a speedup. Unsupported operators may execute on this node's CPU
provider. Initialization can be slow, so compare warm-frame pipeline timings
against the existing CPU ACNet node. This is an experimental path, not a verified
performance improvement. See the [Core ML provider documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).

Without the runtime, or on Linux, an active saved Apple node reports an error
instead of silently becoming a CPU node. Bypass/delete it to use the portable
pipeline. Presets containing it require the runtime on macOS or bypassing the node
on Ubuntu. Display enhancement never changes saved radiometric measurements.

AI nodes default to their current input image. Native or Preview explicitly
resizes the preceding result to 256×192 or 512×384 before enhancement.
Anime4K09's 1–5 passes refine one 2× output; each ACNet pass produces another
2× output. ACNet offers none/light/medium/strong denoising. Amount controls
enhancement strength. Models load lazily and are reused per node/model.

Image processing uses a coordinator and one worker per connected tab, with only
the latest pending frame.
Old pipeline-revision results are ignored. Temperature sampling, graphs and UI
input continue while an image is being processed. Camera status shows latency
and warns above 500 ms; slow pipelines are allowed. Intermediate images are
limited to four megapixels, and oversize chains report an error instead of
skipping a node. Enhancement changes display detail, not sensor resolution.

## Saving and sharing

Version 1–3 pipeline files automatically migrate to version 4. Version 1 and 2 migration keeps the existing tab layout: their software
stack becomes A with its original node identities/settings and a fixed output;
B–D start with an Image source each. Version 1 also moves Image source from the
hardware stack to the first software position. Version 4 adds filter parameters; older
filters retain their kernel, strength and Gaussian/bilateral settings. No nodes
are added to existing pipelines unless required by the older layout migration.
Nodes, order, bypass, parameters and expanded states save with viewer preferences
and reload on startup. Existing display preferences migrate once into a minimal
pipeline, retaining configured hardware and enhancement choices. Existing Analyze
preferences become range/interpolation/filter/enhancement/color nodes.

Import pipelines and Export pipelines use `.pipeline.json` and remember separate
directories. Their dialogs offer per-pipeline checkboxes, Select all, and Clear
selection. Export includes the current pipeline with pending edits and any checked
saved presets. When the current pipeline is a loaded preset, its current edits
represent that preset in the checklist. Exporting only an unnamed current pipeline
retains the original individual-document format; named or multiple selections use
a bundle with format `topdon-duo-pipelines`, version 1, and a `pipelines` mapping
from names to complete documents. Each document contains hardware and all four
software tabs, not units, spots, logging or calibrations.

Import accepts bundles and original individual files, and permits selecting several
files together. Individual files use their filename as the suggested preset name.
The checklist offers Rename, Replace, or Skip for conflicts. Rename suggests a
unique suffix and permits editing the name; Replace explicitly uses the incoming
name. Unresolved duplicate or empty names block import. Invalid files are reported;
valid files remain available to review. The selected presets are saved together
before the dropdown changes. Cancellation or a save failure leaves existing presets
and the active pipeline intact. Load first selected pipeline after import is checked
by default; uncheck it to retain the active pipeline. A hardware failure restores
previously owned fields without resetting calibration fields.

Clear current stack and Clear all retain each Image source and A’s fixed output. Restore default pipeline
returns to Camera preview, Inferno app colors and antialiasing. Restore camera
settings restores the original hardware and bypasses its optional nodes, keeping
the software stack and saved calibration references. Exit restores physical
camera overrides while retaining the pipeline choices for the next launch.
