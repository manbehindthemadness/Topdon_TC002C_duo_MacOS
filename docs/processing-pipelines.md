# Camera processing pipelines

Camera settings has a hardware stack, a row of tabs A–D, and a software stack
for the selected tab. Click a node header to expand its controls; drag the grip
to reorder. Right-click empty space to add nodes or clear the current stack.
Clear all nodes clears hardware and all four software tabs. Right-click a node
to insert above/below or remove it. Image source stays first in every tab and
cannot be removed or bypassed. A also has a fixed Output → viewer node at the
bottom. Hardware controls can appear once; processing and Combine nodes repeat.

Hardware order only organizes the menu. Hardware commands run in an audited,
fixed order, and bypass/removal restores the affected original camera fields.
Software nodes run from top to bottom. Rotation still runs last, using the main
window's existing Rotate button. Spots continue to measure the same physical
sensor pixels through mirror changes and rotation.

## Available nodes

| Camera hardware (one each) | Software (repeatable) |
| --- | --- |
| | Image source (mandatory, first) |
| | Brightness: −100–100%, neutral 0 |
| Processing preset: Balanced, Shadow, Soft | Contrast: 0–3, neutral 1 |
| Brightness and contrast: 0–100 | Gamma: 0.1–3, neutral 1 |
| Gamma: 0–100, neutral 50 | App colors, including white/black hot |
| Boost: Off, Mode 1, Mode 2, Mode 3 | None/bilateral/median/Gaussian/sharpen filter |
| Detail enhancement, amount, Fixed detail checkbox | Horizontal and vertical mirror |
| Camera colors and native palette | Adjustable antialiasing |
| Noise reduction mode and levels | From/To temperature range |
| Relative humidity | Nearest/linear/bicubic/Lanczos interpolation, 1×/2×/4× |
| | Anime4K09 or ACNet enhancement |
| | Combine: another tab, camera preview or raw thermal |
| | Output → viewer (mandatory, last, tab A only) |
| | Pipeline preview (repeatable, active only when expanded) |

Auto calibrate, camera measurement overlay, measurement units, ambient
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
Invert mix mask reverses either type. The mask applies to every blend mode.
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
the preview image to toggle fit-width zoom; click and drag to pan the zoomed
image. Right-click again restores the full-image view. Zoom keeps the preview
height fixed and crops vertically; panning stops at the image edges. Live updates
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
The camera-style software gradients are explicitly labeled approximations;
they are not the manufacturer's exact lookup tables. Removing/bypassing all
range nodes returns to the camera preview.

Brightness, contrast and gamma change luminance while retaining chroma.
App color nodes recolor the previous image's luminance. Repeating filters and
antialiasing adds processing at each location. Interpolation resizes the current
image at its node; the final viewport resize preserves the sensor's aspect.

## Enhancement and performance

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

Version 1 and 2 pipeline files automatically migrate to version 3: their software
stack becomes A with its original node identities/settings and a fixed output;
B–D start with an Image source each. Version 1 also moves Image source from the
hardware stack to the first software position.
Nodes, order, bypass, parameters and expanded states save with viewer preferences
and reload on startup. Existing display preferences migrate once into a minimal
pipeline, retaining configured hardware and enhancement choices. Existing Analyze
preferences become range/interpolation/filter/enhancement/color nodes.

Import and Export use `.pipeline.json` and remember separate directories. A file
contains hardware, all four software tabs and format version, not units, spots,
logging or calibrations. Imports replace the complete pipeline only after full validation. An
invalid document leaves the current configuration intact. A hardware failure
restores previously owned fields without resetting calibration fields.

Clear current stack and Clear all retain each Image source and A’s fixed output. Restore default pipeline
returns to Camera preview, Inferno app colors and antialiasing. Restore camera
settings restores the original hardware and bypasses its optional nodes, keeping
the software stack and saved calibration references. Exit restores physical
camera overrides while retaining the pipeline choices for the next launch.
