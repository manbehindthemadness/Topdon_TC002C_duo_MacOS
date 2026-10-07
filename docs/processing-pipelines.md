# Camera processing pipelines

Camera settings now has two stacks. Click a node header to expand its controls;
drag the grip to reorder. Right-click empty stack space to add a node, clear that
stack, or clear both stacks. Right-click a node to insert above/below or remove
it. The image source stays first and cannot be removed or bypassed. Camera
controls can appear once, including bypassed nodes; software nodes can repeat.

Hardware order only organizes the menu. Hardware commands run in an audited,
fixed order, and bypass/removal restores the affected original camera fields.
Software nodes run from top to bottom. Rotation still runs last, using the main
window's existing Rotate button. Spots continue to measure the same physical
sensor pixels through mirror changes and rotation.

## Available nodes

| Camera hardware (one each) | Software (repeatable) |
| --- | --- |
| Image source (mandatory) | Brightness: −100–100%, neutral 0 |
| Processing preset: Balanced, Shadow, Soft | Contrast: 0–3, neutral 1 |
| Brightness and contrast: 0–100 | Gamma: 0.1–3, neutral 1 |
| Gamma: 0–100, neutral 50 | App colors, including white/black hot |
| Boost: Off, Mode 1, Mode 2, Mode 3 | None/bilateral/median/Gaussian/sharpen filter |
| Detail enhancement, amount, Fixed detail checkbox | Horizontal and vertical mirror |
| Camera colors and native palette | Adjustable antialiasing |
| Noise reduction mode and levels | From/To temperature range |
| Relative humidity | Nearest/linear/bicubic/Lanczos interpolation, 1×/2×/4× |
| | Anime4K09 or ACNet enhancement |

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
only preview are inactive for raw thermal rendering; their selected values
remain saved for returning to Camera preview. Measurement controls work in
either source. Settings, drag operations and imports are locked during logging
and calibration measurements. Mouse-wheel scrolling never edits inputs.

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
not a freeze of the hardware preview. Preview hardware effects become inactive.
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

Image processing runs in a separate worker with only the latest pending frame.
Old pipeline-revision results are ignored. Temperature sampling, graphs and UI
input continue while an image is being processed. Camera status shows latency
and warns above 500 ms; slow pipelines are allowed. Intermediate images are
limited to four megapixels, and oversize chains report an error instead of
skipping a node. Enhancement changes display detail, not sensor resolution.

## Saving and sharing

Nodes, order, bypass, parameters and expanded states save with viewer preferences
and reload on startup. Existing display preferences migrate once into a minimal
pipeline, retaining configured hardware and enhancement choices. Existing Analyze
preferences become range/interpolation/filter/enhancement/color nodes.

Import and Export use `.pipeline.json` and remember separate directories. A file
contains only the two pipelines and format version, not units, spots, logging
or calibrations. Imports replace both stacks only after full validation. An
invalid document leaves the current configuration intact. A hardware failure
restores previously owned fields without resetting calibration fields.

Clear current stack and Clear all retain Image source. Restore default pipeline
returns to Camera preview, Inferno app colors and antialiasing. Restore camera
settings restores the original hardware and bypasses its optional nodes, keeping
the software stack and saved calibration references. Exit restores physical
camera overrides while retaining the pipeline choices for the next launch.
