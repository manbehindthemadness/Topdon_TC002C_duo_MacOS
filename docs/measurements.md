# Measurements

[Measurements how-to](measurements-guide.md) · [Documentation home](../README.md)

The native temperature plane is the source for every measurement. Resizing,
palettes, custom nodes, software enhancement and camera-preview resolution do
not change its geometry or Celsius values. Pipeline transforms and mirrors are
ignored for sampling; measurements stay on the underlying thermal canvas.
The popup displays coordinates in the current viewer-rotated native grid;
persistence reverses that orientation so saved shapes stay on the same physical
sensor pixels.

Use **Measurements** or right-click the thermal image. Pick a shape and **Create**,
then edit its name, coordinates, label fields and exclusion setting. Select
**Apply configuration** to commit; switching selection applies valid edits too.
The coordinate grid supports dragging handles and numeric entry. `P` and the
popup placement checkbox retain the quick image-click workflow for point spots.
Dragging an area in the image translates its whole geometry, keeping every
vertex/control handle inside the sensor. Existing measurement and calibration
locks apply to all mutations.

## Drawing on live video

The viewer's **Rotate** popup selects preview rotation independently from its
portrait/landscape layout. Portrait puts graphs below the video. Both layouts
map drawing and dragging back to native sensor coordinates and exclude graph
pixels and camera padding from measurement input. Rotation preserves saved
measurement identity and sensor coverage; changing layout cancels an unfinished
drawing without creating a measurement.

Choose the shape and **Create** in the popup; **Exclusion** chooses
subtraction behavior. **Auto-hide** is checked by default and hides the popup
during creation; uncheck it to leave the dialog visible. The viewer gains
contextual drawing controls. Completed shapes reopen the popup for names, labels
and detailed edits. **Close**, next to **Apply configuration**, dismisses the
dialog without applying pending edits and remains available while edits are locked.
Releasing a spot/rectangle/oval drag leaves a fixed draft; press Enter to save it,
or drag again to replace its geometry before completing it.

| Tool             | Gesture                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| Spot             | Drag from center to radius; a click makes a radius-zero spot                                         |
| Rectangle / oval | Drag opposite corners of the bounding rectangle                                                      |
| Polygon          | Click successive vertices; at least three are required                                               |
| Bézier           | Click-and-drag successive anchors to set tangent direction/length; at least two anchors are required |

For every shape, **Finish**, Enter or right-click closes the shape. macOS
Control-click also finishes. **Undo**/Backspace removes the last anchor or cancels
the current drag; **Cancel**/Esc discards the draft without creating a measurement.
In fullscreen, this first Esc cancels drawing and preserves fullscreen. Polygon
hovering previews the next vertex; a Bézier drag previews its tangent and curve.
Curve handles are clamped inside the native sensor grid and remain editable in
the popup after creation.

The parent maps the displayed camera pane back to native coordinates, including
resizing and the adjoining graph pane. Graph pixels and letterbox/padding clicks
are not measurement input. Releasing a drag outside the camera discards that
gesture. Drawing takes precedence over ordinary point placement and body dragging;
the original placement toggle is retained afterward. Logging/calibration locks and
viewer rotation cancel unfinished drawings. Drafts exist only in live
presentation: they are not sampled, graphed, recorded or saved as measurements.

## Geometry and sampling

| Shape     | Geometry                                                           | Kernel                                                       |
| --------- | ------------------------------------------------------------------ | ------------------------------------------------------------ |
| Spot      | Integer center; linked radius/diameter, including fractional radii | Pixel centers whose Euclidean distance is at most the radius |
| Rectangle | Two opposite corners                                               | Pixel centers inside the inclusive axis-aligned bounds       |
| Oval      | Two opposite corners of its bounding rectangle                     | Pixel centers inside the ellipse, including the boundary     |
| Polygon   | At least three vertices; concave shapes allowed                    | Even-odd containment, including boundary centers             |
| Bézier    | Closed cubic segments                                              | Same containment rule on a tessellated curve                 |

Bézier coordinates repeat **Anchor, Control 1, Control 2**. The next anchor ends
the current segment; the final segment returns to the first anchor. **Add vertex
/ Bézier segment** adds a complete triple for a curve, and removal deletes a
complete triple. A curve needs at least two segments. Sampling tessellates each
cubic into 48 edges; control coordinates allow subpixel precision. Measurement
paths support at most 192 vertices/handles and 1,000 saved measurements.

The app caches native masks until geometry, position or orientation changes.
It unions all enabled exclusions and removes those pixels from every enabled
user measurement, including radius-zero spots. Disabled exclusions do not
subtract. Exclusions do not affect scene/master statistics, cursor readings,
raw sensor arrays, calibration or captured radiometric data. Overlapping
measurement areas remain independent; only exclusion shapes subtract.

Finite remaining samples supply minimum, maximum and arithmetic average.
Radius-zero spots retain their scalar graph; positive-radius spots and areas
have Min/Avg/Max curves. Empty kernels produce unavailable values, and invalid
camera frames retain held readings while graph histories mark gaps and CSV
logging skips the frame. Exclusion shapes never produce measurement series.

## Labels, histories and storage

Per-measurement checkboxes select number, name, min, max and average. Unchecking
all fields hides the label while retaining its outline. Area leaders attach to
the outer edge; the shared placer reserves shape bounds, point markers and
previous labels. When every candidate overlaps, it chooses the least overlap.
Exclusions use a dashed border and an exclusion identifier without temperatures.

Stable numbers and graph identities survive edits, movement and orientation.
Switching between a scalar spot and an area resets that measurement's history
to avoid mixing one- and three-column samples. Disabling, deleting or converting
a measurement to an exclusion removes its active graph. **Clear selected graph
history** affects only that measurement. Clear all removes geometry and resets
numbering; histories are not persisted across launches.

CSV files keep canonical Celsius and selected-unit values. Area rows have IDs
`spot.<generation>.<number>.min`, `.avg`, `.max`. Point rows keep the previous
`spot.<generation>.<number>` format. Names appear in graph titles and CSV labels.

Preferences retain the existing `spots` key. Legacy point-only version 1 files
remain readable. Version 2 items add a validated `region` object with `kind`,
`radius`, `points`, `exclude` and `labels`. Geometry is validated without running
package code or opening hardware. The Qt 6 popup receives only JSON metadata;
the parent process owns all kernels, temperatures, persistence and lock checks.
