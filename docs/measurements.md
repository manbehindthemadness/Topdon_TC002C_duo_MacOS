# Measurements

The native temperature plane is the source for every measurement. Resizing,
palettes, custom nodes, software enhancement and camera-preview resolution do
not change its geometry or Celsius values. The popup displays coordinates in
the current rotated/mirrored native grid; persistence reverses that orientation
so saved shapes stay on the same physical sensor pixels.

Use **Measurements** or right-click the thermal image. Pick a shape and **Create**,
then edit its name, coordinates, label fields and exclusion setting. Select
**Apply configuration** to commit; switching selection applies valid edits too.
The coordinate grid supports dragging handles and numeric entry. `P` and the
popup placement checkbox retain the quick image-click workflow for point spots.
Dragging an area in the image translates its whole geometry, keeping every
vertex/control handle inside the sensor. Existing measurement and calibration
locks apply to all mutations.

## Geometry and sampling

| Shape | Geometry | Kernel |
| --- | --- | --- |
| Spot | Integer center; linked radius/diameter, including fractional radii | Pixel centers whose Euclidean distance is at most the radius |
| Rectangle | Two opposite corners | Pixel centers inside the inclusive axis-aligned bounds |
| Oval | Two opposite corners of its bounding rectangle | Pixel centers inside the ellipse, including the boundary |
| Polygon | At least three vertices; concave shapes allowed | Even-odd containment, including boundary centers |
| Bézier | Closed cubic segments | Same containment rule on a tessellated curve |

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
