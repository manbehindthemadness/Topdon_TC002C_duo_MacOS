# Classless YOLO

Standalone example package for **Custom → Open module folder…**. Select this
`Classless YOLO` directory; the node title includes that name. This example is
not a built-in node or default preset. It processes the current pipeline image.

It adapts the classless localization stage of
[Inspector](https://github.com/manbehindthemadness/inspector/tree/7717527dc3193ca71824d2f68f43f4b5cc597fc1).
Inspector uses Mobile Object Localizer, then a separate YOLOv10
classifier. **Classless YOLO** is the requested example name; this package runs
the localizer alone and displays boxes and confidence scores without class names.

## Model setup

Use the **four-output** `saved_model/model_float32.onnx` export from
[PINTO's Mobile Object Localizer model folder](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer).
Its [reference inference code](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/151_object_detection_mobile_object_localizer/demo/demo_onnx.py)
uses float32 RGB pixels in 0–255, NCHW `1×3×192×192`, and outputs normalized
`y1,x1,y2,x2` boxes, classes, scores, and count. The class tensor is ignored.
Two-output exports and Inspector's `.pt` weights are incompatible.

With `model_path` left empty, the package requests this model through the viewer's
shared background downloader. `MODEL_SOURCE` in `__init__.py` pins the archive checksum, exact model
member and model checksum. Progress appears in the existing download status;
frames pass through unchanged until installation finishes. Cached weights are
reused offline. A failed download reports a pipeline error; bypass/re-enable the
node to retry. Weights stay in the viewer's model cache, outside the package.

Alternatively, download/extract a local file and set `model_path`. For example:

```sh
mkdir -p "$HOME/Models/inspector"
curl -fL 'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/151_object_detection_mobile_object_localizer/resources.tar.gz' -o "$HOME/Models/inspector/resources.tar.gz"
tar -xzf "$HOME/Models/inspector/resources.tar.gz" -C "$HOME/Models/inspector" saved_model/model_float32.onnx
shasum -a 256 "$HOME/Models/inspector/saved_model/model_float32.onnx"
```

The verified model's SHA-256 is
`4f6f761beb78e4c2aac0033e4792564cf4d2e00cd8c866878c74ee236b51c837`.
Archive/model source and license attribution are recorded in `provenance.json`.
The model is Apache-2.0; retain its applicable license/provenance when distributing
weights. No upstream Python files or weights are copied into this package.

## Load and configure

1. Insert a **Custom** node in a software stack, then open this folder.
2. Leave **Local model override** on **Automatic download** for automatic installation.
   Select an existing local model, or use **Browse…** to add your own ONNX file.
   The dropdown lists cached/bundled models and previously browsed files. Browse
   records the absolute path without copying weights; the file must remain accessible.
   Selected missing files are marked as missing rather than silently falling back.
   Models must match this example's four-output localizer format.
3. Place the node before **Output → viewer** or an expanded **Preview** node.
   Start with a clean preview/grayscale image before decorative palettes or heavy
   upscaling; the detector sees exactly the image at its position in the stack.
4. In **Inference device**, choose CPU, Apple Core ML or NVIDIA CUDA. GPU choices
   are enabled from the viewer's startup capabilities. On Apple Core ML, select
   **Compute devices** to include the GPU and/or Neural Engine.

CPU remains the default. Unavailable saved GPU preferences try the other available
GPU, then CPU, while preserving the saved selection. The callback receives the
effective device, and changing its backend or Apple compute units recreates the
model session. If GPU initialization fails for this model, select CPU; initialization
errors are reported rather than silently accepting an absent GPU provider.

| Setting | Meaning |
| --- | --- |
| `model_path` | Empty for automatic download; absolute or `~` path overrides it. |
| `model` | Optional downloader source override; defaults to embedded `MODEL_SOURCE`. |
| `device` | Execution object with `backend` (`cpu`, `coreml`, `cuda`) and `apple_compute`; default CPU with saved `CPUAndGPU` Apple units. |
| `providers` | Legacy ONNX Runtime provider list, used only when `device` is absent; defaults to `CPUExecutionProvider`. |
| `filter_scores` | Enable the inclusive confidence range; default true. |
| `score_threshold` | Minimum confidence, 0–1; default 0.2 matches Inspector's localizer. |
| `score_maximum` | Maximum confidence, 0–1; default 1. Minimum must not exceed maximum when filtering is enabled. |
| `max_detections` | Highest-scoring boxes to draw, 1–100; default 25. |
| `line_thickness` | Box width, 1–8 pixels; default 1. |
| `show_scores` | Optional confidence labels; default false (boxes only). |
| `nesting` | `Inside-out`: innermost boxes only. `Outside-in`: parents and children, drawn largest first (default). |
| `nested_coverage` | Fraction of a smaller box covered by its parent, 0.5–1; default 0.85. Set 1 for exact containment. |
| `suppress_duplicates` | Keep the strongest overlapping peer box; default true. |
| `duplicate_iou` | Peer overlap threshold (intersection/union), 0.01–1; default 0.5. |
| `border_color` | RGB `#RRGGBB` or `dynamic`; default green `#00ff00`. |
| `fill_color` | Independent RGB `#RRGGBB` or `dynamic`; default green. |
| `fill_opacity` | Fill blend, 0–1; default 0 leaves interiors transparent. |

The package uses the viewer's existing NumPy, OpenCV and ONNX Runtime dependencies.
It uses the shared model downloader and installs no dependencies. Unavailable configured
providers and missing/incompatible weights produce clear pipeline errors.
CPU and actual Apple Core ML inference were verified on an Apple M4 with macOS
26.6.2 and ONNX Runtime 1.30.0 using the checksum-pinned export. Core ML's compute
plan assigned convolution operations to the GPU with `CPUAndGPU`; ONNX Runtime
assigned 216 of 223 graph nodes to three Core ML partitions, with remaining
post-processing on CPU. On a visible-light bus/person image, both CPU and GPU
returned four proposals above 0.2 and the complete Custom node drew the same
boxes. Maximum score difference was below 0.000001. `CPUOnly`, `ALL` and
`CPUAndNeuralEngine` Core ML modes also returned valid outputs; this does not
establish that each mode actually used the Neural Engine. CUDA inference remains
unverified. Normal regression tests use fake sessions and do not run hardware.
Provider registration alone does not prove GPU placement; use Core ML's
[`ProfileComputePlan`](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#available-options-new-api)
and ONNX Runtime profiling when checking another model/platform.

The decoder accepts the count output as `(1,)` or `(1, 1)` and normalizes the
extra batch axis before reading it. Other output shapes and count values remain
strictly validated. The export declares `(1,)` while Core ML execution returns
`(1, 1)`, which triggers ONNX Runtime's `VerifyOutputSizes` warning before decoding.
This example sets its session's `log_severity_level` to `3` (errors and fatal
messages) to suppress the repeated warning. This also suppresses other ONNX
Runtime warnings from this session; errors, decoder validation and other nodes'
logging remain active. Native Core ML messages outside ONNX Runtime's logger may
still appear. For diagnostic logging, change the example's session severity back
to `2` and reload the folder.

`__init__.py` embeds `CONFIG_JSON` defaults/control definitions and `MODEL_SOURCE`,
and owns a per-node session cache. No separate configuration or model-source file
is needed. The editor shows confidence filtering/range, maximum detections, box thickness,
optional labels, nesting, border/fill colors, fill opacity, local-model and device controls,
and hides the configuration/model import
buttons. Advanced settings remain accessible through **Edit JSON configuration…**.
`settings.py` validates configuration,
`detector.py` resizes the full frame and maps normalized boxes back to its size,
`nesting.py` filters/orders boxes, `labels.py` exposes the shared label layout, and
`drawing.py` renders overlays.
Changing thresholds/overlay settings reuses
the session; changing model path, effective device, Apple compute units or a legacy
provider list reloads it. When replacing
weights, use a new filename and update `model_path`, or remove and re-add the node.
Drawn pixels are
quantized to uint8 for OpenCV text, then the Custom runtime returns float32.

Inside-out removes parents covering at least **Nested box coverage** of a smaller
box, including multi-level parents and children whose edges slightly cross the
parent's boundary. Parents must be more than 5% larger in area; nearly equal boxes
are treated as peers rather than false nesting caused by coordinate jitter.
Outside-in retains parent/child relationships and paints parents before children.
Duplicate suppression keeps the highest-confidence peer when their overlap reaches
**Duplicate overlap (IoU)**; distinct parent/child pairs remain visible in Outside-in.
Turn suppression off to retain all peers.

Confidence range filtering runs before nesting, duplicate suppression and the count
limit. **Filter confidence scores** can disable both confidence bounds. The count
limit retains the highest-scoring survivors. Lowering the minimum includes weaker
proposals and can increase clutter; the maximum can exclude stronger proposals
when needed. All scores use 0–1 fractions.

Border and fill each have a color dropdown and **Choose…** color picker. Dynamic
borders reuse the measurement overlay's pixel inversion and contrasting halo;
enabled labels stay white with black outlines in both fixed and dynamic border modes.
Labels reserve positions within the image and avoid one another where space permits.
In the viewer and preview thumbnails, text is drawn after resizing and rotation,
using the same font size as measurement points. Confidence glyphs stay upright
through all viewer rotations and mirror settings while their anchors follow the boxes.
**Show confidence scores** remains optional and defaults off; enable it to display
the labels. Dynamic fill blends inverted image
colors at the chosen opacity. Fills blend once across the union of overlapping
boxes, so nesting does not compound their opacity.
The node's processing session is reused when changing these display controls.

Saved/exported pipelines include the code and JSON but not the external weights.
Each receiving machine downloads/caches the pinned model or supplies a local path. Native sensor
measurements are untouched. This visible-image localizer has not been validated
for thermal object detection; boxes are model proposals, not object identities.

## Verification

Normal tests use synthetic frames and fake ONNX sessions; they need no weights,
network or hardware. An optional still-image experiment exercises actual inference:

```sh
.venv/bin/python experiments/experiment_classless_yolo.py \
  --model "$HOME/Models/inspector/saved_model/model_float32.onnx" \
  --image /absolute/path/to/image.png \
  --output diagnostics/classless-yolo.png
```

Resolve the configured interpreter through the repository's Python tooling before
running it. The experiment reads the supplied model/image and writes the requested
output without camera access or preference changes.

Existing saved Custom nodes embed an older copy of the package. Restart the viewer
for the device-control host support, then open this folder again to load the new
control and code. Folder reload restores package defaults; reapply your preferred
detection/display settings and save the pipeline afterward.
