# Head Pose experiment

An optional Custom package for whole-head detection → square crop → HFHPE pose
inference → three-axis overlay. Select **`nodes/Head Pose`**, not this directory
or `upstream`. The upstream Git repository is preserved as a pinned submodule;
the portable adapter is separate and does not import its camera/demo/training code.

## Setup and use

In the viewer, add **Custom**, choose **Open module folder…**, and select
`examples/custom_nodes/head-pose/nodes/Head Pose`. **Head detector** defaults to
**HGNet-N (fast)**; **DINOv3-S (reference)** is also available. Leave both ONNX
overrides empty to download the pinned detector and pose models automatically.
The image passes through during downloads, with the viewer's shared model progress
and cache. No preparation command is needed for these exact audited releases.
If updating an existing node, reopen this folder and save the pipeline again to
replace its embedded code and add the new JSON-defined dropdown.
The Detector override lists HGNet-N/DINOv3-S head detectors by readable name;
the Pose override lists HFHPE ViT-T 64×64 rollgate. Their JSON discovery hints
hide unrelated cached models, including ACNet image enhancers. Known downloads
that arrive later receive the same names on the next dropdown opening. Browse
remains available for other compatible exports; unlisted saved selections stay
visible as explicit local overrides, and the worker validates them as usual.
Local overrides take precedence over the detector dropdown. First-use automatic
downloads and subsequent inference were verified in an initially empty diagnostics
cache; both models became ready in about 5.6 seconds on this connection. Download
time depends on network speed, and Apple compilation happens separately afterward.

For optional local weights and reproducible still-image checks, run from the viewer
repository root using its configured Python environment:

```sh
git submodule update --init -- examples/custom_nodes/head-pose/upstream
uv run --python .venv/bin/python --script tools/prepare_head_pose.py
.venv/bin/python experiments/experiment_head_pose.py \
  --images /path/to/head-image.png --backends cpu coreml --pose-backend cpu
```

Preparation uses isolated inline ONNX/certificate dependencies; it does not install
them into the viewer or change `uv.lock`. It downloads checksum-pinned release
assets into ignored `diagnostics/head-pose/models`, extracts only the exact regular
pose member, and extracts the detector graph reachable from `label_xyxy_score`.
It adds explicit normalization/preparation metadata and writes `provenance.json`.
The pinned current detector is already reduced to its detection dependencies;
the preparation step verifies that contract rather than assuming every upstream
"boxes-only" export is safe. `--offline` requires existing verified downloads.
Use `--detector dinov3-s --output diagnostics/head-pose/dinov3-models` for the larger
reference detector; `hgnet-n` is the preparation default. Do not mix model files
from different preparation runs without checking provenance.

Browse to a prepared `detector.onnx` and `pose.onnx` to override automatic models.
Alternatively, import a generated
`diagnostics/head-pose/results/*.pipeline.json` preset. Generated presets embed
the adapter and absolute local weight paths, have no hardware nodes, and use a
raw source; adjust the source and any desired display processing for your experiment.
They are diagnostic starting points, not replacements for tuned hardware presets.

Empty overrides use the shared `ModelResolver` to download the selected pinned
detector and ViT-T 64 archive in the background. Both requests start before
waiting for readiness; the normal viewer model progress/error display applies. Verified
weights are reused offline. Failure does not trigger downloads every frame;
reload or bypass/re-enable to retry. The two exact detector release checksums
were audited to contain only detection dependencies; other detector overrides
must carry metadata from the preparation tool. Graph preparation remains outside
the image callback.

Defaults request Apple Core ML `CPUAndGPU` for detection and CPU for pose,
with the viewer's usual capability fallback. Each session is cached independently
and recreated when its model file or effective device changes. Apple compilation
can take seconds on first use. CUDA is exposed through the existing device helper
but was not tested for this experiment. Neither CPU nor GPU runs need Torch.

Controls include a detector choice, two model override pickers, separate device pickers,
confidence threshold,
maximum heads (highest confidence first, default one), input rotation, and boxes/axes/angle
switches. No frame skipping, independent background inference or stale box reuse
is used. Detector class 7 boxes are expanded 5% per side, squared, shifted inside
the image borders and resized to the pose model input. Small images are zero-padded
as in upstream; RGB crops use `x/127.5 - 1` normalization.

Input rotation applies only to inference/drawing, then is undone before returning
the original image geometry. If your viewer rotates the camera image 90° clockwise
after processing, try model input rotation **270° CCW** to match that upright view.
Yaw/pitch/roll text describes the upright model input, and is deferred through the
viewer's final display transforms. Axes are display pixels and may be affected by
later image filters. Keep this node near the end of the display branch. Input is
BGR display data only; native temperatures and radiometry are untouched.

Supported overrides: prepared or exact checksum-matched audited release,
single-input static batch-1 DEIMv2 normalized
`label_xyxy_score [1,Q,6]` detectors; integrated center05 HFHPE 64/128 models with
`cos_sin [1,6]` and `kappa [1,3]`. Dynamic-batch, standalone yaw/pitch, ImageNet pose
and arbitrary YOLO exports are explicitly rejected. The preparation command currently
pins only the two named detector releases; it is not a generic model converter.

## Measured behavior

October 14, 2026, this Apple Silicon Mac, ONNX Runtime 1.30.0, four CPU threads,
largest head only in the original experiment, ten warm runs after two startup calls.
The current node ranks heads by detector confidence before applying the limit.
An upstream 672×720
validation preview sheet exercised detection, crop, pose and drawing through the
real Custom worker. It is a functional sample, not an independent accuracy benchmark.

| Detector / pose | Warm whole-node median | Detector median | Pose median |
| --- | ---: | ---: | ---: |
| HGNet-N CPU / ViT-T 64 CPU | 69.0 ms | 58.7 ms | 2.4 ms |
| HGNet-N Core ML / ViT-T 64 CPU | 49.0 ms | 35.7 ms | 3.5 ms |

CPU and Core ML detection yielded matching boxes and pose estimates on that sample.
HGNet-N Core ML initialization plus its first frame took about 8.1 seconds.
The larger DINOv3-S detector was substantially slower; Core ML pose inference was
also slower than CPU here (29 ms versus 3.4 ms in the earlier five-run comparison).
These timings include worker copies/validation but exclude camera capture, other
pipeline nodes, viewer rendering and USB waits. Registered providers do not mean
every graph operation executes on the GPU.

The saved 576×768 thermal capture in this checkout shows a torso without a visible
head. Neither detector found a head at confidence 0.7. HGNet-N Core ML processed it
in 39.0 ms median; this is not evidence of thermal head-pose accuracy. Use a new
thermal frame containing a head, and compare palette/grayscale/GAN branches
separately. Upstream models are trained on photographic/synthetic head appearances;
thermal accuracy and generated-color reliability remain unverified.

Reports, annotated PNGs and local-weight presets are written only under the specified
experiment output directory. No camera access or user-preference writes occur.

## Upstream updates and provenance

Adapter reference: upstream commit `526439eafbaadfa2e8bc5d85020a3e1e72149788`.
Upstream: <https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation>.
Weights: its `weights` release, ViT-T 64 v10 rollgate and HGNet-N/DINOv3-S detectors.
Exact release SHA-256 pins are in `tools/prepare_head_pose.py` and the node's literal
`MODEL_SOURCE`. The prepared detector digest is recorded locally after conversion.

To review an update without advancing the working pin immediately:

```sh
git -C examples/custom_nodes/head-pose/upstream fetch origin
git -C examples/custom_nodes/head-pose/upstream diff HEAD..origin/main -- README.md demo/demo_hfhpe_onnx.py
# After reviewing crop/normalization/output/provider changes:
git -C examples/custom_nodes/head-pose/upstream checkout --detach origin/main
```

Review new weights separately, verify new digests and rerun preparation, focused
tests and the still-image experiment before recording the new submodule pin. The
adapter is deliberately a maintained snapshot of the processing contract: pulling
upstream cannot silently execute new demo code inside saved pipelines. Reconcile
relevant upstream changes into the adapter, reopen **nodes/Head Pose**, and save
the pipeline again. Existing embedded packages and weight pins do not auto-update.

Upstream code is MIT, copyright 2026 Katsuya Hyodo; its full notice is embedded in
the portable package and remains in `upstream/LICENSE`. Detector/backbone sources
include DEIMv2 (Apache-2.0) and DINOv3 (Meta's DINOv3 license). HFHPE's upstream
README explains derived-weight terms; code licensing alone does not relicense
weights. No ONNX binaries or training data are checked into this viewer repository.
