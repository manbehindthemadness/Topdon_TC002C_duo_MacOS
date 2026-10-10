# NVIDIA JetPack GPU inference

On Linux, **AI enhancement → Execution → NVIDIA CUDA** runs ACNet and the
existing ONNX visual models on GPU zero. The ONNX style node also supports CUDA.
Anime4K09, ordinary image filters, colour conversion, resizing, and radiometric
measurements retain their existing CPU paths. TensorRT execution is not exposed.

The GPU backend uses the original model weights and float32 tensors. TF32 is
disabled. Unsupported operators can run on CPU within a CUDA session. Inference
runs in persistent spawned helpers, with one cache per processing branch, so a
native failure is contained. Removing, bypassing, or changing a node's backend
closes its disconnected helper. ACNet denoising levels, repeated upscale passes,
amount blending, and the 4-megapixel limit apply to CUDA execution too.

Startup runs a small bundled ACNet inference in a disposable process with a
20-second timeout. CUDA controls appear only after provider initialization and
inference succeed. A missing or broken runtime leaves the viewer usable on CPU.
Saved `cuda` and Apple `coreml` preferences remain unchanged on other systems;
an unavailable GPU backend tries the other available GPU before falling back to
CPU. Explicit CPU execution remains CPU. A failure after CUDA was selected is
reported in Camera status. Apple compute-device preferences apply when Core ML
is the effective backend,
including fallback from a saved CUDA preference. They do not configure NVIDIA
execution.

## Install on JetPack 7.2.1 / CUDA 13.2

Use the normal Ubuntu system-library and USB-permission setup in the README.
Install the project dependencies first, then replace the CPU ONNX Runtime
distribution with the tested ARM64 GPU wheel from Microsoft's CUDA 13 feed:

```bash
.venv/bin/uv sync --dev
.venv/bin/uv pip uninstall onnxruntime
.venv/bin/uv pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ 'onnxruntime-gpu==1.28.0.dev20260722004'
.venv/bin/python -m topdon_duo.nvidia_acceleration
.venv/bin/topdon-duo-desktop --rotate 90
```

The probe should report `"available": true` and `"cuda": true`. Open the pipeline
editor, select ACNet or an ONNX model, and choose **NVIDIA CUDA**. New nodes and
existing CPU presets keep CPU execution until you change their backend.

The GPU wheel replaces the `onnxruntime` import provided by the standard CPU
dependency; do not install both distributions in one environment. This optional
native runtime is installed separately from the cross-platform lockfile.
Use entry-point executables directly, or `uv run --no-sync`, on this environment.
An ordinary `uv sync` or `uv run` restores the locked CPU dependency; repeat the
replacement commands afterward. For example:

```bash
.venv/bin/uv run --no-sync topdon-duo-models espcn
.venv/bin/uv run --no-sync pytest
.venv/bin/uv run --no-sync ruff check .
```

Model downloads and conversions remain separate from hardware acquisition.
For other JetPack releases, select a GPU wheel built for that system's Python,
ARM64 architecture, CUDA, and cuDNN versions. PyPI GPU releases before 1.27 use
CUDA 12; releases from 1.27 use CUDA 13. Neither is the pinned nightly setup
verified above. For other Linux hosts, see [general CUDA setup](nvidia-linux.md).
See the official
[ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/) and
[CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Live GPU verification

This command requires a working NVIDIA driver and GPU runtime, but no camera,
network, root permissions, or downloaded models. It is outside normal pytest:

```bash
.venv/bin/python experiments/experiment_nvidia_inference.py
# Choose a different location for retained profiling artifacts if needed:
.venv/bin/python experiments/experiment_nvidia_inference.py --output /usr/src/codex/scratch/jetpack
```

The experiment checks all four bundled ACNet levels, profiles actual CUDA
operator execution, compares native-size float32 output against CPU inference,
and exercises the isolated worker with colour images, partial amount blending,
two passes, and unchanged input arrays. JSON profiles are retained under the
output directory. It does not establish performance for other models or the
complete live viewer.

Verified on an AGX Orin host with JetPack 7.2.1, L4T R39.2.1, CUDA 13.2,
cuDNN 9.20, and the pinned ONNX Runtime GPU wheel above. All four ACNet levels
executed CUDA operators; sampled native-size tensors matched CPU exactly.
Colour/amount outputs matched OpenCV's independent CPU implementation within
one byte value, and two-pass output was 1024×768. Sampled warm GPU inference
took roughly 4–7 ms per native-size ACNet pass, excluding transfers through the
worker, startup, and other pipeline stages.
