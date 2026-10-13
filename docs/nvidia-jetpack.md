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

## Automatic setup on CUDA 13 Orin

Normal viewer startup now detects Tegra234 on Linux ARM64 and first stages
`onnxruntime-gpu==1.28.0.dev20260722004` from Microsoft's CUDA 13 feed in the
separate runtime cache. It uses existing JetPack libraries, verifies bundled CUDA
inference before activation, and leaves the project environment and lockfile intact.
The generic ARM64 1.30.0 wheel resolves successfully here but fails inference with
`cudaErrorNoKernelImageForDevice`; installing additional libraries does not fix
missing GPU kernels. Other NVIDIA hosts retain their existing automatic selection.

Run setup directly without opening a camera:

```bash
.venv/bin/python -m based_thermals.runtime_setup --retry
```

This automatic path was verified on the AGX Orin configuration listed below.
See [automatic runtime setup](onnx-runtime-setup.md) for cache and fallback behavior.
The manual instructions below remain available for managed environments.

## Manual setup on JetPack 7.1 / CUDA 13.0

This is a generic compatibility trial, not the preferred Orin setup. On Orin use
the automatic pinned-build path above; a generic wheel can lack its GPU kernels.
Normal viewer startup uses the separate cache and does not require these manual
environment replacement commands.

The normal `uv sync --dev` install provides CPU ONNX Runtime. NVIDIA detection
correctly reports unavailable until a GPU runtime is installed in that same environment.
CUDA 13.0 matches the published CUDA 13 ONNX Runtime family (1.27–1.30), requiring
cuDNN 9.x. Use JetPack's existing CUDA/cuDNN libraries rather than replacing them
with pip NVIDIA library packages:

```bash
uv sync --dev
.venv/bin/python -m based_thermals.cuda_setup --cuda 13 --system-libraries --dry-run
.venv/bin/python -m based_thermals.cuda_setup --cuda 13 --system-libraries
.venv/bin/based-thermals-desktop --rotate 90
```

The helper first resolves a wheel against the active Python, ARM64 and glibc tags.
If resolution fails, it leaves the CPU runtime intact. It then replaces both ORT
distributions with one GPU build and checks actual bundled ACNet inference in an
isolated process. Setup success requires that inference to pass; package availability
alone does not establish GPU support. Failure prints the probe reason. JetPack 7.1
has not been physically verified by this repository; retain the probe output when
checking this setup. No camera or root access is needed for setup/verification.

After GPU installation, use the entry-point executable above or `uv run --no-sync`.
Ordinary `uv sync` or `uv run` reinstalls the locked CPU build; rerun the GPU setup
after syncing dependencies. Saved CUDA preferences and downloaded models remain intact.
See the official [ONNX Runtime CUDA compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-13x).

## Previously verified JetPack 7.2.1 / CUDA 13.2

Use [Ubuntu system-library and USB-permission setup](setup.md#ubuntu-2404-and-later).
Install the project dependencies first, then replace the CPU ONNX Runtime
distribution with the tested ARM64 GPU wheel from Microsoft's CUDA 13 feed:

```bash
.venv/bin/uv sync --dev
.venv/bin/uv pip uninstall onnxruntime
.venv/bin/uv pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ 'onnxruntime-gpu==1.28.0.dev20260722004'
.venv/bin/python -m based_thermals.nvidia_acceleration
.venv/bin/based-thermals-desktop --rotate 90
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
.venv/bin/uv run --no-sync based-thermals-models espcn
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
