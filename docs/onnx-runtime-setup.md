# Automatic ONNX Runtime setup

The desktop and web viewers check their inference runtime after parsing arguments,
before creating a camera or starting acquisition. Help, version and web USB diagnostics
do not install packages. Setup also runs on later launches, so a dependency sync does
not permanently remove GPU support.

| Host capability | Runtime selection |
| --- | --- |
| macOS with a Metal device | Standard `onnxruntime>=1.23,<1.31`, with CoreML verification |
| Linux ARM64 Tegra234 (Jetson Orin), CUDA driver supporting 13+ | Verified `onnxruntime-gpu==1.28.0.dev20260722004` from Microsoft's CUDA 13 feed, using system libraries; generic CUDA choices follow if unavailable |
| Linux x86-64/ARM64, CUDA driver supporting 13+ | `onnxruntime-gpu>=1.27,<1.31`; try CUDA 12 if unavailable |
| Linux x86-64/ARM64, CUDA driver supporting 12 | `onnxruntime-gpu>=1.23,<1.27` |
| No supported accelerator | Standard ONNX Runtime with CPU execution |

Apple uses CoreML's `CPUAndGPU` execution mode, rather than a separate Metal wheel.
NVIDIA detection calls the CUDA driver in a disposable subprocess; neither `nvidia-smi`
nor an installed compiler toolkit is required. An exact `nvidia,tegra234` entry in
`/proc/device-tree/compatible` selects the Orin-specific CUDA 13 build on Linux ARM64.
Other hosts, including DGX Spark, retain the generic selection. The driver
version reports compatibility, not the version of installed CUDA libraries. Actual
session initialization and bundled ACNet inference must pass before a runtime is
selected. Probe responses use a separate temporary JSON file, keeping native stdout
and process-exit diagnostics from corrupting the result. CoreML may delegate
unsupported operators to CPU; this check does not prove every model operation runs
on GPU. CUDA 11, Windows GPU, ROCm, OpenVINO and
TensorRT are outside the application's current backend support.

A working environment runtime, including a compatible vendor build, is retained.
Otherwise, setup stages a compatible wheel in the application's separate cache and
tests it before activation. CUDA first tries existing system libraries; if that fails,
generic setup installs the CUDA/cuDNN dependencies declared in the wheel's metadata and
tries inference again. `uv` resolves native Python/architecture/glibc wheel tags.
The pinned Orin build uses JetPack's system libraries only. Matching wheel tags alone
do not guarantee compatible GPU kernels: the generic 1.30.0 ARM64 wheel failed on
the verified Orin with `cudaErrorNoKernelImageForDevice`. The pinned build has its own
cache family and must pass the same bundled inference check before activation.
Unsupported wheels or drivers fall back to the next supported option, ending with CPU.
Some ARM64 hosts still need vendor builds; automatic wheel selection cannot create
a build that upstream does not publish. The provisioning path has deterministic
tests and the Orin path was verified on AGX Orin, L4T R39.2.1, CUDA 13.2, cuDNN 9.20,
and Python 3.12. Other JetPack configurations still require physical verification.

The project environment and `uv.lock` are untouched. Verified replacements live in
`~/Library/Caches/based-thermals/runtimes` on macOS or
`$XDG_CACHE_HOME/based-thermals/runtimes` (default `~/.cache/based-thermals/runtimes`) on Linux,
separated by Python ABI, OS, architecture, CUDA family and setup policy version.
`BASED_THERMALS_RUNTIME_DIR` overrides this directory. Cache state is published only
after verification; incomplete downloads are removed, including abandoned staging
directories on the next setup attempt after an interrupted process. Capability
subprocesses and spawned processing workers inherit the same selected package path. `uv sync` can
restore its CPU package without replacing this cache, so ordinary `uv run` is safe
for viewer startup. Model downloads/cache clearing remain separate from native runtimes.

Initial setup needs network access and free disk space, potentially several GB for
NVIDIA libraries. It logs progress before the viewer window opens. No driver, toolkit,
system package, USB permission or administrator setting is changed. `uv` must be on
PATH or beside the running interpreter (as in the Ubuntu bootstrap). A per-runtime
process lock prevents duplicate installation. Another launch uses its existing runtime
while setup is in progress. Installer attempts have a ten-minute timeout; native checks
have a 45-second timeout. Failed setup is retried on a later launch after a five-minute
cooldown; existing installed CPU inference remains available when optional setup fails.

To run setup without opening a camera or retry after an offline/failed attempt:

```bash
.venv/bin/python -m based_thermals.runtime_setup --retry
```

To explicitly disable startup checks/downloads (for offline or managed deployments):

```bash
BASED_THERMALS_RUNTIME_SETUP=0 .venv/bin/based-thermals-desktop
```

This bypass also skips cache activation; the environment's installed runtime is used.
Do not call setup after importing `onnxruntime` in an embedded application: native
runtime replacement requires a new process. For integrations outside the viewer,
call `based_thermals.runtime_setup.ensure_runtime()` before importing ORT or starting workers.
The explicit setup command verifies the selection in its own process; it does not
alter a parent shell's environment. Cached runtimes are application packages and must
not be manually edited while viewers are running. Old successful generations can be
removed with the application closed; removing the runtime cache triggers setup again.

See the official [CoreML provider documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
and [CUDA compatibility requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
