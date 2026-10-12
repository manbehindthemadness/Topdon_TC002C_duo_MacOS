# Automatic ONNX Runtime setup

The desktop and web viewers check their inference runtime after parsing arguments,
before creating a camera or starting acquisition. Help, version and web USB diagnostics
do not install packages. Setup also runs on later launches, so a dependency sync does
not permanently remove GPU support.

| Host capability | Runtime selection |
| --- | --- |
| macOS with a Metal device | Standard `onnxruntime>=1.23,<1.31`, with CoreML verification |
| Linux x86-64/ARM64, CUDA driver supporting 13+ | `onnxruntime-gpu>=1.27,<1.31`; try CUDA 12 if unavailable |
| Linux x86-64/ARM64, CUDA driver supporting 12 | `onnxruntime-gpu>=1.23,<1.27` |
| No supported accelerator | Standard ONNX Runtime with CPU execution |

Apple uses CoreML's `CPUAndGPU` execution mode, rather than a separate Metal wheel.
NVIDIA detection calls the CUDA driver in a disposable subprocess; neither JetPack
identifiers, `nvidia-smi`, nor an installed compiler toolkit are required. The driver
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
setup installs the CUDA/cuDNN dependencies declared in the wheel's metadata and
tries inference again. `uv` resolves native Python/architecture/glibc wheel tags.
Unsupported wheels or drivers fall back to the next supported option, ending with CPU.
Some ARM64 hosts still need vendor builds; automatic wheel selection cannot create
a build that upstream does not publish. The new provisioning path has deterministic
tests; its NVIDIA downloads and inference still require verification on physical hosts.

The project environment and `uv.lock` are untouched. Verified replacements live in
`~/Library/Caches/topdon-duo/runtimes` on macOS or
`$XDG_CACHE_HOME/topdon-duo/runtimes` (default `~/.cache/topdon-duo/runtimes`) on Linux,
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
.venv/bin/python -m topdon_duo.runtime_setup --retry
```

To explicitly disable startup checks/downloads (for offline or managed deployments):

```bash
BASED_THERMALS_RUNTIME_SETUP=0 .venv/bin/based-thermals-desktop
```

This bypass also skips cache activation; the environment's installed runtime is used.
Do not call setup after importing `onnxruntime` in an embedded application: native
runtime replacement requires a new process. For integrations outside the viewer,
call `topdon_duo.runtime_setup.ensure_runtime()` before importing ORT or starting workers.
The explicit setup command verifies the selection in its own process; it does not
alter a parent shell's environment. Cached runtimes are application packages and must
not be manually edited while viewers are running. Old successful generations can be
removed with the application closed; removing the runtime cache triggers setup again.

See the official [CoreML provider documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
and [CUDA compatibility requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
