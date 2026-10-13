# Other Linux systems with NVIDIA CUDA

CUDA inference is attempted on Linux x86-64 and ARM64, without requiring a
Jetson or DGX model identifier. The GPU must support the installed runtime,
and CUDA controls are enabled only after a bundled ACNet inference succeeds
in a disposable process. Missing libraries, unsupported GPUs, driver errors,
and native startup failures leave the viewer usable on CPU.

DGX Spark and the documented JetPack host have live verification. Other
Linux systems are **experimental and unverified on physical hardware**.

Fresh `uv sync` installations provide CPU ONNX Runtime. Both viewers now perform
[automatic runtime setup](onnx-runtime-setup.md) before camera acquisition, selecting
CUDA from driver capabilities and caching verified packages outside the project venv.
Ordinary `uv run` and `uv sync` therefore work with automatic viewer setup. A missing
CUDA provider before setup does not reject a particular NVIDIA product. Detection
and setup do not require JetPack, Jetson or DGX identifiers.

The procedures below remain available for manual environment installation and vendor
builds. Their `uv run --no-sync` advice applies to those environment-installed packages;
automatic viewer setup uses a separate cache. Standalone probes/experiments do not
activate that cache unless their entry point calls `ensure_runtime()` first.

## Existing CUDA 13.0 / cuDNN 9 system libraries

For a Linux x86-64 or ARM64 host with existing CUDA 13.0 (or newer) and cuDNN 9.x,
use the general installer with system libraries:

```bash
.venv/bin/python -m based_thermals.cuda_setup --cuda 13 --system-libraries --dry-run
.venv/bin/python -m based_thermals.cuda_setup --cuda 13 --system-libraries
.venv/bin/based-thermals-desktop
```

This path selects the published CUDA 13 GPU runtime family (1.27–1.30). It validates
wheel tags for the active Python/architecture/glibc before removing the CPU runtime,
then verifies actual bundled inference in a separate process. A failed resolution
leaves the working runtime intact; a failed inference check prints its diagnostic reason.
System mode reuses installed CUDA/cuDNN libraries instead of adding pip NVIDIA packages.
This procedure also applies to JetPack 7.1/CUDA 13.0, without any JetPack-specific gate.

After installation, launch through `.venv/bin/` or `uv run --no-sync`. Ordinary
`uv sync` or `uv run` reinstalls the locked CPU package, requiring GPU setup again.
See the official [CUDA 13 compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-13x).

## Optional setup

Install the normal project dependencies and your distribution's NVIDIA driver
first. Confirm `nvidia-smi` works. Run setup inside the project environment:

```bash
uv sync --dev
# Choose one runtime family supported by your driver/GPU:
.venv/bin/python -m based_thermals.cuda_setup --cuda 12 --dry-run
.venv/bin/python -m based_thermals.cuda_setup --cuda 12
# Or, for CUDA 13:
.venv/bin/python -m based_thermals.cuda_setup --cuda 13
```

The dry run checks Python, architecture, and native wheel compatibility without
changing the environment. Installation resolves packages before replacing CPU
ONNX Runtime, installs pip CUDA/cuDNN libraries, and tests actual GPU inference.
It does not install or update the NVIDIA driver. A successful resolution alone
does not prove the driver or GPU can execute the model.

| Target                    | Package selection                                  | Verification status                                              |
| ------------------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| Linux x86-64, CUDA 12     | ONNX Runtime GPU `>=1.23,<1.27`, cuDNN 9           | Attempted; package resolution checked                            |
| Linux x86-64, CUDA 13     | ONNX Runtime GPU `>=1.27,<1.31`, cuDNN 9           | Attempted; package resolution checked                            |
| Linux ARM64/SBSA, CUDA 13 | Same CUDA 13 releases when compatible wheels exist | DGX Spark verified; other hosts unverified                       |
| Linux ARM64, CUDA 12      | Vendor/source GPU build may be required            | No matching PyPI ARM64 wheel found in the selected release range |
| Jetson / JetPack          | Use [JetPack instructions](nvidia-jetpack.md)      | Documented AGX Orin configuration verified                       |
| DGX Spark                 | Use [Spark instructions](nvidia-spark.md)          | GB10 configuration verified                                      |

The ranges deliberately separate PyPI's CUDA 12 builds from the CUDA 13 builds
introduced in ONNX Runtime 1.27. Follow the official
[CUDA and cuDNN compatibility requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
for minimum runtime/driver support and GPU architecture constraints. CUDA 11,
Windows, ROCm, and TensorRT are outside this setup path.

For an existing compatible system CUDA/cuDNN installation, add
`--system-libraries` to omit pip NVIDIA libraries. This is also useful for vendor
environments, but the helper still uses PyPI wheels. For vendor-specific builds,
follow the vendor's installation procedure and use the same verification commands
below. Older runtimes without the preloading API need their libraries visible
through the system loader or `LD_LIBRARY_PATH`.

If installation fails, the helper attempts to restore CPU ONNX Runtime. If GPU
verification fails after installation, the GPU package remains installed with
CPU execution available; setup exits with status 1 and explains how to inspect
the failure. Use `uv sync` to return to the locked CPU environment.

## Run and verify

```bash
.venv/bin/python -m based_thermals.nvidia_acceleration
.venv/bin/python experiments/experiment_nvidia_inference.py \
  --output diagnostics/linux-cuda
.venv/bin/based-thermals-desktop --rotate 90
```

The probe must report `"available": true`. The experiment verifies CUDA operator
execution, CPU agreement, and isolated workers for all four bundled ACNet levels;
it needs neither a camera nor downloaded weights. Other models may still have
unsupported operators or model-specific failures.

Choose **NVIDIA CUDA** in the enhancement or style node's Execution selector.
Explicit CPU selections remain CPU; unavailable saved GPU preferences fall back
to an available GPU or CPU. A model failure after GPU selection is reported in
Camera status. Radiometric measurements and ordinary filters remain on CPU.

Use `.venv/bin/` executables or `uv run --no-sync` afterward: ordinary `uv sync`
or `uv run` restores the locked CPU dependency. Repeat optional setup after a
dependency sync. Desktop operation also needs your distribution's GUI libraries;
USB capture needs libusb and camera permissions. The
[setup guide's Ubuntu apt and udev commands](setup.md#ubuntu-2404-and-later) are
examples, not package-manager instructions for every distro.
