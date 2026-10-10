# Other Linux systems with NVIDIA CUDA

CUDA inference is attempted on Linux x86-64 and ARM64, without requiring a
Jetson or DGX model identifier. The GPU must support the installed runtime,
and CUDA controls are enabled only after a bundled ACNet inference succeeds
in a disposable process. Missing libraries, unsupported GPUs, driver errors,
and native startup failures leave the viewer usable on CPU.

DGX Spark and the documented JetPack host have live verification. Other
Linux systems are **experimental and unverified on physical hardware**.

## Optional setup

Install the normal project dependencies and your distribution's NVIDIA driver
first. Confirm `nvidia-smi` works. Run setup inside the project environment:

```bash
uv sync --dev
# Choose one runtime family supported by your driver/GPU:
.venv/bin/python -m topdon_duo.cuda_setup --cuda 12 --dry-run
.venv/bin/python -m topdon_duo.cuda_setup --cuda 12
# Or, for CUDA 13:
.venv/bin/python -m topdon_duo.cuda_setup --cuda 13
```

The dry run checks Python, architecture, and native wheel compatibility without
changing the environment. Installation resolves packages before replacing CPU
ONNX Runtime, installs pip CUDA/cuDNN libraries, and tests actual GPU inference.
It does not install or update the NVIDIA driver. A successful resolution alone
does not prove the driver or GPU can execute the model.

| Target | Package selection | Verification status |
| --- | --- | --- |
| Linux x86-64, CUDA 12 | ONNX Runtime GPU `>=1.23,<1.27`, cuDNN 9 | Attempted; package resolution checked |
| Linux x86-64, CUDA 13 | ONNX Runtime GPU `>=1.27,<1.31`, cuDNN 9 | Attempted; package resolution checked |
| Linux ARM64/SBSA, CUDA 13 | Same CUDA 13 releases when compatible wheels exist | DGX Spark verified; other hosts unverified |
| Linux ARM64, CUDA 12 | Vendor/source GPU build may be required | No matching PyPI ARM64 wheel found in the selected release range |
| Jetson / JetPack | Use [JetPack instructions](nvidia-jetpack.md) | Documented AGX Orin configuration verified |
| DGX Spark | Use [Spark instructions](nvidia-spark.md) | GB10 configuration verified |

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
.venv/bin/python -m topdon_duo.nvidia_acceleration
.venv/bin/python experiments/experiment_nvidia_inference.py \
  --output diagnostics/linux-cuda
.venv/bin/topdon-duo-desktop --rotate 90
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
USB capture needs libusb and camera permissions. The README's Ubuntu apt and
udev commands are examples, not package-manager instructions for every distro.
