# NVIDIA DGX Spark GPU inference

DGX Spark uses the existing Linux USB capture and NVIDIA CUDA inference backend.
Follow the Ubuntu system-library and udev setup in the README first. A graphical
desktop is required for `topdon-duo-desktop`; the browser viewer also works from
a terminal without a display.

## Install on CUDA 13.0

This setup was verified on DGX Spark with NVIDIA GB10, Ubuntu ARM64, Python 3.12,
driver 580.173.02, and the system CUDA 13.0 toolkit. Use the released ARM64 ONNX
Runtime GPU wheel and install cuDNN into the project environment:

```bash
uv sync --dev
uv pip uninstall --python .venv/bin/python onnxruntime
uv pip install --python .venv/bin/python --no-deps \
  'onnxruntime-gpu==1.30.0' 'nvidia-cudnn-cu13==9.20.0.48'
.venv/bin/python -m topdon_duo.nvidia_acceleration
.venv/bin/topdon-duo --diagnose
.venv/bin/topdon-duo-desktop --rotate 90
```

The `--no-deps` installation uses the Spark's existing system CUDA libraries
(including cuBLAS, cuFFT, cuRAND, and NVRTC). The ordinary project setup supplies
ONNX Runtime's Python dependencies. The viewer preloads NVIDIA libraries before
creating CUDA sessions, allowing it to find pip-installed cuDNN in each spawned
inference worker without changing system libraries or `LD_LIBRARY_PATH`.
The probe must report `"available": true` and `"cuda": true`.

Only one ONNX Runtime distribution should be installed: `onnxruntime-gpu`
replaces the `onnxruntime` import. As with the JetPack environment, ordinary
`uv sync` or `uv run` restores the locked CPU package. Use the `.venv/bin/`
executables directly or `uv run --no-sync` after installing the GPU runtime;
repeat the replacement commands after syncing dependencies.

Select **Camera → Pipeline → AI enhancement → Execution → NVIDIA CUDA** for
ACNet or an ONNX enhancement model. The style node also supports CUDA. CPU
nodes retain their selected backend. Saved Apple GPU preferences automatically
use NVIDIA when CUDA is available. Radiometric measurements and ordinary image
filters retain their existing CPU paths; TensorRT is not exposed.

ONNX Runtime's official [CUDA requirements and library preloading guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
documents the CUDA 13 and cuDNN 9 requirements for this release.

## Verification on this host

```bash
.venv/bin/python experiments/experiment_nvidia_inference.py \
  --output diagnostics/spark-gpu
.venv/bin/python experiments/experiment_capture_only.py \
  --seconds 10 --frame-pump --usb-queue-depth 32 \
  --output diagnostics/spark-capture.jsonl
```

Close camera viewers before the capture experiment. Its output file must not
already exist. GPU verification does not open the camera or download weights.

All four bundled ACNet levels executed CUDA operators on this GB10. Sampled
native-size float32 outputs matched CPU exactly; colour blending matched the
independent OpenCV reference within one byte value. The isolated worker also
passed two-pass 1024×768 output and unchanged-input checks. Sampled warm GPU
inference took approximately 0.9–1.9 ms per native-size pass, excluding worker
transfers, startup, and the rest of the pipeline.

USB verification received 249 complete frames over ten seconds, approximately
25 fps, with no timeouts and two startup packet/frame rejections. A separate
15-second desktop check with CUDA ACNet enhancement processed 335 frames,
reported no pipeline errors, and shut down cleanly. It used temporary viewer
preferences, leaving the user's saved pipeline untouched. These bounded checks
do not establish long-session stability or performance for other models.
