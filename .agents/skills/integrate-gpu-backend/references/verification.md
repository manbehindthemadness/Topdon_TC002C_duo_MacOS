# GPU evidence and regression verification

Keep hardware/network experiments separate from deterministic application tests.
Use the configured SDK through `python-tools`, with mocked boundaries for pytest.

## Build an evidence matrix

Record one row per observed host/build combination:

- GPU/device and kernel architecture; OS and process architecture; Python ABI.
- Driver support, installed native runtime/cuDNN or equivalent vendor libraries,
  actual loaded library locations when relevant, and package build/source/hash.
- Official compatibility evidence and date; wheel/source availability.
- Import and advertised providers; initialized session providers/options.
- Inference output correctness; evidence of GPU operator execution or partial offload.
- Cold setup/first inference, warm inference and end-to-end pipeline performance.
- Outcome, fallback reason, cleanup/restoration and exact reproduction command.

An unresolved row is not a supported configuration. Label published compatibility,
package resolution and physically measured behavior separately. Keep diagnostics
free of credentials and do not upload local artifacts without a request to publish.

## Use representative models without claiming universal support

Start with bundled ACNet to avoid network/model variability. Inspect the existing
`experiments/experiment_nvidia_inference.py` and `experiment_onnx_visual.py` before
creating a new camera-free `experiments/experiment_<backend>_inference.py` helper.
Reuse model/data fixtures where they provide independent expected results.

Then cover models actually relevant to the request: different ACNet levels,
rectangular/dynamic inputs, grayscale/RGB, denoising/style/colorization or a Custom
ONNX example. Do not download the entire model catalog to prove one backend.
If a model needs external weights, document its source/checksum and setup separately.

Compare shape, dtype, finite values, range and CPU agreement with justified
tolerances. Precision changes, nondeterministic kernels and graph specialization
may require model-specific tolerances; do not generate expected output from the
new implementation and call it an independent reference.

Provider membership alone does not prove accelerated operator placement. Use
provider profiling, partition/engine reports or vendor execution evidence where
available. Inspect unexpected CPU fallback. Permit documented unsupported-operator
fallback while requiring GPU execution evidence for an acceleration claim.

Measure cold initialization/compilation separately from warm inference, then include
transfers, conversion, resizing, branches and Custom callbacks in pipeline latency.
Synchronize asynchronous GPU work when required by the timing method. Record input
sizes, compute/precision mode, warmup, sample count and latency variation. A faster
tiny kernel or high utilization alone is not proof of a faster thermal pipeline.

## Verify failure containment and startup integration

Cover the boundaries affected by the implementation rather than merely asserting
package strings:

- Missing/unsupported wheels, libraries or drivers; wrong ABI or missing kernels;
  native import/session crash, timeout and unavailable device.
- Native stdout/stderr and buffered teardown noise with a valid structured reply;
  malformed, missing or inconsistent result files must remain unavailable.
- Existing vendor build retained; staged replacement verified before activation;
  failed stage cleaned up without touching a working runtime.
- Application restart after `uv sync`; cached family/ABI isolation; corrupt state,
  interrupted downloads, concurrent setup, offline retry and read-only deployment.
- Capability probes and spawned workers select the same runtime; stale sessions
  close and recreate when effective provider/device/model settings change.
- Explicit CPU honored; saved GPU intent retained through fallback, close/reopen
  and document import. Unsupported choices explain why they are unavailable.
- Built-in and Custom provider paths agree; locked UI controls stay locked;
  browsing/import validation neither executes Custom code nor creates sessions.
- Camera construction/acquisition starts after setup; help/version/USB diagnostics
  perform no installations. Native temperature arrays and measurement results remain
  identical to CPU/display-only baselines, including transforms and branch combines.

Read `tests/test_runtime_setup.py`, `test_nvidia_acceleration.py`,
`test_apple_acceleration.py` and applicable processing/Custom device tests to reuse
fixtures and subprocess conventions. Keep all live driver/weight/download checks
outside the normal test collection. A fake provider test validates application
behavior; it does not verify the physical GPU or package's kernels.

## Primary-source research entry points

Use these as discovery links, then follow the current provider-specific requirements
and exact release/source metadata. Do not copy an old version matrix into policy.

- [ONNX Runtime providers](https://onnxruntime.ai/docs/execution-providers/)
- [ONNX Runtime installation](https://onnxruntime.ai/docs/install/)
- [CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [CoreML provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
- [Published GPU package metadata](https://pypi.org/pypi/onnxruntime-gpu/json)
- [Published standard package metadata](https://pypi.org/pypi/onnxruntime/json)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [Apple Metal documentation](https://developer.apple.com/documentation/metal)
- [Apple Core ML documentation](https://developer.apple.com/documentation/coreml)

For a new vendor/provider, use the corresponding official requirements and SDK
documentation. A community wheel or forum report is a lead; inspect its provenance,
build options and licensing, then verify the actual target before making it a default.
