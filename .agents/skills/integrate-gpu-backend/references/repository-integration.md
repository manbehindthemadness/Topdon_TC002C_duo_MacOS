# Repository integration map

Paths below are relative to the project root. Inspect the current implementations
before editing; this map is an entry point, not a replacement for source discovery.
The Python namespace remains `topdon_duo` despite the based-thermals project name.

## Runtime provisioning and capabilities

| Boundary             | Entry points                                      | What must stay consistent                                                                               |
| -------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Startup setup        | `src/topdon_duo/runtime_setup/__init__.py`        | Host candidates, fallback order, isolated cache families, activation and retry/locking policy           |
| Package selection    | `runtime_setup/install.py`                        | Version/build/index provenance, wheel tags, declared dependency requirements and system-library choices |
| Native probes        | `runtime_setup/probes.py`                         | Driver/runtime distinction, disposable process, strict result schema, result file and real inference    |
| Viewer startup       | `desktop_app/session.py`, `web.py`                | Setup after argument parsing, before camera construction; help/version/USB diagnostics remain read-only |
| Capability reporting | `apple_acceleration.py`, `nvidia_acceleration.py` | Native isolation, unavailable reasons and cached startup results                                        |
| Effective selection  | `acceleration.py`                                 | Requested preference retained, explicit CPU honored, deterministic fallback                             |

Paths in shorter table entries share the `src/topdon_duo/` prefix. Also read
[the runtime setup guide](../../../../docs/onnx-runtime-setup.md) and applicable
NVIDIA notes in `docs/nvidia-linux.md`, `docs/nvidia-jetpack.md` and
`docs/nvidia-spark.md`. Their verified scope is specific to the recorded host/build.

Current startup distinguishes generic CUDA families from an exact Tegra234/Orin
exception. A generic ARM64 wheel previously passed packaging checks but lacked
usable Orin kernels. Keep the proven variant until its replacement passes an
equivalent hardware experiment; do not use the exception as a generic JetPack gate.

Keep successful runtime generations outside the project environment so `uv sync`
cannot replace them. Preserve state validation, nonblocking installation locks,
atomic publication, failed/interrupted staging cleanup, cooldown and bounded probes.
If selection semantics or ABI compatibility change, update cache identity/policy
so old verified state cannot be applied to a different runtime family. Cache
directory names are not sufficient evidence: revalidate the selected runtime.

The parent, capability subprocesses and multiprocessing workers must resolve the
same package. Preserve `BASED_THERMALS_ORT_PATH`/`PYTHONPATH` propagation and clear
stale capability caches when activating a replacement. Once a native runtime is
imported, change it in a new process rather than attempting an in-process reload.
Respect runtime-cache overrides and the explicit setup-disable path.

Use compatible system libraries before downloading another native library stack
where the current setup contract supports that choice. Evaluate dependency markers
and the staged build's metadata rather than inferring NVIDIA package names from
CUDA marketing versions. A source/vendor build without installable extras may need
its own documented system-library path. Do not globally rewrite loader variables
or install two overlapping ORT distributions into the project's environment.

## Provider execution and session lifetime

Inspect `onnx_upsampling.py`, `coreml_upsampling.py`, `coreml_model.py`,
`processing/branch.py`, `processing/scheduler.py` and applicable enhancement modules.
Search usages of existing backend identifiers to find model-specific paths.

For a provider change, verify construction/options, disabled automatic session
fallback where currently required, model layout/dtype, static/dynamic shapes,
specialization, graph transformations and output-size validation. Apple provider
padding workarounds currently transform an in-memory model copy; retain original
bundled weights and CPU behavior. A new chip does not justify rewriting models.

Extend cache keys only with values that affect executable/session compatibility.
Include effective device, compute mode, model and shape specialization where
needed; engine-building providers may also depend on hardware and runtime versions.
Branch processors own model caches; the scheduler owns executors. Verify cleanup,
timeouts and failure reporting before adding provider concurrency or shared sessions.

Keep GPU processing display-only. Native radiometry and region measurements must
remain independent of processed pixels and transforms. Retain the 4-megapixel
software output budget and existing coverage/label/branch geometry semantics.

## Saved documents, controls and Custom packages

Inspect `pipeline_model/catalog.py` and validation/migration code, `pipeline_ui/compute.py`,
`pipeline_ui/custom_devices.py`, `custom_nodes/devices.py`, `custom_nodes/runtime.py`,
and viewer/pipeline state propagation. Follow actual call sites before changing an enum.

A new backend identifier must be accepted and routed consistently across built-in
nodes and optional Custom `device` controls. Preserve older saved backend and
`apple_compute` objects. Decide and test unsupported-import behavior rather than
silently replacing the user's requested value. Extend schemas/migrations only when
needed, keeping legacy plain JSON Custom configurations uninterpreted.

UI controls consume capability state; they do not import package code or initialize
GPU sessions. Respect pipeline locks and explain unavailable/experimental choices.
Worker callbacks receive effective devices through existing helpers. Custom packages
using `onnx_providers` should gain supported provider selection through that shared
contract; preserve packages with their own legacy model/session handling.

Prefer local adaptations over a registry/framework refactor. If adding a backend
exposes a genuinely inadequate abstraction, make the shared contract explicit and
test existing CPU/CoreML/CUDA behavior through it before introducing new choices.
