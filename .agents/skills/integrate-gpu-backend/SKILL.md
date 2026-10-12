---
name: integrate-gpu-backend
description: Discover GPU and inference-runtime compatibility, then safely add or update acceleration backends in based-thermals. Use for new GPU providers, CUDA/cuDNN or ONNX Runtime versions, Apple GPU generations and CoreML/Metal changes; ordinary image filters and USB camera reverse engineering have separate workflows.
---

# Integrate a GPU backend

Establish which runtime can execute this project's models on the target host,
then implement the requested support while preserving working backends, saved
preferences and CPU fallback. A compatibility investigation can conclude that
an upstream build is unavailable; do not manufacture support through a UI flag.

## Establish scope and evidence

- Follow repository instructions and inspect current code and uncommitted changes.
  This skill belongs to the project; keep its resources in `.agents/skills/`.
  Research-only requests remain read-only. Implementation requests authorize
  relevant application changes, not unrelated driver/OS changes or publication.
- Identify whether this is an existing runtime-family update, a device-specific
  compatibility exception, or a genuinely new inference backend. A new Apple
  chip, such as M5, is a research target rather than proof that a new Metal wheel
  or backend identifier is required. Check current CoreML support first.
- Record OS/version, process architecture and Python ABI, GPU identity and kernel
  architecture, driver, installed native libraries, current runtime package/source,
  available providers and the exact error or desired behavior. Distinguish host
  facts from assumptions and remote reports. Read device state without opening USB.
- For CUDA, distinguish the driver's supported CUDA version, installed runtime,
  compiler toolkit, cuDNN ABI and GPU compute capability. Do not require JetPack
  or a vendor product name for generic CUDA eligibility. A compatible ARM64 wheel
  tag does not prove that its GPU kernels support that ARM64 device.
- Use the `python-tools` skill and the IDE's configured environment before Python
  toolchain commands. Ask for target-host facts only when needed; continue source
  research and deterministic implementation work without live hardware.

## Research the actual build

Consult current primary sources rather than relying on remembered compatibility
tables. Start with the [ONNX Runtime provider catalog](https://onnxruntime.ai/docs/execution-providers/),
the target provider's documentation, release notes, published wheel metadata and
the hardware vendor's driver/runtime support documentation. Use exact release or
build identifiers and record source URLs and investigation dates.

For each candidate, establish Python/OS/architecture wheel support, native-library
ABIs, GPU kernel coverage, model/operator/opset support and provider options.
Different execution providers may require different distributions or source builds;
do not assume `onnxruntime-gpu` supplies every GPU backend. TensorRT-like providers
can also introduce engine-build time and device-specific cache requirements.

Prefer a released compatible build. When a vendor/nightly/source build is needed,
document the reason, exact provenance and reproduction method; scope its selection
to the demonstrated compatibility case. Do not widen a version bound, choose an
unpinned nightly, or erase an existing verified exception just to make resolution pass.
Retain source/license notices for any artifacts the task actually redistributes.

Produce a small candidate matrix with observed host/build requirements, evidence,
known limitations and the next discriminating test. Keep package resolution,
successful inference, actual GPU execution and useful speedup as separate claims.

## Prove compatibility before enabling support

Read [verification.md](references/verification.md) before native experiments.

- Use a disposable, time-bounded process and an isolated staged runtime. Native
  import/session crashes, missing kernels and bad library loading must leave the
  current environment and camera acquisition recoverable.
- Keep structured replies outside stdout/stderr. Native libraries can emit or
  flush diagnostics during teardown; the project's temporary JSON response file
  exists because valid CoreML inference was previously rejected as `Extra data`.
- Verify model output shape, finite values and CPU agreement before benchmarking.
  An advertised provider or successful wheel install is insufficient. Confirm
  session/provider initialization and investigate operator placement or another
  provider-specific execution signal before claiming GPU acceleration.
- Cover representative built-in and Custom models, worker startup and fallback.
  Partial CPU operator fallback may be legitimate; report it instead of describing
  the entire model as GPU-executed. Never label untested hardware as verified.
- After a crash, timeout, kernel/ABI mismatch or unexpected fallback, retain the
  evidence and inspect the cause before another bounded experiment. Do not cycle
  speculative package pins, global library paths or system reinstalls until one works.

## Integrate at the existing boundaries

Read [repository-integration.md](references/repository-integration.md) when changing
runtime setup, provider selection, workers, saved documents or execution controls.

Use the smallest change justified by the evidence. A runtime-version change usually
belongs in startup selection and verification, not in a new pipeline node. A new
backend requires a stable identifier and consistent support across validators,
capabilities, processing, Custom device helpers, UI choices and persistence.

Preserve these application invariants:

- Stage and verify replacements outside uv-managed shared package files; retain
  compatible vendor builds. Setup occurs before native imports, camera construction
  and worker startup, never in acquisition, per-frame processing or document browsing.
- Reuse startup capability state in the UI and pass effective devices to workers.
  No session creation, probing or downloading in a selector. Preserve requested
  GPU preferences during fallback and honor explicit CPU selection.
- Keep provider/version/library/device/model settings in session/cache identity as
  appropriate. Close branch-owned sessions and rebuild them on effective changes.
  Every child process must inherit the same selected runtime and library behavior.
- Keep native thermal arrays, measurement sampling, USB assembly and platform
  permissions independent of display acceleration. GPU failures must not reset
  camera controls, alter calibration or silently modify measurement resolution.

Do not turn an application runtime experiment into a system driver, toolkit,
firmware or security-setting change without authorization for that action. Use the
existing user authorization for application work; do not add approval gates to
ordinary reversible edits or safe read-only discovery.

## Complete with an honest handoff

Run deterministic regression tests and required repository checks; keep live GPU,
network downloads and model conversion out of normal pytest collection. Record
hardware experiments under `experiments/` and local outputs under `diagnostics/`.
Maintain relevant runtime documentation and `summary.txt` with the exact tested
matrix, provenance, failure reasons and cache/recovery behavior.

Report what changed, what still falls back, where GPU execution and performance
were measured, and which configurations remain experimental. If target hardware
is unavailable, deliver testable integration with that limitation stated and a
precise camera-free verification command rather than claiming end-to-end success.
