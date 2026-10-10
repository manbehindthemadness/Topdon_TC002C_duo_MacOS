---
name: create-custom-node
description: Create or adapt a folder-based Python and JSON package for this thermal viewer's Custom pipeline node, including configuration, portable packaging, examples, and verification. Use for user-designed image processing nodes; camera protocols and built-in node changes have separate workflows.
---

# Custom pipeline node builder

Build a package that users load through **Custom → Open module folder…**. Keep
examples independent of built-in nodes, default pipelines and user presets unless
integration is explicitly requested. Respect the user's node name and target
location; for bundled examples use `examples/custom_nodes/<display name>/`.
The selected folder's basename becomes the Custom node's display label. Folder
names may contain spaces; runtime module names are generated independently.

## Inspect the contract

Resolve the repository root from `AGENTS.md` and read its instructions. Consult
[`docs/custom-nodes.md`](../../../docs/custom-nodes.md) and the current
`src/topdon_duo/custom_nodes/bundle.py` and `runtime.py`; their implemented
contract takes precedence over stale examples. Before Python toolchain commands,
resolve the module interpreter through the available `python-tools` skill/IDE
`get_python_environment` workflow required by this repository.

Use [`local_contrast`](../../../examples/custom_nodes/local_contrast) as the
minimal starter. Use [`Classless YOLO`](../../../examples/custom_nodes/Classless%20YOLO)
when adapting an external model, session cache or detection overlay. Read only
what applies to the requested node. Do not require those model-specific choices
for ordinary filters.

If behavior is unclear, ask about the transformation, expected inputs/outputs or
external requirements that materially affect the implementation. Otherwise use
the user's description and source program. Verify third-party model schemas from
actual code/docs; preserve license/provenance. Distinguish the requested display
name from the algorithm actually executed.

## Package and processing contract

- Root `__init__.py` exposes `process(image, config) -> np.ndarray`. Put substantive
  operations in cohesive helper submodules, imported relatively. Follow the
  repository's annotations, multiline docstrings and 800-line file limit.
- Prefer literal module-level `CONFIG_JSON` and `MODEL_SOURCE` declarations in
  `__init__.py` for self-contained defaults and model metadata. The editor reads
  these without execution and hides their separate import loaders. `CONFIG_JSON`
  accepts a JSON string or dictionary; use `defaults`/`controls` to define typed
  node fields. Consult `docs/custom-nodes.md` and `configuration.py` for supported
  types, bounds and persistence. Do not use computed expressions for declarations.
  Use `color` controls for RGB hex/dynamic choices and `model` controls for local
  ONNX dropdowns with Browse. Model controls register paths without copying or
  loading weights; validate model compatibility in the processing callback.
  For packages that support execution selection, add an optional `device` control
  whose value has `backend` (`cpu`/`coreml`/`cuda`) and `apple_compute` (the built-in
  Apple unit choices). The callback receives the effective device after standard
  GPU/CPU fallback; saved preferences remain intact. Use
  `custom_nodes.devices.onnx_providers(config[key])` for ONNX provider options and
  recreate cached sessions when the effective device/model changes. The control
  alone does not accelerate arbitrary Python; keep model/provider compatibility
  checks in the worker. Do not probe or initialize inference in schema/UI code.
  Dynamic colors are metadata; reuse the viewer's contrast compositor when
  matching its measurement overlays, rather than inventing a separate style.
  For upright text with measurement-sized glyphs, call
  `custom_nodes.labels.defer_labels([(text, (x, y))], image.shape[:2])` with
  in-image anchors. It queues display annotations after viewer rotation/resizing
  and preview downsampling; subsequent mirrors and connected image branches move
  anchors without rotating glyphs. If it returns False outside a worker, use the
  shared `confidence_mask` layout to render standalone text. These annotations
  remain independent of subsequent pixel filters and blend colors.
  Optional root `config.json`/`model.json` remain fallbacks. Configuration is a JSON object
  with finite numbers, <=64 KiB. Validate node-specific types/ranges before use;
  reject booleans where numeric parameters are required. Use the callback's
  `config` argument for current settings, not the packaged defaults file.
- Input is a copied H×W×3 BGR display image, normally float32 in 0–255. It contains
  no sensor temperatures or radiometric arrays. Return a finite numeric H×W×3
  image with positive dimensions and <=4,000,000 pixels; runtime clips to 0–255
  float32. Preserve geometry/alignment when overlays must match sensor pixels.
- Avoid camera acquisition, viewer windows, UI loops and keyboard polling inside
  `process`. Work on the current pipeline image. Cache expensive initialization
  per package; initialize external resources lazily during processing, and reset
  caches when their relevant configuration changes.
- Module globals are private to each node and persist across frames. Do not add
  independent background workers or long-lived resources without accounting for
  the runtime's actual cleanup contract; unloading does not call a user teardown
  hook. Keep code imports free of unnecessary side effects.
- Only Python and JSON files are embedded, without a fixed package-size or file-count limit.
  Hidden files and caches are excluded; symbolic links and unsafe paths are
  refused. Non-code assets such as ONNX weights remain external. Document a
  supported format and model source. Use the shared `ModelResolver` API described
  in `docs/custom-nodes.md` for automatic models: retain one resolver per package,
  call `resolve` during processing and return the input while it returns `None`.
  Use catalog names or HTTPS sources with pinned SHA-256; archive sources also
  need a member path and archive checksum. Document download/cache/retry behavior
  and retain model licenses/provenance. Do not implement ad-hoc blocking downloads
  or install dependencies during import or per-frame processing.
- Custom code runs with the application's privileges, without a sandbox or
  execution timeout. Normal callback errors use existing pipeline-error handling
  and keep the last valid frame; that does not contain native crashes, infinite
  loops or arbitrary process termination. Explain actual external requirements,
  without adding confirmation flows beyond the user's authorized scope.

## Adapt and deliver

Extract only the image-processing part of an existing application; replace its
capture/UI loop with the callback. Preserve verified channel order, normalization,
layout, output schema and coordinate mapping. Keep unfamiliar export variants
explicitly unsupported rather than guessing. For ML nodes, record where weights
come from and their license, and make failed downloads/unsupported weights a clear error.
Dependencies already in the viewer are available; keep extra installations
optional and separate from project dependencies unless requested.

Provide usable JSON defaults, a short setup/usage document, and provenance when
adapting external code. Documentation should identify the selected folder, config
settings, reload behavior, external assets, tested backends, and limitations.
Link bundled examples from `docs/custom-nodes.md` and update `summary.txt`.

Verify application-owned behavior with synthetic images and fake inference or
external-resource boundaries. Load the finished folder with `load_folder`, create
it with `node('software', 'custom', **params)`, validate a save/export round trip,
and run it with `CustomProcessor`. Check the requested transformation, image
shape/dtype, configuration changes, cache separation and informative failures.
Use the existing offscreen Qt helper only when editor interactions changed.
Keep real model/network/device experiments outside normal pytest collection and
report separately what was and was not run. Finish with relevant tests, Ruff,
changed-file IDE inspection and complete diff review as required by `AGENTS.md`.
