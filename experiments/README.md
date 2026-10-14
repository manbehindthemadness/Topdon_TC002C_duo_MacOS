# External resource checks

Run the model checks below explicitly from the repository root using the configured project
environment. They are outside pytest's normal `tests/` collection and do not access
the camera.

## Optional visual ONNX weights

```sh
.venv/bin/python experiments/experiment_onnx_visual.py --models realesr-general-x4v3
.venv/bin/python experiments/experiment_onnx_visual.py
# On macOS, also check V1 Core ML initialization:
.venv/bin/python experiments/experiment_onnx_visual.py --coreml
```

Install the desired optional models separately using the README's model installer
or documented conversion commands. The experiment verifies installed checksums
before inference and reports missing weights as skipped. It checks scaled output
geometry and compares V1 dynamic/fixed dimension CPU outputs when those weights
are present. It does not download models or export weights. Core ML requires its
ONNX Runtime provider and may create compiler caches; compilation restrictions
are reported as failures. A failed check returns a nonzero exit status.

Normal application tests retain fake sessions and isolated temporary model files
to cover pipeline behavior independently of the user's model cache.

## Head Pose

Preserve the upstream checkout and prepare optional pinned external weights separately:

```sh
git submodule update --init -- examples/custom_nodes/head-pose/upstream
uv run --python .venv/bin/python --script tools/prepare_head_pose.py
.venv/bin/python experiments/experiment_head_pose.py \
  --images /path/to/head-image.png --backends cpu coreml --pose-backend cpu
```

Preparation uses isolated ONNX/certificate dependencies and writes into ignored
`diagnostics/head-pose/models`. Preparation is optional for viewer use: the Custom
node's JSON detector dropdown and empty local overrides download audited pinned
weights automatically. The experiment measures real Custom worker cold/warm
timings, writes annotated stills and predictions, and exports portable local-weight
presets under `diagnostics/head-pose/results`. It does not acquire a camera or alter
preferences. CPU-only hosts can use `--backends cpu`; unavailable GPU requests use
the normal fallback, and the report records the registered providers. The
[Head Pose README](../examples/custom_nodes/head-pose/README.md) documents the
verified model schemas, measured backends, update workflow and thermal limitations.

## First-use denoiser preparation

```sh
.venv/bin/python experiments/experiment_model_preparation.py dncnn-25 ffdnet-gray --model-dir /tmp/based-thermals-denoiser-check
```

This explicitly downloads isolated conversion dependencies and official weights,
checks conversion parity, verifies offline cache reuse, and runs same-size CPU
inference. Use a separate model directory as shown to preserve viewer data. It
requires uv and internet access; normal pytest never performs conversion or downloads.
