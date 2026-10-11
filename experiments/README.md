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
