# DDColor Custom node

Select **Custom → Open module folder… → `examples/custom_nodes/DDColor`**.
This is a separate example from Based GAN. It runs DDColor ONNX inference using
the viewer's existing NumPy, OpenCV and ONNX Runtime dependencies.

DDColor predicts Lab **a/b color channels** and combines them with the input's
original-resolution **L lightness**. It retains image dimensions and thermal
lightness structure; conversion back to display BGR clips colors to the display
gamut. Predicted hues are guesses. Native temperature arrays and measurements
remain separate.

## Models and loading

With **Local ONNX model override** empty, the processing worker downloads a
[DDColor tiny FP16 512 export](https://huggingface.co/edgetools/ddcolor/tree/c4c98361d19eda29908cc960be39e8eeb44fc530)
from a version-pinned URL and verifies its SHA-256. It is about 135 MB, with
float32 input/output and FP16 internal weights. This is a separate export of
DDColor, rather than a model from the linked instant-high archive. The worker
returns the input while downloading; folder loading never downloads or loads
weights. The viewer's shared cache and download status are used. Toggle bypass
off/on to retry a failed download.

For the faster tested variant, Browse for the local **`ddcolor_paper_tiny.onnx`**
from [instant-high/DDColor-onnx's model archive](https://github.com/instant-high/DDColor-onnx).
The archive is about 1.9 GB; its tiny member is 270 MB with a 256×256 model input.
The two large exports are about 980 MB each and were not tested.

This checkout retains tested weights and reports in `.artifacts/`, excluded from
Git and from embedded package snapshots. For a ready local-model pipeline, import:

```text
examples/custom_nodes/DDColor/.artifacts/worker/ddcolor_paper_tiny-pipeline.json
```

That preset selects the original 256 model on Apple CPU + GPU and uses an
absolute local model path. It is specific to this checkout. Browse dialogs may
require **Cmd+Shift+.** to reveal `.artifacts` on macOS, or paste its full file
path. The package itself remains portable; models stay external to saved presets.
Saved presets embed code. Reselect the module folder after source changes, then
save again to update embedded code.

## Controls

| Control | Behavior |
| --- | --- |
| Local ONNX model override | Empty uses automatic 512 export; Browse selects local weights |
| Inference device | CPU, Apple Core ML or NVIDIA CUDA through the viewer's capability fallback |
| Color strength | Scales predicted chroma; zero returns input without inference |
| Invert model input | Changes model polarity while retaining original output lightness |
| Model input rotation (CCW) | Rotates before inference and undoes rotation on chroma; output stays aligned |

Use a grayscale thermal branch before palettes, annotations and upscaling.
Model input rotation is independent of the final viewer rotation. Sessions are
cached per node and rebuilt when the model or effective device changes. Compatible
exports have one fixed batch-one float32 RGB NCHW input and one matching-resolution
float32 two-channel Lab output. No additional ImageNet normalization is applied.

## Local verification

Apple M4, viewer Python 3.12 / ONNX Runtime 1.30, five timed still-image worker
calls after three warmup calls, including preprocessing/postprocessing:

| Model | CPU median | Core ML CPU + GPU median |
| --- | ---: | ---: |
| Original tiny, 256×256 | 148.6 ms | 89.1 ms |
| FP16 tiny export, 512×512 | 782.4 ms | 405.2 ms |

Initial Core ML compilation plus first inference took approximately 45–47 seconds
for these exports. GPU provider registration was verified; this does not establish
that every graph operator runs on GPU. The 512 model is smaller on disk but slower
because its spatial input is larger. Neither measured configuration fits the
camera's 40 ms frame budget. CUDA and live-video stability remain untested.

Both CPU outputs match the linked wrapper's rounded uint8 result exactly on the
same saved thermal frame. Deterministic tests use fake sessions, validate a saved
pipeline JSON round trip and check lightness, channel order, input controls,
geometry, session ownership and errors. Actual external-weight checks use:

```sh
uv run experiments/experiment_ddcolor.py \
  --models examples/custom_nodes/DDColor/.artifacts/ddcolor_paper_tiny.onnx \
           examples/custom_nodes/DDColor/.artifacts/ddcolor-tiny-fp16.onnx \
  --image /path/to/grayscale-thermal.png \
  --output examples/custom_nodes/DDColor/.artifacts/worker --runs 5
```

Add `--backends cpu` on systems without Core ML; CUDA can be selected explicitly.
`--rotation 90` or `--invert` test alternative model inputs without camera access.
These experiments remain outside normal pytest collection.

## Sources

Preprocessing/reconstruction follow
[instant-high/DDColor-onnx](https://github.com/instant-high/DDColor-onnx/tree/fa513b61fdb987398f03d3ebd9b5205f310262bf).
Original model architecture/training:
[piddnad/DDColor](https://github.com/piddnad/DDColor), *DDColor: Towards
Photo-Realistic Image Colorization via Dual Decoders*, Kang et al., ICCV 2023.
The original project publishes Apache-2.0; its license is retained in
`LICENSE-DDColor.txt`. The automatic export is published by
[edgetools](https://huggingface.co/edgetools/ddcolor). Exact revisions, model hashes
and the tested archive member are recorded in `provenance.json`.
