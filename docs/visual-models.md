# Visual AI models

[Documentation home](../README.md) · [Imaging guide](imaging-guide.md)

Run installation and conversion commands from the repository root. Most downloadable models install automatically on first use; manual preinstallation is optional. DnCNN/FFDNet conversion is a separate developer operation.

Complete [setup](setup.md) first. Commands assume uv is on PATH; use `.venv/bin/uv`
for an environment-local installation. Conversion downloads developer dependencies
in a separate script environment and should be done outside camera acquisition.

## Enhancement models

To preinstall optional weights before a session:

```bash
uv run based-thermals-models espcn mewzoom
uv run based-thermals-models mewzoom-v0-2x
# Optional newer TrunkNet models (keeps V0 available):
uv run based-thermals-models mewzoom-v1-2x mewzoom-v1-4x
# Lightweight general-purpose 4x upscaler:
uv run based-thermals-models realesr-general-x4v3
# One-time conversion of the authors' denoising weights:
uv run scripts/export_visual_denoisers.py dncnn-25 ffdnet-gray
# Then launch the camera viewer (install commands do not launch it):
uv run based-thermals-desktop
```

In **Imaging → Pipeline**, insert **AI enhancement** in a
software tab. Start with **Native sensor** input and choose **ESPCN 3×**,
**MewZoom V0 2× (legacy)**, **MewZoom V0 4× (legacy)**, **MewZoom V1 2× (TrunkNet)**, or
**MewZoom V1 4× (TrunkNet)**, or **Real-ESRGAN general 4× v3**.
These models appear alongside ACNet and Anime4K09; do not
stack upscalers initially. ONNX upscalers use one fixed-scale pass, an enhancement blend,
and default to CPU execution. Capable systems also expose Apple Core ML with a
compute-device preference, or NVIDIA CUDA with the optional GPU runtime.
Saved GPU preferences are retained; an unavailable GPU backend tries the other
GPU before CPU. Explicit CPU selections remain CPU.
The **Execution** selector displays the effective backend on the current system,
including NVIDIA CUDA when a saved Apple preference falls back to NVIDIA.
The **Compute devices** selector displays **CPU only** and is disabled while
execution is set to CPU. Selecting Apple Core ML restores the saved device choice.

- [ESPCN](https://huggingface.co/onnxmodelzoo/super-resolution-10): luminance-only
  3× enhancement, about 240 KB, Apache-2.0. The published fixed 224×224 export
  uses overlapping tiles to preserve the camera feed's rectangular geometry.
- [MewZoom V0 2X](https://huggingface.co/andrewdalpino/MewZoom-V0-2X): full-RGB
  2× enhancement, about 7.65 MB ONNX download, Apache-2.0. This is the non-control variant.
- [MewZoom V0 4X](https://huggingface.co/andrewdalpino/MewZoom-V0-4X): full-RGB
  4× enhancement, about 57.3 MB, Apache-2.0. This is the non-control variant;
  its enhancement amount blends against bicubic, not independent noise/blur controls.
- [MewZoom V1 2X](https://huggingface.co/andrewdalpino/MewZoom-V1-2X): full-RGB
  TrunkNet, 5.3M parameters, about 23 MB ONNX download, Apache-2.0.
- [MewZoom V1 4X](https://huggingface.co/andrewdalpino/MewZoom-V1-4X): full-RGB
  TrunkNet, 21M parameters, Apache-2.0. This is larger than V0 4X, so it is
  not automatically faster. The larger UNet variants are not included.
- [Real-ESRGAN general x4v3](https://github.com/xinntao/Real-ESRGAN): compact
  full-RGB 4× model, BSD-3-Clause. Uses the checksum-pinned float32
  [SkillSafe ONNX conversion](https://huggingface.co/skillsafe-ai/realesr-general-x4v3)
  (4.64 MB), with its documented upstream checkpoint and conversion recipe.
  Enhancement amount is a bicubic blend, not Real-ESRGAN's two-weight denoise control.

For same-size denoising, insert another **AI enhancement** node before an
upscaler and select DnCNN or FFDNet. They process luminance and preserve chroma
and current image dimensions; input-size and pass controls are hidden:

- **DnCNN luminance (fixed noise 25)** uses KAIR's 17-layer `dncnn_25` weights.
  Adjust the denoising blend; its trained noise level is fixed.
- **FFDNet luminance (adjustable noise)** uses KAIR's 15-layer `ffdnet_gray`
  weights. Start at sigma 15; the 0–75 sigma control describes 8-bit image noise,
  not degrees or sensor calibration. Odd dimensions are padded and cropped back.

The old separate ONNX super-resolution, ONNX denoising and Apple Core ML ACNet
nodes are removed from Add/Insert menus. Use **AI enhancement** for those models
and execution controls. Existing pipelines and presets still load their legacy
nodes unchanged; model caches are reused and no reinstallation is needed.

The exporter downloads weights from the authors' [KAIR release](https://github.com/cszn/KAIR/releases/tag/v1.0)
over verified HTTPS, loads tensor-only checkpoints with `weights_only=True`,
checks ONNX validity and CPU/PyTorch parity at two rectangular sizes, then
installs weights, an offline checksum, and a provenance JSON in the model cache.
PyTorch and ONNX export dependencies live in a separate `uv` script environment;
the viewer does not require them. Both models are MIT licensed (Kai Zhang;
notice in `scripts/LICENSE-KAIR.txt`). Apple execution is optional and experimental;
provider availability is not a performance guarantee. Preferences survive on CPU-only systems.

## Image styling

Select a style in **Imaging → Pipeline → AI image styling** to download its
weights automatically on first use. Downloads run in the background; the live
feed and thermal measurements continue without the style effect until it is
ready. Progress appears in the viewer status and Pipeline panel. Verified weights
are cached for offline reuse. On a download failure, bypass/re-enable the node
to retry. Blend 0 or bypassed nodes do not trigger downloads.

This also applies to downloadable ONNX enhancement models. DnCNN/FFDNet still
require their documented one-time local export; the viewer does not install
Torch or launch an export automatically.

Manual pre-installation remains available if desired:

```bash
uv run based-thermals-models style-mosaic style-candy
uv run based-thermals-models style-rain-princess style-udnie style-pointillism style-line-art style-animegan-sketch
uv run based-thermals-desktop
```

In **Imaging → Pipeline**, add **AI image styling** and choose Mosaic, Candy,
Rain Princess, Udnie, Pointillism, Line Art, or AnimeGAN Portrait Sketch.
Start at the default 50% style blend. The separate node can go before
an upscaler, so styling does not interfere with your AI enhancement selections.
Blend 0 skips inference. The five painting styles use ONNX Model Zoo's verified
[Fast Neural Style exports](https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style)
(about 6.7 MB each; the upstream README lists BSD-3-Clause).

These exports have a fixed 224×224 RGB input in the **0–255** range, unlike
the enhancement models' normalized inputs. The node fits the entire frame into
that square with aspect-preserving padding, removes the padding from the styled
output, then restores the original image dimensions. This is a low-resolution
art effect, not an enhancement of sensor detail. CPU works on Linux and macOS;
Apple systems also expose optional Core ML execution, which needs local testing.
Linux NVIDIA systems expose CUDA with a compatible GPU runtime.
Saved GPU preferences are retained but ignored on CPU-only systems.

**Line Art** uses the [Informative Drawings ONNX conversion](https://github.com/josephrocca/image-to-line-art-js)
of the [MIT-licensed original project](https://github.com/carolineec/informative-drawings).
It works at a bounded 256×256 resolution, consumes normalized RGB 0–1, and
returns a single-channel drawing expanded to RGB for blending.

**AnimeGAN Portrait Sketch** uses the author's [public ONNX release](https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/releases/tag/1.0).
Its input/output are NHWC RGB in −1 to 1 at a 512×512 working resolution.
The [AnimeGANv3 terms](https://github.com/TachibanaYoshino/AnimeGANv3#scroll-license)
restrict these weights to **non-commercial use**; the installer and selector
identify this restriction. It was trained for portraits, not thermal scenes;
this node applies it to the full frame without face detection or alignment.
Both new model families use aspect-preserving padding and restore the incoming
image dimensions. All weights are checksum-verified; Udnie's publisher checksum
is resolved at installation and retained alongside the cached model.

AnimeGAN comic and 8-bit styles are demonstrated upstream, but no verified public
ONNX weight downloads were found for them, so they are not exposed as working
options. Reference-image style transfer is not integrated.

These are experimental natural-image models, not thermal-measurement models.
They may invent visual detail or flicker between frames. Hover temperatures,
spots, graphs, and saved radiometric data still use the existing sensor measurements.
Native 256×192 input produces 768×576 with ESPCN, 512×384 with either MewZoom 2X,
or 1024×768 with either MewZoom 4X or Real-ESRGAN.
Full 512×384 preview input is also supported; outputs must remain below the
4-megapixel budget. Apple provider availability does not prove all operations
execute on the GPU; each model needs local speed/quality testing.

Weights are checksum-verified and cached outside the repository (macOS:
`~/Library/Caches/based-thermals/models`; Linux: `$XDG_CACHE_HOME/based-thermals/models`
or `~/.cache/based-thermals/models`). `BASED_THERMALS_MODEL_DIR` overrides the directory;
the obsolete `TOPDON_MODEL_DIR` override is no longer read.
Inference never downloads weights or executes model-provided Python code.
V1 installation obtains the publisher's SHA256 from its Git LFS pointer over
verified HTTPS and saves that checksum beside the verified weights for offline
use. V0 and ESPCN retain their built-in checksum pins. Existing V0 presets and
weights are preserved; selecting V1 never silently replaces a saved V0 model.
For V1, Real-ESRGAN, and denoiser Apple inference, the runtime specializes the symbolic input dimensions
to the actual frame size before Core ML conversion, avoiding ORT's dynamic
layout-conversion axis error. Changing the input size creates a new Apple
session and may trigger compilation again. CPU inference remains dynamic and
unchanged; weights and measurements are never rewritten.
On macOS the downloader supplements Python's trust store with `/etc/ssl/cert.pem`
to support framework Python installations with missing default CA roots. TLS
certificate and hostname verification remain enabled; explicit `SSL_CERT_FILE`
and `SSL_CERT_DIR` settings are respected. Linux keeps its default trust configuration.
CPU, Apple, and NVIDIA inference run in persistent spawned helpers, containing
native crashes and keeping model work off the GUI and USB acquisition threads.
