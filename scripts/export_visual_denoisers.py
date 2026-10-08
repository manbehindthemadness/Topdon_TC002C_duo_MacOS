# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6,<3", "onnx>=1.17,<2", "onnxruntime>=1.23,<2", "numpy>=2.2,<3"]
# ///
"""One-time KAIR weight conversion; the viewer never imports PyTorch.

Architectures follow cszn/KAIR's network_dncnn.py and network_ffdnet.py.
Copyright (c) 2019 Kai Zhang; MIT notice in LICENSE-KAIR.txt.
FFDNet accepts even dimensions; the viewer pads/crops odd dimensions.
"""

import argparse
import hashlib
import io
import json
import sys
import tempfile
import urllib.request
from itertools import pairwise
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from topdon_duo.onnx_models import download_ssl_context, model_path, verified_model

CHECKPOINTS = {"dncnn-25": "dncnn_25.pth", "ffdnet-gray": "ffdnet_gray.pth"}


class Denoiser(nn.Module):
    def __init__(self, ffdnet=False):
        super().__init__()
        self.ffdnet = ffdnet
        count = 15 if ffdnet else 17
        channels = [5 if ffdnet else 1] + [64] * (count - 1) + [4 if ffdnet else 1]
        layers = []
        for index, (incoming, outgoing) in enumerate(pairwise(channels)):
            layers.append(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=True))
            if index < count - 1:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x, sigma=None):
        if not self.ffdnet:
            return x - self.model(x)
        down = F.pixel_unshuffle(x, 2)
        noise = sigma.expand(-1, -1, down.shape[2], down.shape[3])
        return F.pixel_shuffle(self.model(torch.cat((down, noise), dim=1)), 2)


def checkpoint(model):
    url = "https://github.com/cszn/KAIR/releases/download/v1.0/" + CHECKPOINTS[model]
    with urllib.request.urlopen(url, timeout=60, context=download_ssl_context()) as source:
        data = source.read(10_000_001)
    if len(data) > 10_000_000:
        raise ValueError("Checkpoint exceeds the 10 MB safety limit")
    return data, url


def export_model(model, force=False):
    path = model_path(model)
    if not force:
        try:
            verified_model(model)
        except ValueError:
            pass
        else:
            print(f"Already installed: {path}", flush=True)
            return path
    print(f"Downloading official KAIR {CHECKPOINTS[model]}...", flush=True)
    weights, url = checkpoint(model)
    network = Denoiser(model == "ffdnet-gray").eval()
    # No unrestricted pickle loading and no downloaded Python code.
    network.load_state_dict(
        torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True), strict=True
    )
    torch.set_num_threads(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="denoiser-export-", dir=path.parent) as directory:
        temporary = Path(directory) / "model.onnx"
        torch.manual_seed(17)
        x = torch.rand(1, 1, 24, 32)
        args = (x, torch.full((1, 1, 1, 1), 15 / 255)) if network.ffdnet else (x,)
        names = ["x", "sigma"] if network.ffdnet else ["x"]
        print(f"Exporting and checking {model}...", flush=True)
        torch.onnx.export(
            network,
            args,
            str(temporary),
            input_names=names,
            output_names=["output"],
            dynamic_axes={"x": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
            opset_version=17,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(str(temporary)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            str(temporary), sess_options=options, providers=["CPUExecutionProvider"]
        )
        errors = []
        for height, width in ((24, 32), (32, 46)):
            for noise in (0, 15, 75) if network.ffdnet else (25,):
                image = torch.rand(1, 1, height, width)
                feed = {"x": image.numpy()}
                with torch.no_grad():
                    if network.ffdnet:
                        sigma = torch.full((1, 1, 1, 1), noise / 255)
                        feed["sigma"] = sigma.numpy()
                        reference = network(image, sigma).numpy()
                    else:
                        reference = network(image).numpy()
                output = session.run(None, feed)[0]
                np.testing.assert_allclose(output, reference, rtol=1e-4, atol=1e-5)
                if output.shape != reference.shape or not np.isfinite(output).all():
                    raise ValueError("Invalid exported output")
                errors.append(float(np.max(np.abs(output - reference))))
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        sidecar = Path(directory) / "model.sha256"
        sidecar.write_text(digest + "\n")
        provenance = Path(directory) / "model.json"
        provenance.write_text(
            json.dumps(
                {
                    "source": url,
                    "checkpoint_sha256": hashlib.sha256(weights).hexdigest(),
                    "onnx_sha256": digest,
                    "max_parity_error": max(errors),
                    "torch": torch.__version__,
                    "onnx": onnx.__version__,
                    "ort": ort.__version__,
                    "license": "MIT; Copyright (c) 2019 Kai Zhang",
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(path)
        sidecar.replace(path.with_suffix(".sha256"))
        provenance.replace(path.with_suffix(".json"))
    print(f"Installed {path} (parity max error {max(errors):.3g})", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=tuple(CHECKPOINTS))
    parser.add_argument("--force", action="store_true", help="Re-export installed models")
    args = parser.parse_args()
    try:
        for model in args.models:
            export_model(model, args.force)
    except Exception as exc:  # noqa: BLE001 - report third-party export/runtime errors at the CLI
        print(f"Denoiser export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
