"""Export KAIR's MIT-licensed, batch-normalization-merged DnCNN blind weights.

Developer dependencies: torch (CPU is sufficient), numpy, onnx.
Source: https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth
Architecture: KAIR commit fc1732f4a4514e42ce15e5b3a1e18c828af47a1e,
models/network_dncnn.py, DnCNN(nb=20, act_mode='R').
The graph includes input minus predicted noise, so output is the clean image.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


def export(source: Path, destination: Path) -> None:
    state = torch.load(source, map_location="cpu", weights_only=True)
    expected_keys = {
        f"model.{2 * layer}.{kind}" for layer in range(20) for kind in ("weight", "bias")
    }
    if set(state) != expected_keys:
        raise ValueError(
            "Expected the 20-layer, batch-normalization-merged grayscale blind checkpoint"
        )
    nodes, initializers = [], []
    current = "input"
    for layer in range(20):
        cin, cout = (1 if layer == 0 else 64), (1 if layer == 19 else 64)
        weight = state[f"model.{2 * layer}.weight"].numpy()
        bias = state[f"model.{2 * layer}.bias"].numpy()
        if weight.shape != (cout, cin, 3, 3) or bias.shape != (cout,):
            raise ValueError(f"Unexpected dimensions in convolution {layer}")
        for array, name in ((weight, f"w{layer}"), (bias, f"b{layer}")):
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid float32 weights in convolution {layer}")
            initializers.append(numpy_helper.from_array(array, name))
        convolution = f"conv{layer}"
        nodes.append(
            helper.make_node(
                "Conv",
                [current, f"w{layer}", f"b{layer}"],
                [convolution],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        )
        current = convolution
        if layer != 19:
            current = f"relu{layer}"
            nodes.append(helper.make_node("Relu", [convolution], [current]))
    nodes.append(helper.make_node("Sub", ["input", current], ["output"]))
    graph = helper.make_graph(
        nodes,
        "dncnn-gray-blind",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "h", "w"])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        producer_name="topdon-duo DnCNN export",
        ir_version=7,
    )
    model.doc_string = "KAIR DnCNN grayscale blind denoiser; MIT license; see NOTICE.txt."
    helper.set_model_props(
        model,
        {
            "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "upstream_commit": "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e",
        },
    )
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
