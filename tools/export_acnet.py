"""Export MIT-licensed upstream ACNet legacy weights to portable ONNX.

Developer tool only: pip install onnx==1.17.0. Runtime uses existing OpenCV.
Pass core/include/AC/Core/Model/Param/ACNet.p from Anime4KCPP v3.2.0,
commit 50c5d1b99965d804eeecfab9b48acfcaffddee3d.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def export(source: Path, destination: Path) -> None:
    text = source.read_text()
    destination.mkdir(parents=True, exist_ok=True)
    for level in range(4):

        def array(name, level=level):
            match = re.search(
                rf"ACNetLegacy_HDN{level}_NHWC_{name}\[\]\s*=\s*\{{(.*?)\}};",
                text,
                re.DOTALL,
            )
            if match is None:
                raise ValueError(f"Missing HDN{level} {name}")
            return np.array(
                re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", match[1]), dtype=np.float32
            )

        kernels, biases = array("kernels"), array("biases")
        if kernels.size != 4712 or biases.size != 72:
            raise ValueError("Unexpected model dimensions")
        nodes, initializers = [], []
        initializers.append(
            numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")
        )
        current = "input"
        for layer in range(9):
            cin = 1 if layer == 0 else 8
            offset = 0 if layer == 0 else 72 + (layer - 1) * 576
            weight = kernels[offset : offset + 8 * 9 * cin].reshape(8, 3, 3, cin)
            initializers.extend(
                [
                    numpy_helper.from_array(weight.transpose(0, 3, 1, 2).copy(), f"w{layer}"),
                    numpy_helper.from_array(biases[layer * 8 : (layer + 1) * 8], f"b{layer}"),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Pad", [current, "pads"], [f"p{layer}"], mode="edge"),
                    helper.make_node(
                        "Conv",
                        [f"p{layer}", f"w{layer}", f"b{layer}"],
                        [f"c{layer}"],
                        kernel_shape=[3, 3],
                    ),
                    helper.make_node("Relu", [f"c{layer}"], [f"r{layer}"]),
                ]
            )
            current = f"r{layer}"
        final = kernels[4680:].reshape(2, 2, 8).transpose(2, 0, 1)[:, None].copy()
        initializers.append(numpy_helper.from_array(final, "upscale"))
        nodes.append(
            helper.make_node(
                "ConvTranspose",
                [current, "upscale"],
                ["output"],
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        )
        graph = helper.make_graph(
            nodes,
            f"acnet-legacy-hdn{level}",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, "h", "w"])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "h2", "w2"])],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 11)],
            producer_name="topdon-duo ACNet export",
            ir_version=7,
        )
        model.doc_string = "Anime4KCPP v3.2.0 ACNet legacy; MIT license; see NOTICE.txt."
        onnx.checker.check_model(model)
        onnx.save(model, destination / f"acnet-legacy-hdn{level}.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
