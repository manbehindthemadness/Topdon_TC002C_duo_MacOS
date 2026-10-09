import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from topdon_duo import onnx_models


def fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    bad_output: Any = False,
    fixed_shape: Any = False,
    invert_style: Any = False,
) -> Any:
    """
    Fake runtime.
    """
    calls = []
    monkeypatch.setattr("topdon_duo.onnx_upsampling.verified_model", lambda model: model.encode())
    monkeypatch.setattr("topdon_duo.onnx_upsampling.explicit_coreml_padding", lambda data: data)
    monkeypatch.setattr(
        "topdon_duo.onnx_upsampling.coreml_input_shape_overrides",
        lambda data, shape, **kwargs: {
            "batch": shape[0],
            "height": shape[2],
            "width": shape[3],
        },
    )

    class Options:
        def __init__(self) -> None:
            """
            Init.
            """
            self.overrides = {}

        def add_free_dimension_override_by_name(self, name: Any, value: Any) -> None:
            """
            Add free dimension override by name.
            """
            self.overrides[name] = value

    class Session:
        def __init__(self, data: Any, **kwargs: Any) -> None:
            """
            Init.
            """
            self.model = data.decode()
            self.providers = kwargs["providers"]
            calls.append(kwargs)

        def disable_fallback(self) -> None:
            """
            Disable fallback.
            """

        def get_providers(self) -> Any:
            """
            Get providers.
            """
            return [p[0] if isinstance(p, tuple) else p for p in self.providers]

        def get_inputs(self) -> Any:
            """
            Get inputs.
            """
            inputs = [SimpleNamespace(name="x", shape=(1, 1, 224, 224) if fixed_shape else ())]
            if self.model == "ffdnet-gray":
                inputs.append(SimpleNamespace(name="sigma", shape=(1, 1, 1, 1)))
            return inputs

        def run(self, _outputs: Any, feed: Any) -> Any:
            """
            Run.
            """
            blob = feed["x"]
            calls.append(blob.copy())
            if self.model == "ffdnet-gray":
                calls[-1] = (blob.copy(), feed["sigma"].copy())
            factor = onnx_models.MODELS[self.model]["factor"]
            output = np.repeat(np.repeat(blob, factor, axis=2), factor, axis=3)
            if onnx_models.MODELS[self.model].get("output_channels") == 1:
                output = output.mean(axis=1, keepdims=True)
            if invert_style and onnx_models.MODELS[self.model].get("task") == "style":
                output = 255 - output
            if bad_output:
                output[:] = np.nan
            return [output]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=Options,
            InferenceSession=Session,
            get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
    )
    return calls
