"""Pipeline behavior: ordered image operations, safe ownership, and portable state."""

from typing import Any
from unittest.mock import Mock

import numpy as np

from topdon_duo.camera import (
    HEADER_U16,
    SENSOR_PIXELS,
    decode_duo_frame,
)
from topdon_duo.hardware_controls import HARDWARE_CONTROLS
from topdon_duo.pipeline import (
    default_pipeline,
    node,
)
from topdon_duo.pipeline_processing import PipelineProcessor

from .frames import frame_with_preview


def raw_pipeline(*nodes: Any) -> Any:
    """
    Raw pipeline.
    """
    document = default_pipeline()
    document["software"][0]["params"]["source"] = "raw"
    document["software"] = [
        document["software"][0],
        node("software", "range", low=10.0, high=60.0),
        *nodes,
        node("software", "output"),
    ]
    return document


def process(document: Any, scale: Any = 1) -> Any:
    """
    Process.
    """
    frame, _ = frame_with_preview()
    words = np.frombuffer(frame, dtype="<u2").copy()
    yy, xx = np.indices((192, 256))
    temperatures = 25 + xx / 20 + ((xx // 7 + yy // 7) % 2) * 5
    words[HEADER_U16 : HEADER_U16 + SENSOR_PIXELS] = ((temperatures.ravel() + 50) * 64).astype(
        np.uint16
    )
    frame = words.tobytes()
    _, raw, _ = decode_duo_frame(frame)
    return PipelineProcessor().process(frame, raw.astype(np.float32), document, scale=scale)[0]


def fake_hardware() -> Any:
    """
    Fake hardware.
    """
    hw = Mock(
        original={"baseline": True},
        gamma=50,
        boost=0,
        processing_preset="balanced",
        fixed_range=False,
    )
    hw.state.return_value = {
        name: {"value": spec.minimum, "enabled": False, "available": True}
        for name, spec in HARDWARE_CONTROLS.items()
    }

    def set_value(name: Any, value: Any, enabled: Any) -> None:
        """
        Set value.
        """
        hw.state.return_value[name].update(value=value, enabled=enabled)

    hw.set.side_effect = set_value
    hw.set_processing_preset.side_effect = lambda value: setattr(hw, "processing_preset", value)
    hw.set_tone.side_effect = lambda gamma, boost: (
        setattr(hw, "gamma", gamma),
        setattr(hw, "boost", boost),
    )
    hw.restore_tone.side_effect = lambda: (setattr(hw, "gamma", 50), setattr(hw, "boost", 0))
    hw.set_fixed_range.side_effect = lambda value: setattr(hw, "fixed_range", value)
    hw.restore_fixed_range.side_effect = lambda: setattr(hw, "fixed_range", False)
    return hw
