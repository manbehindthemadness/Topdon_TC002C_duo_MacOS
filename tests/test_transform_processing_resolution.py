"""
Downstream raster consumers retain thermal detail, coordinate units and allocation bounds.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from support.pipeline import raw_pipeline
from test_transform_resampling import striped_frame

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.enhancement_limits import MAX_PIXELS
from based_thermals.pipeline import node
from based_thermals.pipeline_processing import PipelineProcessor
from based_thermals.processing.branch import BranchProcessor
from based_thermals.processing.resolution import bounded_promotion


def test_custom_after_transform_receives_detail_and_runs_once(tmp_path: Path) -> None:
    """
    Arbitrary callbacks see the preserved raster once, rather than being replayed after a blend.
    """
    (tmp_path / "__init__.py").write_text(
        'calls = 0\n'
        'def process(image, config):\n'
        '    global calls\n'
        '    calls += 1\n'
        '    assert calls == 1\n'
        '    assert image.shape == (384, 512, 3)\n'
        '    return 255 - image\n', encoding="utf-8",
    )
    custom = node("software", "custom")
    custom["params"].update(load_folder(tmp_path))
    document = raw_pipeline(node("software", "brightness", amount=100))
    document["branches"]["B"] = raw_pipeline(
        node("software", "transform", width=50, height=50, method="nearest"), custom,
    )["software"][:-1]
    document["software"].insert(-1, node("software", "combine", mode="replace", tab="B"))
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, _ = processor.process(striped_frame(), None, document, scale=2)
        np.testing.assert_array_equal(output[20, :250, 0], (1 - np.arange(250) % 2) * 255)
        np.testing.assert_array_equal(output[250:, 300:], 255)
    finally:
        processor.close()


@pytest.mark.parametrize("unit,x,y", [("pixels", 10, 20), ("percent", 10 / 256 * 100, 20 / 192 * 100)])
def test_second_transform_keeps_positions_and_labels_in_declared_units(
    unit: str, x: float, y: float, tmp_path: Path,
) -> None:
    """
    Promotion must not double a later pixel offset or move its normalized deferred anchor.
    """
    processor = BranchProcessor(apple_available=False, nvidia_available=False)

    (tmp_path / "__init__.py").write_text(
        'from based_thermals.custom_nodes.labels import defer_labels\n'
        'def process(image, config):\n'
        '    defer_labels([("Origin", (0, 0))], image.shape[:2])\n'
        '    return image\n', encoding="utf-8",
    )
    custom = node("software", "custom")
    custom["params"].update(load_folder(tmp_path))
    document = raw_pipeline(custom,
        node("software", "transform", width=50, height=50, method="nearest"),
        node("software", "transform", position_unit=unit, position_x=x, position_y=y, method="nearest"),
        node("software", "brightness", amount=0),
    )
    try:
        output, _ = processor.process(striped_frame(), None, document)
        assert output.shape == (384, 512, 3)
        assert processor.output_logical_size == (256, 192)
        assert processor.custom.labels[0].x == pytest.approx(10 / 255)
        assert processor.custom.labels[0].y == pytest.approx(20 / 191)
        region = processor.output_coverage
        assert region is not None
        assert region[40, 20] == 1 and region[39, 20] == 0 and region[40, 19] == 0
        np.testing.assert_array_equal(output[50, 21:250, 0], (1 - np.arange(229) % 2) * 255)
    finally:
        processor.close()


@pytest.mark.parametrize("following", [
    [node("software", "interpolation", scale=4)],
    [node("software", "transform", resize_mode="resolution", width=400, height=400)],
    [node("software", "onnx_superresolution", model="espcn", input="current")],
])
def test_promotion_reserves_downstream_allocation_budget(following: list[dict[str, Any]]) -> None:
    """
    A protective raster increase cannot turn a valid declared upscale into an oversized allocation.
    """
    size = bounded_promotion((256, 192), (2560, 1920), following, (256, 192))
    factor = 3 if following[0]["type"] == "onnx_superresolution" else 4
    assert size[0] >= 256 and size[1] >= 192
    assert size[0] * size[1] * factor**2 <= MAX_PIXELS


def test_native_model_input_resets_density_without_changing_its_requested_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Input Native keeps its established resolution even after a denser processing canvas.
    """
    inputs: list[tuple[int, ...]] = []

    class FakeUpsampler:
        """
        Observe the actual model input boundary without downloading weights or opening a device.
        """

        @staticmethod
        def apply(image: np.ndarray, *_args: Any, **_kwargs: Any) -> np.ndarray:
            """
            Return a deterministic 3x model output.
            """
            inputs.append(image.shape)
            return cv2.resize(image, (768, 576), interpolation=cv2.INTER_NEAREST)

        def close(self) -> None:
            """
            Match the model cache's cleanup contract.
            """

    monkeypatch.setattr("based_thermals.processing.branch.ONNXUpsampler", FakeUpsampler)
    processor = BranchProcessor(apple_available=False, nvidia_available=False)
    document = raw_pipeline(node("software", "transform", width=50, height=50),
        node("software", "onnx_superresolution", model="espcn", input="native"),
    )
    try:
        output, _ = processor.process(striped_frame(), None, document)
        assert inputs == [(192, 256, 3)]
        assert output.shape == (576, 768, 3)
        assert processor.output_logical_size == (768, 576)
    finally:
        processor.close()


@pytest.mark.parametrize("nested", [False, True])
def test_nested_combine_and_mask_only_inputs_preserve_promoted_detail(nested: bool) -> None:
    """
    Detail must survive a second branch consumer and masks must influence its raster budget too.
    """
    document = raw_pipeline(node("software", "brightness", amount=100))
    document["branches"]["C"] = raw_pipeline(
        node("software", "transform", width=50, height=50, method="nearest"),
        node("software", "gamma", amount=2),
    )["software"][:-1]
    if nested:
        document["branches"]["B"] = raw_pipeline(node("software", "brightness", amount=100),
            node("software", "combine", tab="C", mode="replace"),
            node("software", "gamma", amount=2),
        )["software"][:-1]
        combine = node("software", "combine", tab="B", mode="replace")
    else:
        document["branches"]["B"] = raw_pipeline()["software"][:-1]
        document["branches"]["B"][1]["params"].update(low=60, high=100)
        combine = node("software", "combine", tab="B", mode="replace",
                       mask_source="C", mask_kind="threshold")
    document["software"].insert(-1, combine)
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        output, _ = processor.process(striped_frame(), None, document, scale=2)
        expected = np.arange(250) % 2 * 255
        if not nested:
            expected = 255 - expected
        np.testing.assert_array_equal(output[20, :250, 0], expected)
    finally:
        processor.close()
