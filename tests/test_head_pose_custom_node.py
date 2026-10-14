"""
Head Pose package portability, crop geometry, session ownership and inference boundaries.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from support.pipeline import raw_pipeline

from based_thermals.custom_nodes.bundle import load_folder
from based_thermals.custom_nodes.runtime import CustomProcessor
from based_thermals.pipeline import node, validate_pipeline

FOLDER = Path(__file__).parents[1] / "examples/custom_nodes/head-pose/nodes/Head Pose"


class FakeSession:
    """
    Supply independently specified boxes and angles while recording all input tensors.
    """

    instances: ClassVar[list[Any]] = []
    boxes = np.array([[[7, .2, .25, .6, .75, .9],
                       [7, .7, .1, .9, .3, .95],
                       [0, .1, .1, .9, .9, .99]]], dtype=np.float32)
    bad = ""

    def __init__(self, path: str, options: Any, providers: list[Any]) -> None:
        """
        Identify the fake detector by its filename, with no real model access.
        """
        self.detector = "detector" in Path(path).name
        self.pose_size = 128 if "128" in Path(path).name else 64
        self.providers = [entry if isinstance(entry, str) else entry[0] for entry in providers]
        self.tensors: list[np.ndarray] = []
        assert options.intra_op_num_threads == 4
        self.instances.append(self)

    def disable_fallback(self) -> None:
        """
        Match the cached session interface.
        """

    def get_providers(self) -> list[str]:
        """
        Expose effective device choices for cache tests.
        """
        return self.providers

    def get_inputs(self) -> list[Any]:
        """
        Advertise static float32 batch-1 tensors or an unsupported dynamic batch.
        """
        size = 32 if self.detector else self.pose_size
        batch = "N" if self.bad == "schema" else 1
        return [SimpleNamespace(name="images", type="tensor(float)", shape=[batch, 3, size, size])]

    def get_outputs(self) -> list[Any]:
        """
        Describe the independently generated detection or pose results.
        """
        shapes = {"label_xyxy_score": [1, 3, 6]} if self.detector else {
            "cos_sin": [1, 6], "kappa": [1, 3],
        }
        return [SimpleNamespace(name=name, shape=shape, type="tensor(float)")
                for name, shape in shapes.items()]

    def get_modelmeta(self) -> Any:
        """
        Simulate only detectors approved by the external preparation step.
        """
        meta = {"based_thermals.detector_pruned": "1",
                "based_thermals.detector_norm": "imagenet"} if self.detector else {}
        if self.bad == "unprepared":
            meta = {}
        return SimpleNamespace(custom_metadata_map=meta)

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Emit yaw=90, pitch=0, roll=0 without deriving expected values from node code.
        """
        self.tensors.append(feed["images"].copy())
        if self.detector:
            assert names == ["label_xyxy_score"]
            boxes = self.boxes.copy()
            if self.bad == "boxes":
                boxes[0, 0, 2] = np.nan
            return [boxes]
        assert names == ["cos_sin", "kappa"]
        pairs = np.array([[0, 1, 1, 0, 1, 0]], dtype=np.float32)
        if self.bad == "pose":
            pairs[0, 0] = np.nan
        if self.bad == "zero":
            pairs[:] = 0
        return [pairs, np.array([[20, 30, 40]], dtype=np.float32)]


@pytest.fixture
def setup_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[Any, ...]]:
    """
    Load the portable folder and round-trip a document before fake worker inference.
    """
    FakeSession.instances = []
    FakeSession.bad = ""
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)
    monkeypatch.setattr("onnxruntime.get_available_providers",
                        lambda: ["CPUExecutionProvider", "CoreMLExecutionProvider"])
    for name in ("detector.onnx", "pose.onnx"):
        (tmp_path / name).write_bytes(b"fake inference boundary")
    params = load_folder(FOLDER)
    config = json.loads(params["config"])
    config.update(detector_path=str(tmp_path / "detector.onnx"),
                  pose_path=str(tmp_path / "pose.onnx"))
    config["detector_device"]["backend"] = "cpu"
    params["config"] = json.dumps(config)
    item = node("software", "custom", **params)
    document = validate_pipeline(json.loads(json.dumps(raw_pipeline(item))))
    processor = CustomProcessor(apple_available=True, nvidia_available=False)
    try:
        yield processor, document["software"][2], config
    finally:
        processor.close()


def namespace(processor: CustomProcessor, item: dict[str, Any]) -> dict[str, Any]:
    """
    Inspect experiment results without relying on runtime-generated module names.
    """
    callback = processor.packages[item["id"]].process
    assert isinstance(callback, FunctionType)
    return callback.__globals__


def test_highest_confidence_head_rgb_normalization_angles_and_labels(
    setup_node: tuple[Any, ...],
) -> None:
    """
    Select the smaller, higher-score head while preserving pixels and expected angles.
    """
    processor, item, _ = setup_node
    image = np.full((80, 100, 3), (20, 80, 180), dtype=np.float32)
    original = image.copy()
    output = processor.apply(image, item)
    assert output.shape == image.shape and output.dtype == np.float32
    np.testing.assert_array_equal(image, original)
    assert np.any(output != image)
    detector, pose = FakeSession.instances
    assert len(pose.tensors) == 1
    expected_rgb = np.array([180, 80, 20], dtype=np.float32)
    expected_detector = (expected_rgb / 255 - [.485, .456, .406]) / [.229, .224, .225]
    np.testing.assert_allclose(detector.tensors[0][0, :, 0, 0], expected_detector, atol=1e-6)
    np.testing.assert_allclose(pose.tensors[0][0, :, 32, 32], expected_rgb / 127.5 - 1,
                               atol=1e-6)
    assert pose.tensors[0].flags.c_contiguous
    results = namespace(processor, item)["last_poses"]
    assert results[0].angles == (90, 0, 0)
    assert results[0].kappa == (20, 30, 40)
    np.testing.assert_allclose(results[0].head.box, (70, 8, 90, 24), atol=1e-5)
    assert results[0].head.score == pytest.approx(.95)
    assert processor.labels[0].text == "y90.0 p+0.0 r+0.0"


@pytest.mark.parametrize("rotation", ["0", "90", "180", "270"])
def test_rotation_restores_geometry_and_label_anchors(
    setup_node: tuple[Any, ...], rotation: str,
) -> None:
    """
    Input-only rotations leave source pixels unchanged when drawing is disabled.
    """
    processor, item, config = setup_node
    config.update(input_rotation=rotation, show_boxes=False, show_axes=False)
    item["params"]["config"] = json.dumps(config)
    image = np.arange(80 * 100 * 3, dtype=np.float32).reshape(80, 100, 3) % 256
    output = processor.apply(image, item)
    np.testing.assert_array_equal(output, image)
    label = processor.labels[0]
    # Independent pixel-coordinate mapping for the fake normalized box's upper-left corner.
    anchors = {"0": (70, 2), "90": (95, 56), "180": (29, 77), "270": (4, 23)}
    x, y = anchors[rotation]
    assert label.x == pytest.approx(x / 99)
    assert label.y == pytest.approx(y / 79)


def test_crop_margin_border_shift_and_padding(setup_node: tuple[Any, ...]) -> None:
    """
    Verify square dimensions and border context independently of the crop implementation.
    """
    processor, item, _ = setup_node
    processor.apply(np.zeros((80, 100, 3), np.float32), item)
    module = namespace(processor, item)["Estimator"].estimate.__globals__["crop_head"]
    head_type = module.__globals__["Head"]
    image = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
    image = np.repeat(image[:, :, None], 3, axis=2)
    crop = module(image, head_type((20, 20, 60, 60), .9), 44)
    np.testing.assert_allclose(crop, image[18:62, 18:62], atol=.01)
    border = module(image, head_type((0, 0, 40, 40), .9), 44)
    np.testing.assert_allclose(border, image[:44, :44], atol=.01)
    small = np.full((10, 20, 3), 100, np.float32)
    padded = module(small, head_type((0, 0, 19, 9), .9), 20)
    assert np.all(padded[:5] == 0) and np.all(padded[5:15] == 100)


@pytest.mark.parametrize("angles, pixel, color", [
    ((0, 0, 0), (40, 52), (0, 0, 255)),  # Frontal cheek axis points right.
    ((0, 0, 0), (52, 40), (0, 255, 0)),  # Chin axis points down.
    ((90, 0, 0), (40, 28), (255, 0, 0)),  # Positive yaw nose axis points left.
    ((0, 0, 90), (28, 40), (0, 0, 255)),  # Positive roll rotates cheek axis CCW.
    ((0, 90, 0), (28, 40), (255, 0, 0)),  # Positive pitch nose axis points up.
])
def test_axis_projection_conventions(
    setup_node: tuple[Any, ...], angles: tuple[int, int, int],
    pixel: tuple[int, int], color: tuple[int, int, int],
) -> None:
    """
    Check the anatomical axis signs at independent, known ninety-degree orientations.
    """
    processor, item, _ = setup_node
    processor.apply(np.zeros((80, 100, 3), np.float32), item)
    draw = namespace(processor, item)["draw_axes"]
    pose_type = draw.__globals__["Pose"]
    head_type = draw.__globals__["Head"]
    canvas = np.zeros((80, 80, 3), np.float32)
    draw(canvas, pose_type(head_type((20, 20, 60, 60), .9), angles, (1, 1, 1)))
    np.testing.assert_array_equal(canvas[pixel], color)


def test_cache_device_change_config_updates_and_node_separation(setup_node: tuple[Any, ...]) -> None:
    """
    Reuse sessions for display edits and rebuild only the affected model/device session.
    """
    processor, item, config = setup_node
    image = np.zeros((80, 100, 3), np.float32)
    processor.apply(image, item)
    config.update(max_heads=2, show_boxes=False)
    item["params"]["config"] = json.dumps(config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 2
    assert len(namespace(processor, item)["last_poses"]) == 2
    assert [pose.head.score for pose in namespace(processor, item)["last_poses"]] == (
        pytest.approx([.95, .9])
    )
    config["pose_device"]["backend"] = "coreml"
    item["params"]["config"] = json.dumps(config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 3
    assert "CoreMLExecutionProvider" in FakeSession.instances[-1].providers
    config["pose_device"]["backend"] = "cuda"
    item["params"]["config"] = json.dumps(config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 3  # Unavailable CUDA resolves to the existing Apple session.
    another = node("software", "custom", **item["params"])
    processor.apply(image, another)
    assert len(FakeSession.instances) == 5
    Path(config["pose_path"]).write_bytes(b"replaced model boundary")
    processor.apply(image, item)
    assert len(FakeSession.instances) == 6


@pytest.mark.parametrize("failure", ["schema", "unprepared", "boxes", "pose", "zero"])
def test_unsupported_models_and_malformed_predictions(
    setup_node: tuple[Any, ...], failure: str,
) -> None:
    """
    Report unsafe model contracts and inference output failures through pipeline errors.
    """
    processor, item, _ = setup_node
    FakeSession.bad = failure
    with pytest.raises(ValueError):
        processor.apply(np.zeros((80, 100, 3), np.float32), item)


@pytest.mark.parametrize("changes", [
    {"max_heads": True}, {"score_threshold": None}, {"input_rotation": 90},
    {"show_axes": 1}, {"pose_path": []}, {"detector_path": []}, {"detector_model": "unknown"},
    {"pose_model": "unknown"}, {"pose_model": []},
])
def test_invalid_configuration(setup_node: tuple[Any, ...], changes: dict[str, Any]) -> None:
    """
    Validate user JSON before model inference.
    """
    processor, item, config = setup_node
    config.update(changes)
    item["params"]["config"] = json.dumps(config)
    with pytest.raises(ValueError):
        processor.apply(np.zeros((80, 100, 3), np.float32), item)
    assert not FakeSession.instances


def test_pending_pose_download_and_no_heads_clear_old_results(
    setup_node: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Pass through pending models and remove old predictions on empty detection frames.
    """
    processor, item, config = setup_node
    image = np.zeros((80, 100, 3), np.float32)
    processor.apply(image, item)
    config["pose_path"] = ""
    monkeypatch.setattr("based_thermals.custom_nodes.models.ModelResolver.resolve",
                        lambda _self, _source: None)
    item["params"]["config"] = json.dumps(config)
    processor.labels.clear()  # The branch clears its collector at each frame boundary.
    np.testing.assert_array_equal(processor.apply(image, item), image)
    assert not namespace(processor, item)["last_poses"] and not processor.labels
    config["pose_path"] = str(Path(config["detector_path"]).with_name("pose.onnx"))
    config["score_threshold"] = 1
    item["params"]["config"] = json.dumps(config)
    pose_calls = len(FakeSession.instances[1].tensors)
    processor.labels.clear()
    np.testing.assert_array_equal(processor.apply(image, item), image)
    assert len(FakeSession.instances[1].tensors) == pose_calls
    assert not namespace(processor, item)["last_poses"] and not processor.labels


def test_detector_choice_round_trip_and_pending_download(
    setup_node: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Publish the detector choices through JSON and request both models without UI side effects.
    """
    processor, item, config = setup_node
    controls = json.loads(item["params"]["controls"])
    choice = next(control for control in controls if control["key"] == "detector_model")
    assert choice["type"] == "choice"
    assert choice["options"] == ["HGNet-N (fast)", "DINOv3-S (reference)"]
    config.update(detector_path="", pose_path="", detector_model="DINOv3-S (reference)")
    item["params"]["config"] = json.dumps(config)
    sources = []

    def pending(_resolver: Any, source: dict[str, Any]) -> None:
        """
        Record requests at the download boundary without contacting the network.
        """
        sources.append(source)

    monkeypatch.setattr("based_thermals.custom_nodes.models.ModelResolver.resolve", pending)
    image = np.zeros((80, 100, 3), np.float32)
    np.testing.assert_array_equal(processor.apply(image, item), image)
    assert len(sources) == 2 and not FakeSession.instances
    assert sources[0]["sha256"] == (
        "9b64e68cd6cf2e085613ae1dd06a42941fc0b7fce57cd80441dfb45ce90a1323"
    )
    config["detector_model"] = "HGNet-N (fast)"
    item["params"]["config"] = json.dumps(config)
    processor.apply(image, item)
    assert sources[2]["sha256"] == (
        "155c3c6f37dcf398288735f96a18738b67eb8f7d02ec54f024c75bed0a169e31"
    )


def test_downloaded_detector_runs_and_choice_rebuilds_only_detector(
    setup_node: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Start inference when downloads finish, and keep the pose session across detector changes.
    """
    processor, item, config = setup_node
    hgnet = Path(config["detector_path"])
    dinov3 = tmp_path / "dinov3-detector.onnx"
    dinov3.write_bytes(b"second fake detector boundary")
    pose = Path(config["pose_path"])
    config.update(detector_path="", pose_path="")
    item["params"]["config"] = json.dumps(config)

    def resolve(_resolver: Any, source: dict[str, Any]) -> Path:
        """
        Supply completed external downloads through the resolver boundary.
        """
        if "archive_member" in source:
            return pose
        return dinov3 if "DINOv3" in source["name"] else hgnet

    monkeypatch.setattr("based_thermals.custom_nodes.models.ModelResolver.resolve", resolve)
    image = np.full((80, 100, 3), 100, np.float32)
    output = processor.apply(image, item)
    assert np.any(output != image) and len(FakeSession.instances) == 2
    config["detector_model"] = "DINOv3-S (reference)"
    item["params"]["config"] = json.dumps(config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 3
    assert len(FakeSession.instances[1].tensors) == 2


@pytest.mark.parametrize("choice, digest, filename", [
    ("ViT-T/16 128×128 (v11)",
     "02136d0746d0acad26c0bea6d405826d028c780d0ec6a81d43b93e33004946a7",
     "hfhpe_vitt_1x3x128x128.onnx"),
    ("ViT-T/16 128×128 (rollgate v11)",
     "0f4311d7cf8560d2c04a2d40e7f166a8195ad50107d0a6b64336fc8fad517db8",
     "hfhpe_vitt_rollgate_1x3x128x128.onnx"),
])
def test_pose_choice_download_crop_size_and_independent_session_change(
    setup_node: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    choice: str, digest: str, filename: str,
) -> None:
    """
    Switch from 64 to each pinned 128 export without rebuilding the detector session.
    """
    processor, item, config = setup_node
    image = np.full((80, 100, 3), (20, 80, 180), np.float32)
    processor.apply(image, item)
    downloaded = tmp_path / filename
    downloaded.write_bytes(b"fake 128 pose boundary")
    requests = []

    def resolve(_resolver: Any, source: dict[str, Any]) -> Path | None:
        """
        Simulate a pending first request and a completed next-frame download.
        """
        requests.append(source)
        return downloaded if len(requests) > 1 else None

    monkeypatch.setattr("based_thermals.custom_nodes.models.ModelResolver.resolve", resolve)
    config.update(pose_path="", pose_model=choice)
    item["params"]["config"] = json.dumps(config)
    np.testing.assert_array_equal(processor.apply(image, item), image)
    assert len(FakeSession.instances) == 2
    processor.apply(image, item)
    assert len(FakeSession.instances) == 3
    assert len(FakeSession.instances[0].tensors) == 2
    assert len(FakeSession.instances[1].tensors) == 1
    tensor = FakeSession.instances[2].tensors[0]
    assert tensor.shape == (1, 3, 128, 128)
    np.testing.assert_allclose(tensor[0, :, 64, 64],
                               np.array([180, 80, 20]) / 127.5 - 1, atol=1e-6)
    assert requests[0]["sha256"] == digest
    assert requests[0]["archive_member"].endswith("/" + filename)
    assert requests[0]["archive_sha256"] == (
        "74a50a5c1cc6e85c2393bd61bf3d96f30a6049f398081f3d00b36b8640c17b31"
    )
    controls = json.loads(item["params"]["controls"])
    selector = next(control for control in controls if control["key"] == "pose_model")
    assert choice in selector["options"]


def test_raw_detector_requires_exact_audited_checksum(
    setup_node: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Accept release bytes without preparation metadata only through an exact checksum match.
    """
    import hashlib

    processor, item, config = setup_node
    image = np.full((80, 100, 3), (20, 80, 180), np.float32)
    processor.apply(image, item)
    detector_type = namespace(processor, item)["Detector"]
    known = detector_type.__init__.__globals__["NORMALIZATIONS"]
    path = Path(config["detector_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(known, digest, "div255")
    FakeSession.bad = "unprepared"
    detector = detector_type(path, {"backend": "cpu", "apple_compute": "CPUOnly"})
    detector.detect(image, .7, 1)
    np.testing.assert_allclose(FakeSession.instances[-1].tensors[0][0, :, 0, 0],
                               np.array([180, 80, 20]) / 255, atol=1e-6)
    path.write_bytes(b"changed raw graph")
    with pytest.raises(ValueError, match="pinned detector release"):
        detector_type(path, {"backend": "cpu", "apple_compute": "CPUOnly"})
