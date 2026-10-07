"""Connected pipeline tabs: numerical blends, frame coherence and lazy threads."""

from copy import deepcopy
from threading import Barrier, current_thread

import numpy as np
import pytest
from test_render import frame_with_preview

from topdon_duo.camera import decode_duo_frame
from topdon_duo.pipeline import (
    default_pipeline,
    execution_dependencies,
    node,
    preview_required,
    validate_pipeline,
)
from topdon_duo.pipeline_hardware import desired_hardware
from topdon_duo.pipeline_processing import PipelineProcessor, combine_images


def connect(document, target, owner="A", **params):
    nodes = document["software"] if owner == "A" else document["branches"][owner]
    nodes.insert(
        -1 if owner == "A" else len(nodes), node("software", "combine", tab=target, **params)
    )


def test_migration_preserves_single_pipeline_and_all_tabs_persist(tmp_path, monkeypatch):
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = default_pipeline()
    legacy = {"version": 2, "hardware": [], "software": deepcopy(document["software"][:-1])}
    snapshot = deepcopy(legacy)
    migrated = validate_pipeline(legacy)
    assert legacy == snapshot
    assert migrated["software"][:-1] == legacy["software"]
    assert migrated["software"][-1]["type"] == "output"
    assert all(
        len(nodes) == 1 and nodes[0]["type"] == "source" for nodes in migrated["branches"].values()
    )
    migrated["branches"]["B"].append(node("software", "gamma", amount=1.8))
    connect(migrated, "B", mode="screen", opacity=0.25)
    connect(migrated, "D", "B", mode="difference", opacity=1.0)
    save_settings({"pipeline": migrated})
    assert load_settings()["pipeline"] == migrated
    assert validate_pipeline(migrated) == migrated


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["software"].pop(),
        lambda d: d["software"][-1].update(bypass=True),
        lambda d: d["software"].insert(1, d["software"].pop()),
        lambda d: d["branches"]["B"].append(node("software", "output")),
        lambda d: d["branches"]["B"].append(node("software", "source")),
        lambda d: d["branches"]["B"][0].update(bypass=True),
        lambda d: d["branches"].pop("C"),
        lambda d: connect(d, "A"),
        lambda d: (connect(d, "C", "B"), connect(d, "B", "C")),
        lambda d: (connect(d, "B"), connect(d, "C", "B"), connect(d, "A", "C")),
        lambda d: connect(d, "Z"),
        lambda d: connect(d, "B", opacity=float("inf")),
        lambda d: d["branches"]["D"][0].update(id=d["software"][0]["id"]),
    ],
)
def test_invalid_connections_and_pinned_endpoints_rejected(mutation):
    document = default_pipeline()
    mutation(document)
    with pytest.raises(ValueError):
        validate_pipeline(document)


def test_bypassed_connection_does_not_activate_a_branch():
    document = default_pipeline()
    connect(document, "B")
    connect(document, "C", "B")
    assert execution_dependencies(document) == {"C": set(), "B": {"C"}, "A": {"B"}}
    document["software"][-2]["bypass"] = True
    assert execution_dependencies(validate_pipeline(document)) == {"A": set()}


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("opacity", 200),
        ("weighted", 255),
        ("add", 255),
        ("subtract", 0),
        ("difference", 140),
        ("multiply", 60 * 200 / 255),
        ("screen", 255 - (255 - 60) * (255 - 200) / 255),
        ("overlay", 2 * 60 * 200 / 255),
        ("lighten", 200),
        ("darken", 60),
        ("and", 60 & 200),
        ("or", 60 | 200),
        ("xor", 60 ^ 200),
        ("mask", 60),
    ],
)
def test_combine_modes_and_opacity(mode, expected):
    base = np.full((4, 6, 3), 60, np.float32)
    incoming = np.full((4, 6, 3), 200, np.float32)
    p = node("software", "combine", mode=mode, opacity=1.0)["params"]
    assert np.allclose(combine_images(base, incoming, p), expected, atol=1e-4)
    p["opacity"] = 0.25
    assert np.allclose(combine_images(base, incoming, p), 60 * 0.75 + expected * 0.25, atol=1e-4)
    p["opacity"] = 0
    assert np.array_equal(combine_images(base, incoming, p), base)
    assert np.all(base == 60) and np.all(incoming == 200)


def test_weighted_mask_resize_and_overlay_upper_branch():
    base = np.full((4, 6, 3), 180, np.float32)
    incoming = np.full((2, 3, 3), 200, np.float32)
    p = node(
        "software",
        "combine",
        mode="weighted",
        opacity=1.0,
        base_weight=0.2,
        input_weight=0.5,
        offset=-20,
    )["params"]
    assert np.allclose(combine_images(base, incoming, p), 116)
    p.update(mode="mask", invert=True)
    assert np.all(combine_images(base, incoming, p) == 0)
    p.update(invert=False, threshold=210)
    assert np.all(combine_images(base, incoming, p) == 0)
    p["mode"] = "overlay"
    assert np.allclose(combine_images(base, incoming, p), 255 - 2 * 75 * 55 / 255)


def test_connected_branches_run_once_on_separate_threads_using_same_frame(monkeypatch):
    processor = PipelineProcessor()
    document = default_pipeline()
    connect(document, "B")
    connect(document, "C")
    connect(document, "D", "B")
    connect(document, "D", "C")
    tabs = {"A": document["software"], **document["branches"]}
    identity = {nodes[0]["id"]: tab for tab, nodes in tabs.items()}
    calls = []
    barrier = Barrier(2)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    averaged = raw.astype(np.float32)

    def fake(self, incoming_frame, average, branch, palette, inputs):
        tab = identity[branch["software"][0]["id"]]
        assert incoming_frame is frame and average is averaged
        calls.append((tab, current_thread().name, tuple(inputs)))
        if tab in "BC":
            barrier.wait(timeout=3)  # Both must be running concurrently.
        return np.full((4, 6, 3), 100, np.float32), "preview"

    monkeypatch.setattr(PipelineProcessor, "_process_single", fake)
    try:
        image, _ = processor.process(frame, averaged, document, scale=1)
        assert image.shape == (192, 256, 3)
        assert sorted(tab for tab, *_ in calls) == list("ABCD")
        assert len({thread for _, thread, _ in calls}) == 4
        assert {tab: set(inputs) for tab, _, inputs in calls} == {
            "A": {"B", "C"},
            "B": {"D"},
            "C": {"D"},
            "D": set(),
        }
        calls.clear()
        for item in document["software"]:
            if item["type"] == "combine":
                item["bypass"] = True
        processor.process(frame, averaged, document)
        assert [tab for tab, *_ in calls] == ["A"]
        assert set(processor.executors) == {"A"} and set(processor.branches) == {"A"}
    finally:
        processor.close()


def test_combine_occurs_in_node_order_and_raw_readings_unchanged():
    processor = PipelineProcessor()
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    snapshot = raw.copy()
    document = default_pipeline()
    document["software"] = [document["software"][0], document["software"][-1]]
    document["software"][0]["params"]["source"] = "raw"
    document["branches"]["B"][0]["params"]["source"] = "raw"
    document["branches"]["B"].append(node("software", "brightness", amount=20.0))
    connect(document, "B", opacity=1.0)
    document["software"].insert(-1, node("software", "gamma", amount=2.0))
    try:
        first, _ = processor.process(frame, None, document, scale=1)
        document["software"][1:-1] = document["software"][1:-1][::-1]
        second, _ = processor.process(frame, None, document, scale=1)
        assert not np.array_equal(first, second)
        assert np.array_equal(raw, snapshot)
    finally:
        processor.close()


def test_hardware_remains_active_if_connected_tab_uses_preview():
    document = default_pipeline()
    document["hardware"].append(node("hardware", "brightness", value=70))
    document["software"][0]["params"]["source"] = "raw"
    assert not preview_required(document)
    assert desired_hardware(document)[0] == {}
    connect(document, "B")
    assert preview_required(document)
    assert desired_hardware(document)[0] == {"brightness": 70}
    document["branches"]["B"].append(node("software", "range"))
    assert not preview_required(document)


@pytest.mark.parametrize(
    "mode", ("opacity", "add", "screen", "multiply", "difference", "xor", "weighted")
)
def test_optional_soft_and_binary_masks_limit_each_blend(mode):
    base = np.full((2, 3, 3), 60, np.float32)
    incoming = np.full((2, 3, 3), 200, np.float32)
    mask = np.tile(np.array([0, 127.5, 255], np.float32), (2, 1))
    p = node("software", "combine", mode=mode, opacity=0.6)["params"]
    unmasked = combine_images(base, incoming, p)
    soft = combine_images(base, incoming, p, mask)
    assert np.array_equal(soft[:, 0], base[:, 0])
    assert np.allclose(soft[:, 1], (base[:, 1] + unmasked[:, 1]) / 2)
    assert np.allclose(soft[:, 2], unmasked[:, 2])
    p["mask_invert"] = True
    inverted = combine_images(base, incoming, p, mask)
    assert np.allclose(inverted[:, 0], unmasked[:, 0])
    assert np.array_equal(inverted[:, 2], base[:, 2])
    p.update(mask_kind="threshold", mask_threshold=128, mask_invert=False)
    binary = combine_images(base, incoming, p, mask)
    assert np.array_equal(binary[:, :2], base[:, :2])
    assert np.allclose(binary[:, 2], unmasked[:, 2])


def test_mask_only_tab_activates_and_cycles_are_rejected():
    document = default_pipeline()
    connect(document, "raw", mask_source="C")
    connect(document, "preview", "C", mask_source="D")
    assert execution_dependencies(validate_pipeline(document)) == {
        "D": set(),
        "C": {"D"},
        "A": {"C"},
    }
    connect(document, "raw", "D", mask_source="A")
    with pytest.raises(ValueError, match="cycle"):
        validate_pipeline(document)


@pytest.mark.parametrize(
    "blend_input,mask_source",
    [
        ("preview", "none"),
        ("raw", "none"),
        ("B", "preview"),
        ("B", "raw"),
        ("raw", "B"),
        ("B", "input"),
    ],
)
def test_camera_raw_and_pipeline_inputs_and_masks_use_the_current_frame(blend_input, mask_source):
    frame, _ = frame_with_preview()
    _, raw, preview = decode_duo_frame(frame)
    before = raw.copy()
    document = default_pipeline()
    document["software"] = [document["software"][0], document["software"][-1]]
    document["software"][0]["params"]["source"] = "raw"
    document["branches"]["B"][0]["params"]["source"] = "raw"
    document["branches"]["B"].append(node("software", "brightness", amount=15.0))
    connect(document, blend_input, mask_source=mask_source, raw_low=10, raw_high=60)
    processor = PipelineProcessor()
    try:
        image, _ = processor.process(frame, raw.astype(np.float32), document, scale=1)
        assert image.shape == (192, 256, 3) and image.dtype == np.uint8
        assert set(processor.executors) == (
            {"A", "B"} if "B" in (blend_input, mask_source) else {"A"}
        )
        assert np.array_equal(raw, before) and np.any(preview)
    finally:
        processor.close()


def test_preview_mask_activates_hardware_and_raw_mask_bounds_are_validated():
    document = default_pipeline()
    document["software"][0]["params"]["source"] = "raw"
    document["hardware"].append(node("hardware", "contrast", value=70))
    connect(document, "raw", mask_source="preview")
    assert preview_required(document)
    assert desired_hardware(document)[0] == {"contrast": 70}
    document["software"][-2]["params"]["raw_high"] = 10
    with pytest.raises(ValueError, match="From"):
        validate_pipeline(document)


def test_failed_branch_drains_frame_and_next_revision_recovers(monkeypatch):
    document = default_pipeline()
    connect(document, "B")
    connect(document, "C")
    original = PipelineProcessor._process_single
    b_id = document["branches"]["B"][0]["id"]
    failed = False

    def intermittent(self, *args):
        nonlocal failed
        if args[2]["software"][0]["id"] == b_id and not failed:
            failed = True
            raise ValueError("test branch failure")
        return original(self, *args)

    monkeypatch.setattr(PipelineProcessor, "_process_single", intermittent)
    processor = PipelineProcessor()
    frame, _ = frame_with_preview()
    try:
        with pytest.raises(ValueError, match="test branch failure"):
            processor.process(frame, None, document)
        image, _ = processor.process(frame, None, document)
        assert image.shape == (576, 768, 3)
    finally:
        processor.close()
