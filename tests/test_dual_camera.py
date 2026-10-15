"""
Offline dual-stream acquisition, native radiometry and independently owned camera settings.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from support.qt_process import run_popup

from based_thermals.camera import CameraError
from based_thermals.camera_backends import create_camera
from based_thermals.camera_backends.virtual import VirtualDualCamera
from based_thermals.camera_backends.visible import (
    VisibleFrame,
    visible_image,
    visible_mapping_valid,
)
from based_thermals.desktop_app.dual_camera import apply_camera_hardware
from based_thermals.pipeline import default_pipeline, node, validate_pipeline
from based_thermals.pipeline_hardware import PipelineHardware
from based_thermals.pipeline_model.sources import source_required
from based_thermals.pipeline_processing import PipelineProcessor
from based_thermals.render import ThermalRenderer


def dual_document() -> dict:
    """
    Construct two active hardware stacks using the same control name with different values.
    """
    document = default_pipeline()
    document["hardware"] = [node("hardware", "device_control", control="brightness", value=70)]
    document["visible_hardware"] = [
        node("visible_hardware", "device_control", control="brightness", value=20)
    ]
    document["software"].insert(1, node("software", "combine", tab="visible"))
    return document


def test_virtual_backend_lifecycle_and_independent_stream_rates() -> None:
    """
    Exercise actual paced acquisition without USB and permit reopen after cancellation.
    """
    camera = create_camera("virtual-dual")
    assert isinstance(camera, VirtualDualCamera) and camera.profile.experimental
    frame = camera.make_frame(1)
    assert frame.raw_counts is not None and frame.preview is not None
    assert frame.visible is not None and frame.visible.bgr is not None
    assert frame.raw_counts.shape == (60, 80)
    assert frame.preview.shape == (120, 160)
    assert frame.visible.bgr.shape == (240, 320, 3)
    assert frame.sequence == 1 and frame.visible.sequence == 0
    assert frame.timestamp != frame.visible.timestamp
    assert visible_mapping_valid(frame)
    with pytest.raises(CameraError, match="Open"):
        next(camera.frames())
    camera.open()
    stream = camera.frames()
    assert next(stream).sequence == 0
    camera.stop_stream()
    assert list(stream) == []
    camera.close()
    camera.open()
    assert next(camera.frames()).sequence == 0
    camera.close()


def test_visible_pixels_are_owned_and_faults_are_independent() -> None:
    """
    Black optical pixels remain valid, while missing/stale optical data fails explicitly.
    """
    camera = VirtualDualCamera()
    frame = camera.make_frame(10)
    pixels = np.zeros((240, 320, 3), np.uint8)
    assert frame.visible is not None
    stream = replace(frame.visible, bgr=pixels)
    assert stream.bgr is not None
    pixels[:] = 255
    assert not stream.bgr.any() and not stream.bgr.flags.writeable
    assert not visible_image(replace(frame, visible=stream)).any()
    for faulty in [
        camera.make_frame(10, visible_valid=False),
        replace(frame, visible=replace(stream, timestamp=-1)),
    ]:
        with pytest.raises(ValueError, match="unavailable|stale"):
            visible_image(faulty)
        renderer = ThermalRenderer(scale=1)
        assert renderer.render_detailed(faulty).measurements_valid
        renderer.pipeline_processor.close()
    with pytest.raises(ValueError, match="homography"):
        VisibleFrame(pixels, to_thermal=(0,) * 9, registration_source="test")
    assert not visible_mapping_valid(camera.make_frame(10, registered=False))
    assert not visible_mapping_valid(replace(frame, visible=replace(stream, timestamp=None)))
    assert not visible_mapping_valid(
        replace(frame, visible=replace(stream, to_thermal=(0.25, 0, 20, 0, 0.25, 0, 0, 0, 1)))
    )


def test_optical_processing_holds_thermal_without_holding_optical() -> None:
    """
    FFC retains native readings but processes fresh optical pixels and records both clocks.
    """
    camera = VirtualDualCamera()
    renderer = ThermalRenderer(scale=1, image_source="visible")
    try:
        first = renderer.render_detailed(camera.make_frame(0))
        held = renderer.render_detailed(camera.make_frame(25, thermal_valid=False))
        np.testing.assert_array_equal(first.temperatures_celsius, held.temperatures_celsius)
        assert not held.measurements_valid and not held.measurement_mapping_valid
        assert not np.array_equal(first.image, held.image)
        assert held.streams["thermal"]["sequence"] == 0
        assert held.streams["visible"]["sequence"] == 15
        unregistered = renderer.render_detailed(camera.make_frame(26, registered=False))
        assert unregistered.measurements_valid and not unregistered.measurement_mapping_valid
    finally:
        renderer.pipeline_processor.close()


def test_sources_blends_masks_and_round_trip() -> None:
    """
    Optical branches participate in scheduling without activating disconnected hardware.
    """
    document = dual_document()
    assert validate_pipeline(document, hardware_profile="virtual-dual") == document
    assert source_required(document, "preview") and source_required(document, "visible")
    document["branches"]["B"] = [node("software", "source", source="visible")]
    document["software"][1]["params"].update(tab="B", mask_source="visible")
    processor = PipelineProcessor(apple_available=False, nvidia_available=False)
    try:
        image, source = processor.process(
            VirtualDualCamera().make_frame(0), None, document, scale=1
        )
        assert image.shape == (60, 80, 3) and source == "preview"
        document["software"][1]["bypass"] = True
        assert not source_required(document, "visible")
        document["software"][0]["params"]["source"] = "visible"
        document["software"].insert(1, node("software", "range"))
        with pytest.raises(ValueError, match="requires a thermal source"):
            validate_pipeline(document, hardware_profile="virtual-dual")
    finally:
        processor.close()


def test_hardware_settings_do_not_cross_camera_owners() -> None:
    """
    Apply same-name settings, bypass one stream, and restore both independent baselines.
    """
    camera = VirtualDualCamera()
    camera._brightness.update(thermal=63, visible=27)
    thermal, optical = camera.create_controls(), camera.create_visible_controls()
    session = SimpleNamespace(
        pipeline_hardware=PipelineHardware(thermal),
        visible_pipeline_hardware=PipelineHardware(optical, source="visible"),
        api=SimpleNamespace(CameraError=CameraError),
    )
    document = dual_document()
    original = camera.make_frame(0)
    apply_camera_hardware(session, document)
    assert thermal.values["brightness"] == 70 and optical.values["brightness"] == 20
    changed = camera.make_frame(0)
    np.testing.assert_array_equal(original.raw_counts, changed.raw_counts)
    assert not np.array_equal(original.preview, changed.preview)
    assert original.visible is not None and changed.visible is not None
    assert not np.array_equal(original.visible.bgr, changed.visible.bgr)
    candidate = deepcopy(document)
    candidate["visible_hardware"][0]["bypass"] = True
    apply_camera_hardware(session, candidate, previous_document=document)
    assert thermal.values["brightness"] == 70 and optical.values["brightness"] == 50
    bad = deepcopy(document)
    bad["hardware"][0]["params"]["value"] = 80
    bad["visible_hardware"][0]["params"]["value"] = 200
    with pytest.raises(ValueError, match="brightness"):
        apply_camera_hardware(session, bad, previous_document=candidate)
    assert thermal.values["brightness"] == 70 and optical.values["brightness"] == 50
    thermal.restore()
    optical.restore()
    assert thermal.values["brightness"] == 63 and optical.values["brightness"] == 27


def test_dual_camera_editor_hides_secondary_stack_and_uses_own_metadata(tmp_path: Path) -> None:
    """
    Exercise the real isolated Qt editor, persisted stacks and per-camera control pickers.
    """
    result = run_popup(
        """
from PySide6.QtWidgets import QApplication
from based_thermals.pipeline_ui.editor import PipelineEditor
from based_thermals.view_window import ControlRow
from based_thermals.camera_backends.virtual import VirtualDualCamera
app = QApplication([])
editor = PipelineEditor(lambda value: None, ControlRow)
assert editor.stacks["visible_hardware"].isHidden()
camera = VirtualDualCamera()
a, b = camera.create_controls(), camera.create_visible_controls()
a.load(); b.load()
state = {"camera_profile": camera.profile.as_dict(), "camera_capabilities": a.capabilities(),
         "visible_camera_capabilities": b.capabilities(), "hardware": a.state(),
         "visible_hardware": b.state()}
editor.update_state(state, False)
assert not editor.stacks["visible_hardware"].isHidden()
editor.insert("visible_hardware", "device_control", 0)
item = editor.nodes("visible_hardware")[0]
fields = editor.hardware_widgets[item["id"]]
fields.select("brightness")
assert item["params"] == {"control": "brightness", "value": 50}
assert not editor.nodes("hardware")
state["camera_profile"]["visible"] = False
editor.update_state(state, False)
assert editor.stacks["visible_hardware"].isHidden()
assert editor.nodes("visible_hardware")[0]["params"]["control"] == "brightness"
print("dual UI passed")
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dual_hardware_failed_first_apply_restores_baseline() -> None:
    """
    A rejected optical setting cannot leave a startup thermal override applied.
    """
    camera = VirtualDualCamera()
    thermal, optical = camera.create_controls(), camera.create_visible_controls()
    session = SimpleNamespace(
        pipeline_hardware=PipelineHardware(thermal),
        visible_pipeline_hardware=PipelineHardware(optical, source="visible"),
        api=SimpleNamespace(CameraError=CameraError),
    )
    document = dual_document()
    document["visible_hardware"][0]["params"]["value"] = 200
    with pytest.raises(ValueError, match="brightness"):
        apply_camera_hardware(session, document)
    assert thermal.values["brightness"] == optical.values["brightness"] == 50
    assert not thermal.enabled and not optical.enabled


def test_optional_optical_stack_round_trips_and_rejects_duplicate_controls(tmp_path: Path) -> None:
    """
    Preserve camera identities in exported bundles and reject duplicates within an owner.
    """
    from based_thermals.pipeline_ui.transfer.documents import read_pipelines, write_pipelines

    document = dual_document()
    path = tmp_path / "dual.pipeline.json"
    write_pipelines(path, {"Dual": document}, single=False)
    assert read_pipelines(path)[0].document == document
    duplicate = node("visible_hardware", "device_control", control="brightness", value=30)
    document["visible_hardware"].append(duplicate)
    with pytest.raises(ValueError, match="only be included once"):
        validate_pipeline(document, hardware_profile="virtual-dual")


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_registered_optical_landmarks_follow_view_rotation(rotation: int) -> None:
    """
    Check an independently constructed repeated-pixel image against exact thermal-grid landmarks.
    """
    frame = VirtualDualCamera().make_frame(0)
    native = np.zeros((60, 80, 3), np.uint8)
    native[:30, :40] = (10, 50, 100)
    native[30:, 40:] = (150, 200, 250)
    optical = np.repeat(np.repeat(native, 4, axis=0), 4, axis=1)
    assert frame.visible is not None
    frame = replace(frame, visible=replace(frame.visible, bgr=optical))
    renderer = ThermalRenderer(scale=1, rotation=rotation, image_source="visible")
    try:
        rendered = renderer.render_detailed(frame)
        np.testing.assert_array_equal(rendered.image, np.rot90(native, -(rotation // 90)))
        assert rendered.measurement_mapping_valid
        assert rendered.raw_counts is not None
        assert rendered.raw_counts.shape == rendered.temperatures_celsius.shape
    finally:
        renderer.pipeline_processor.close()


# Reuse the established desktop dependency boundary; no windows or devices are opened.

desktop_dependencies = viewer_fixture


def test_virtual_desktop_initializes_both_hardware_owners(viewer: object) -> None:
    """
    Exercise the public desktop constructor and startup settings path with the virtual backend.
    """
    del viewer
    from based_thermals import desktop
    from based_thermals.desktop_app.session import DesktopSession

    session = DesktopSession(desktop, ["--camera", "virtual-dual", "--image-source", "visible"])
    try:
        assert session.initialize_hardware()
        assert session.visible_hardware is not session.hardware
        session.set_pipeline(dual_document())
        assert session.hardware.state()["brightness"]["value"] == 70
        assert session.visible_hardware.state()["brightness"]["value"] == 20
        assert isinstance(session.camera, VirtualDualCamera)
        session.rendered = session.renderer.render_detailed(session.camera.make_frame(1))
        state = session.view_state()
        assert state["camera_profile"]["visible"]
        assert state["visible_hardware"]["brightness"]["value"] == 20
        assert "untested" in state["status"].lower()
    finally:
        session.hardware.restore()
        session.visible_hardware.restore()
        session.renderer.pipeline_processor.close()
