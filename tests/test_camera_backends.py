"""
Different camera geometry, radiometry, capabilities and settings without live hardware.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from support.desktop_recording import viewer_fixture
from support.frames import frame_with_preview

from topdon_duo import camera_backends, desktop
from topdon_duo.camera import CameraError, decode_duo_frame
from topdon_duo.camera_backends import CameraFrame, CameraProfile, ControlSpec, ReportedReading
from topdon_duo.camera_backends.controls import DeviceControls
from topdon_duo.desktop_app.session import DesktopSession
from topdon_duo.frame_pump import CameraFramePump
from topdon_duo.pipeline import default_pipeline, node, validate_pipeline
from topdon_duo.pipeline_hardware import PipelineHardware
from topdon_duo.pipeline_processing import PipelineProcessor
from topdon_duo.pipeline_ui.transfer.documents import read_pipelines, write_pipelines
from topdon_duo.render import ThermalRenderer
from topdon_duo.settings_preferences import load_settings, save_settings

viewer = viewer_fixture
PROFILE = CameraProfile("test-sensor", "Test camera", (6, 4), 9, calibration_key="lens-A")


def convert(counts: np.ndarray, **_kwargs: Any) -> np.ndarray:
    """
    Independently define the fake camera's encoding, which differs from Duo.
    """
    return counts.astype(np.float32) * 0.5 - 10


def hardware() -> tuple[DeviceControls, dict[tuple[str, str | None], Any], Mock]:
    """
    Model an atomic device setting protocol with device and individual-spot fields.
    """
    values: dict[tuple[str, str | None], Any] = {("gain", None): 1, ("humidity", None): 40,
              ("distance", "A"): 2.0, ("distance", "B"): 3.0}
    specs = {
        "gain": ControlSpec("Gain", minimum=0, maximum=3, default=1),
        "humidity": ControlSpec("Humidity", effect="measurement", unit="%", default=40),
        "distance": ControlSpec("Distance", minimum=0.5, maximum=10, step=0.05,
                                unit="m", scope="spot", effect="measurement", default=2),
    }

    def write(name: str, value: Any, spot: str | None) -> None:
        """
        Keep unrelated settings untouched in this independent fake transport.
        """
        values[name, spot] = value

    writes = Mock(side_effect=write)
    owner = DeviceControls(PROFILE, specs, lambda name, spot: values[name, spot], writes,
                           spot_ids=("A", "B"))
    return owner, values, writes


def test_duo_decoded_bridge_preserves_native_counts_and_readings() -> None:
    """
    Compare the bridge against the existing independently covered byte decoder.
    """
    packet, _ = frame_with_preview()
    decoded = camera_backends.decode_frame(packet)
    _, counts, preview = decode_duo_frame(packet)
    np.testing.assert_array_equal(decoded.raw_counts, counts)
    np.testing.assert_array_equal(decoded.preview, preview)
    np.testing.assert_array_equal(decoded.temperatures(), counts / 64 - 50)
    assert decoded.source_bytes is packet
    assert decoded.raw_counts is not None
    assert not decoded.raw_counts.flags.writeable


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_foreign_camera_geometry_conversion_and_pipeline(rotation: int) -> None:
    """
    Retain native temperature samples while display filters operate at another size.
    """
    counts = np.arange(24, dtype=np.uint16).reshape(4, 6) + 80
    frame = CameraFrame(PROFILE, counts, converter=convert)
    renderer = ThermalRenderer(scale=2, rotation=rotation)
    renderer.camera_profile = PROFILE
    document = default_pipeline()
    document["software"][0]["params"]["source"] = "raw"
    renderer.set_pipeline(document)
    try:
        rendered = renderer.render_detailed(frame)
        expected = np.rot90(convert(counts), -(rotation // 90))
        np.testing.assert_array_equal(rendered.temperatures_celsius, expected)
        assert rendered.image.shape == (*[dimension * 2 for dimension in expected.shape], 3)
        assert rendered.measurements_valid
        np.testing.assert_array_equal(frame.raw_counts, counts)
    finally:
        renderer.pipeline_processor.close()


def test_preview_only_black_pixels_are_valid_display_without_fake_radiometry() -> None:
    """
    Show a black preview and device spot telemetry without inventing a native temperature plane.
    """
    profile = CameraProfile("preview", "Preview camera", (6, 4), 12, radiometry=False)
    reading = ReportedReading("Spot temperature", 31.5, "°C", spot_id="A")
    frame = CameraFrame(profile, preview_bgr=np.zeros((8, 12, 3), np.uint8), readings=(reading,))
    renderer = ThermalRenderer(scale=1)
    processor = PipelineProcessor()
    try:
        rendered = renderer.render_detailed(frame)
        assert not rendered.measurements_valid and not rendered.radiometry_available
        assert rendered.raw_counts is None and np.isnan(rendered.temperatures_celsius).all()
        assert rendered.reported_readings == (reading,)
        document = default_pipeline()
        document["software"] = [node("software", "source"), node("software", "output")]
        output, source = processor.process(frame, None, document, scale=1)
        assert source == "preview" and output.shape == (4, 6, 3) and not output.any()
        document["software"][0]["params"]["source"] = "raw"
        with pytest.raises(ValueError, match="radiometry is unavailable"):
            processor.process(frame, None, document)
    finally:
        processor.close()
        renderer.pipeline_processor.close()


def test_new_profile_does_not_hold_previous_camera_measurements() -> None:
    """
    Invalidate averages and held frames when the camera calibration identity changes.
    """
    renderer = ThermalRenderer(scale=1)
    renderer.render_detailed(CameraFrame(PROFILE, np.full((4, 6), 100), converter=convert))
    other = CameraProfile("other", "Other camera", (6, 4), 9)
    rendered = renderer.render_detailed(CameraFrame(
        other, np.full((4, 6), 200), converter=convert, measurement_status="FFC active",
    ))
    assert renderer.last_valid_frame is None and renderer.averaged_raw_counts is None
    assert not rendered.measurements_valid and np.isnan(rendered.temperatures_celsius).all()


def test_controls_restore_only_owned_global_and_spot_settings() -> None:
    """
    Maintain independent spot baselines and protect measurement fields from pipelines.
    """
    owner, values, writes = hardware()
    assert not writes.called
    owner.load()
    owner.set("humidity", 60, True)
    owner.set_spot("distance", 4.05, "A", True)
    assert values["distance", "B"] == 3
    owner.restore()
    assert values["humidity", None] == 40 and values["distance", "A"] == 2
    assert not owner.enabled and not owner.spot_state()["A"]["distance"]["enabled"]
    document = default_pipeline()
    document["hardware"] = [node("hardware", "device_control", control="humidity", value=50)]
    writes.reset_mock()
    with pytest.raises(ValueError, match="outside pipelines"):
        PipelineHardware(owner).apply(document)
    writes.assert_not_called()


def test_missing_radiometry_holds_readings_and_restarts_average() -> None:
    """
    Hold a compatible camera's valid frame through a temporary missing native plane.
    """
    renderer = ThermalRenderer(scale=1, smoothing=0.1)
    valid = CameraFrame(PROFILE, np.full((4, 6), 100), converter=convert)
    missing = CameraFrame(PROFILE, preview=np.full((4, 6), 255, np.uint8))
    try:
        initial = renderer.render_detailed(valid)
        held = renderer.render_detailed(missing)
        assert not held.measurements_valid
        assert held.radiometry_available
        assert "readings held" in held.measurement_status
        assert renderer.last_valid_frame is valid
        np.testing.assert_array_equal(held.image, initial.image)
        np.testing.assert_array_equal(held.temperatures_celsius, initial.temperatures_celsius)
        recovered = renderer.render_detailed(CameraFrame(
            PROFILE, np.full((4, 6), 200), converter=convert,
        ))
        assert recovered.measurements_valid
        np.testing.assert_array_equal(recovered.temperatures_celsius, convert(np.full((4, 6), 200)))
    finally:
        renderer.pipeline_processor.close()


def test_restore_continues_after_independent_failures_and_can_retry() -> None:
    """
    Restore successful globals and spots while retaining all failures for retry.
    """
    owner, values, writes = hardware()
    owner.set("gain", 3, True)
    owner.set("humidity", 60, True)
    owner.set_spot("distance", 4, "A", True)
    owner.set_spot("distance", 5, "B", True)

    def rejecting_write(name: str, value: Any, spot: str | None) -> None:
        """
        Reject two original settings but permit their independent rollback writes.
        """
        if (name == "gain" and value == 1) or (spot == "A" and value == 2):
            raise CameraError("Restore rejected")
        values[name, spot] = value

    writes.side_effect = rejecting_write
    with pytest.raises(CameraError, match=r"gain:.*distance \(A\):"):
        owner.restore()
    assert values["humidity", None] == 40 and values["distance", "B"] == 3
    assert owner.enabled == {"gain"}
    assert owner.spot_state()["A"]["distance"]["enabled"]
    assert not owner.spot_state()["B"]["distance"]["enabled"]

    def write(name: str, value: Any, spot: str | None) -> None:
        """
        Permit a later retry without changing original baseline ownership.
        """
        values[name, spot] = value

    writes.side_effect = write
    owner.restore()
    assert values["gain", None] == 1 and values["distance", "A"] == 2
    assert not owner.enabled and not owner.spot_state()["A"]["distance"]["enabled"]


def test_pipeline_retains_unsupported_nodes_and_restores_new_features() -> None:
    """
    Keep unsupported settings serialized without executing them on another camera.
    """
    owner, values, writes = hardware()
    document = default_pipeline()
    document["hardware"] = [node("hardware", "preset", value="shadow"),
                            node("hardware", "device_control", control="gain", value=3)]
    manager = PipelineHardware(owner)
    manager.apply(document)
    assert values["gain", None] == 3 and len(manager.document["hardware"]) == 2
    writes.reset_mock()
    document["hardware"][0]["expanded"] = False
    manager.apply(document)
    writes.assert_not_called()
    document["hardware"] = []
    manager.apply(document)
    assert values["gain", None] == 1


def test_device_readback_failure_rolls_back_and_owned_restore_can_retry() -> None:
    """
    A rejected write restores the observed value without losing cleanup ownership.
    """
    owner, values, writes = hardware()
    owner.load()
    writes.side_effect = lambda *_args: None
    with pytest.raises(CameraError, match="did not apply"):
        owner.set("gain", 3, True)
    assert values["gain", None] == 1 and "gain" not in owner.enabled


@pytest.mark.parametrize("spot", [None, "A"])
def test_failed_rollback_retains_global_and_spot_cleanup_ownership(spot: str | None) -> None:
    """
    Retry restoration even when an initial override and its rollback both fail.
    """
    owner, values, writes = hardware()
    owner.load()
    name, target = ("gain", 3) if spot is None else ("distance", 4.05)
    baseline = values[name, spot]

    def interrupted(field: str, value: Any, spot_id: str | None) -> None:
        """
        Apply the first write, lose its acknowledgment, and reject rollback.
        """
        if value == target:
            values[field, spot_id] = value
        raise CameraError("Disconnected during write")

    normal_write = writes.side_effect
    writes.side_effect = interrupted
    with pytest.raises(CameraError, match="rollback failed"):
        if spot is None:
            owner.set(name, target, True)
        else:
            owner.set_spot(name, target, spot, True)
    assert values[name, spot] == target
    writes.side_effect = normal_write
    owner.restore()
    assert values[name, spot] == baseline


def test_legacy_node_names_do_not_impose_duo_control_bounds() -> None:
    """
    A same-named target field remains editable through its own metadata-driven node.
    """
    values = {"contrast": 1}
    writes = Mock(side_effect=lambda name, value, _spot: values.update({name: value}))
    owner = DeviceControls(PROFILE, {"contrast": ControlSpec("Contrast", maximum=3)},
                           lambda name, _spot: values[name], writes)
    manager = PipelineHardware(owner)
    document = default_pipeline()
    document["hardware"] = [node("hardware", "contrast", value=50)]
    manager.apply(document)
    writes.assert_not_called()
    document["hardware"].append(node("hardware", "device_control", control="contrast", value=2))
    manager.apply(document)
    assert values["contrast"] == 2


def test_profile_settings_do_not_inherit_duo_calibrations_or_other_optics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Share display choices but isolate overrides, native spots and calibration compatibility.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_settings({"hardware": {"humidity": 50}, "display": {"temperature_unit": "F"},
                   "distance_calibration": {"old": "Duo"}})
    assert load_settings(PROFILE)["hardware"] == {}
    assert load_settings(PROFILE)["display"]["temperature_unit"] == "F"
    owner, _, _ = hardware()
    save_settings({"hardware": {"humidity": 60}, "spot_hardware": {"A": {"distance": 4.05}}},
                  PROFILE)
    assert load_settings(PROFILE, owner.specs)["hardware"] == {"humidity": 60}
    assert load_settings(PROFILE, owner.specs)["spot_hardware"] == {"A": {"distance": 4.05}}
    assert load_settings()["hardware"] == {"humidity": 50}
    other_lens = CameraProfile("test-sensor", "Test camera", (6, 4), 9, calibration_key="lens-B")
    assert load_settings(other_lens, owner.specs)["hardware"] == {}


@pytest.mark.usefixtures("viewer")
def test_desktop_uses_registered_backend_without_duo_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Exercise actual desktop initialization, pipeline edits and UI state for a different camera.
    """
    owner, values, writes = hardware()

    class Camera:
        """
        Supply an explicit camera identity and target-specific control owner.
        """

        profile = PROFILE
        close = Mock()

        @staticmethod
        def create_controls() -> DeviceControls:
            """
            Return this fake's reversible settings owner.
            """
            return owner

    monkeypatch.setattr(camera_backends, "_FACTORIES", {"test-sensor": Camera})
    session = DesktopSession(desktop, ["--camera", "test-sensor"])
    assert session.initialize_hardware()
    assert not writes.called and not session.calibration_available
    document = default_pipeline()
    document["hardware"] = [node("hardware", "device_control", control="gain", value=2)]
    session.set_pipeline(document)
    assert values["gain", None] == 2
    state = session.view_state()
    assert state["camera_profile"]["native_size"] == (6, 4)
    assert state["camera_capabilities"]["nodes"]["preset"]["supported"] is False
    assert session.spots.native_size == (6, 4) and session.recorder.fps == 9
    session.diagnostics = Mock()
    session.shutdown()
    Camera.close.assert_called_once()
    assert values["gain", None] == 1


def test_backend_specific_dependencies_do_not_replace_duo_rules() -> None:
    """
    Separate document syntax from each camera's evidenced compatibility rules.
    """
    document = default_pipeline()
    document["hardware"] = [node("hardware", "preset", value="shadow"),
                            node("hardware", "detail", fixed=True)]
    with pytest.raises(ValueError, match="Fixed detail"):
        validate_pipeline(document)
    assert validate_pipeline(document, hardware_profile=PROFILE.id) == document


def test_pipeline_transfer_preserves_documents_for_other_backends(tmp_path: Path) -> None:
    """
    Store foreign settings structurally and defer Duo compatibility checks to activation.
    """
    document = default_pipeline()
    document["hardware"] = [node("hardware", "preset", value="shadow"),
                            node("hardware", "detail", fixed=True)]
    path = tmp_path / "foreign.pipeline.json"
    write_pipelines(path, {"Foreign camera": document}, single=True)
    incoming = read_pipelines(path)
    assert len(incoming) == 1 and incoming[0].document == document
    with pytest.raises(ValueError, match="Fixed detail"):
        validate_pipeline(incoming[0].document)


def test_frame_pump_rejects_foreign_profile_and_keeps_backend_shutdown_public() -> None:
    """
    Prevent replacement frames from being interpreted using another camera's preferences.
    """
    wrong_profile = CameraProfile("replacement", "Replacement", (6, 4), 9)
    camera = Mock(profile=PROFILE, timeout_ms=1000)
    camera.frames.return_value = iter([CameraFrame(
        wrong_profile, np.full((4, 6), 100), converter=convert,
    )])
    pump = CameraFramePump(camera)
    try:
        with pytest.raises(CameraError, match="incompatible decoded frame"):
            list(pump)
    finally:
        pump.close()
    camera.stop_stream.assert_called_once()
