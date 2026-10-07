"""Pipeline behavior: ordered image operations, safe ownership, and portable state."""

from copy import deepcopy
from threading import Event
from time import monotonic, sleep
from unittest.mock import Mock

import numpy as np
import pytest
from test_desktop_recording import viewer as viewer_fixture
from test_render import frame_with_preview

from topdon_duo.camera import HEADER_U16, SENSOR_PIXELS, CameraError, decode_duo_frame
from topdon_duo.hardware_controls import HARDWARE_CONTROLS
from topdon_duo.pipeline import (
    default_pipeline,
    geometry,
    migrate_pipeline,
    node,
    thermal_source,
    validate_pipeline,
)
from topdon_duo.pipeline_hardware import PipelineHardware, desired_hardware
from topdon_duo.pipeline_processing import PipelineProcessor, PipelineWorker
from topdon_duo.render import ThermalRenderer

viewer = viewer_fixture


def raw_pipeline(*nodes):
    document = default_pipeline()
    document["software"][0]["params"]["source"] = "raw"
    document["software"] = [
        document["software"][0],
        node("software", "range", low=10.0, high=60.0),
        *nodes,
        node("software", "output"),
    ]
    return document


def process(document, scale=1):
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


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(version=4),
        lambda d: d["software"].append(node("software", "source")),
        lambda d: d["software"].insert(0, node("software", "gamma")),
        lambda d: d["software"][0].update(bypass=True),
        lambda d: d["software"].append({**node("software", "gamma"), "type": "unknown"}),
        lambda d: d["software"].append(node("software", "gamma", amount=float("nan"))),
        lambda d: d["software"].append(node("software", "range", low=60.0, high=10.0)),
        lambda d: d["software"].append(node("software", "enhance", passes=1.2)),
        lambda d: d["software"].append(deepcopy(d["software"][0])),
        lambda d: d["hardware"].extend(
            [node("hardware", "brightness"), node("hardware", "brightness", value=20)]
        ),
        lambda d: d["hardware"].append(node("hardware", "detail", enabled=False, fixed=True)),
        lambda d: d["hardware"].append(node("hardware", "humidity", value=10.123)),
    ],
)
def test_reject_invalid_pipeline_atomically(mutation):
    document = default_pipeline()
    mutation(document)
    with pytest.raises((ValueError, TypeError)):
        validate_pipeline(document)


def test_repeat_nodes_and_portable_preferences(tmp_path, monkeypatch):
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = raw_pipeline(node("software", "gamma"), node("software", "gamma", amount=2.0))
    document["software"][1].update(bypass=True, expanded=True)
    valid = validate_pipeline(document)
    assert valid == document and valid is not document
    save_settings({"pipeline": document, "capture_cursor": True})
    assert load_settings()["pipeline"] == document
    assert load_settings()["capture_cursor"] is True


def test_pipeline_order_and_repetition_have_visible_effects():
    bright = node("software", "brightness", amount=20.0)
    gamma = node("software", "gamma", amount=2.0)
    first = process(raw_pipeline(bright, gamma))
    second = process(raw_pipeline(gamma, bright))
    assert not np.array_equal(first, second)
    once = process(raw_pipeline(gamma))
    twice = process(raw_pipeline(gamma, node("software", "gamma", amount=2.0)))
    assert not np.array_equal(once, twice)
    gamma["bypass"] = True
    assert np.array_equal(process(raw_pipeline(gamma)), process(raw_pipeline()))


def test_range_after_processing_preserves_previous_operations():
    narrow = node("software", "range", low=15.0, high=55.0)
    boosted = process(raw_pipeline(node("software", "brightness", amount=10.0), narrow))
    original = process(raw_pipeline(narrow))
    assert not np.array_equal(boosted, original)
    # First range defines the source normalization, even with a filter before it.
    before = raw_pipeline(node("software", "filter", filter="gaussian", amount=1.0))
    after = deepcopy(before)
    after["software"][1:-1] = after["software"][1:-1][::-1]
    assert np.array_equal(process(before), process(after))


def test_camera_preview_range_recolors_thermal_and_keeps_previous_filters():
    document = default_pipeline()
    document["hardware"].append(node("hardware", "camera_colors", palette=11))
    document["software"] = [
        document["software"][0],
        node("software", "brightness", amount=20.0),
        node("software", "range", low=10.0, high=60.0),
        node("software", "output"),
    ]
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    processor = PipelineProcessor()
    enhanced, source = processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert source == "raw" and thermal_source(document)
    document["software"] = [document["software"][0], *document["software"][2:]]
    plain, _ = processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert not np.array_equal(enhanced, plain)
    document["software"] = [document["software"][0], document["software"][-1]]
    _, source = processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert source == "preview"


def test_filters_interpolation_aa_are_real_cumulative_operations():
    document = raw_pipeline(
        node("software", "interpolation", method="nearest", scale=2),
        node("software", "antialiasing", amount=1.0),
    )
    once = process(document, 3)
    document["software"].insert(-1, node("software", "antialiasing", amount=1.0))
    twice = process(document, 3)
    assert not np.array_equal(once, twice)
    document["software"][2]["params"]["method"] = "lanczos"
    assert not np.array_equal(twice, process(document, 3))


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
def test_pipeline_mirrors_match_sensor_coordinates_after_final_rotation(rotation):
    frame, _ = frame_with_preview()
    document = raw_pipeline(node("software", "mirror", horizontal=True))
    renderer = ThermalRenderer(scale=1, rotation=rotation)
    renderer.set_pipeline(document)
    rendered = renderer.render_detailed(frame)
    _, raw, _ = decode_duo_frame(frame)
    assert np.array_equal(rendered.raw_counts, np.rot90(np.fliplr(raw), -(rotation // 90)))
    assert (renderer.mirror_horizontal, renderer.mirror_vertical) == geometry(document, rotation)
    document["software"].insert(-1, node("software", "mirror", horizontal=True))
    renderer.set_pipeline(document)
    assert np.array_equal(
        renderer.render_detailed(frame).raw_counts, np.rot90(raw, -(rotation // 90))
    )


def test_display_operations_never_modify_temperature_statistics():
    frame, _ = frame_with_preview()
    baseline = ThermalRenderer(scale=1).render_detailed(frame)
    document = raw_pipeline(
        node("software", "contrast", amount=2.0),
        node("software", "colors", palette="turbo"),
        node("software", "enhance", model="anime4k09", passes=1),
    )
    renderer = ThermalRenderer(scale=1)
    renderer.set_pipeline(document)
    enhanced = renderer.render_detailed(frame)
    assert enhanced.stats == baseline.stats
    assert np.array_equal(enhanced.raw_counts, baseline.raw_counts)
    assert np.array_equal(enhanced.temperatures_celsius, baseline.temperatures_celsius)


def test_cap_precedes_expensive_allocation():
    document = raw_pipeline(*[node("software", "interpolation", scale=4) for _ in range(3)])
    with pytest.raises(ValueError, match="4 megapixel"):
        process(document)


def fake_hardware():
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

    def set_value(name, value, enabled):
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


def test_hardware_reorder_makes_no_usb_writes_and_bypass_restores_only_owned_fields():
    hw = fake_hardware()
    hw.state.return_value["emissivity"].update(value=0.95, enabled=True)
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"] += [
        node("hardware", "brightness", value=60),
        node("hardware", "contrast", value=70),
    ]
    controller.apply(document)
    hw.reset_mock()
    document["hardware"].reverse()
    controller.apply(document)
    assert not hw.mock_calls
    document["hardware"][0]["bypass"] = True
    controller.apply(document)
    assert hw.set.call_args.args[0] == "contrast" and hw.set.call_args.args[2] is False
    assert hw.state.return_value["emissivity"]["value"] == 0.95
    hw.restore.assert_not_called()


def test_hardware_failure_rolls_back_without_clearing_calibration_fields():
    hw = fake_hardware()
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"].append(node("hardware", "brightness", value=60))
    controller.apply(document)
    candidate = deepcopy(document)
    candidate["hardware"][0]["params"]["value"] = 80
    original = hw.set.side_effect

    def fail(name, value, enabled):
        if value == 80:
            raise CameraError("failed")
        original(name, value, enabled)

    hw.set.side_effect = fail
    with pytest.raises(CameraError, match="failed"):
        controller.apply(candidate)
    assert controller.document == document
    assert hw.state.return_value["brightness"]["value"] == 60
    assert not any(
        call.args[0] in ("ambient", "reflected", "emissivity", "distance")
        for call in hw.set.call_args_list
    )


def test_fixed_dependency_and_raw_preview_exclusion():
    document = default_pipeline()
    document["hardware"].append(node("hardware", "detail", enabled=True, fixed=True))
    controller = PipelineHardware(fake_hardware())
    controller.apply(document)
    assert controller.hardware.fixed_range
    document["software"].insert(-1, node("software", "range"))
    controls, _, _, _, fixed = desired_hardware(document)
    assert not controls and not fixed
    controller.apply(document)
    assert not controller.hardware.fixed_range
    document["software"] = [document["software"][0], document["software"][-1]]
    controller.apply(document)
    assert controller.hardware.fixed_range
    document["hardware"].append(node("hardware", "gamma", value=30))
    with pytest.raises(ValueError, match="Fixed detail"):
        validate_pipeline(document)


def test_migration_retains_analyze_settings_and_old_mirror_geometry():
    document = migrate_pipeline(
        {
            "rotation": 90,
            "display": {
                "analyze_mode": True,
                "mirror_horizontal": True,
                "raw_temperature_low": 20.0,
                "raw_temperature_high": 55.0,
                "raw_upsampling": "anime4k09",
                "raw_sharpen_amount": 0.6,
                "raw_palette": "plasma",
            },
            "hardware": {"palette": 11.0, "ambient": 22.0, "emissivity": 0.95},
        }
    )
    assert geometry(document, 90) == (True, False)
    assert thermal_source(document)
    assert [n["type"] for n in document["software"]] == [
        "source",
        "range",
        "filter",
        "enhance",
        "colors",
        "mirror",
        "antialiasing",
        "output",
    ]
    assert not any(n["type"] in ("ambient", "emissivity") for n in document["hardware"])
    assert next(n for n in document["software"] if n["type"] == "range")["params"] == {
        "low": 20.0,
        "high": 55.0,
    }


def test_worker_overwrites_pending_frames_and_rejects_stale_results():
    worker = PipelineWorker()
    started, release = Event(), Event()
    calls = []

    def slow(frame, *_args):
        calls.append(frame)
        started.set()
        release.wait(2)
        return np.zeros((3, 3, 3), np.uint8), "raw"

    worker.processor.process = slow
    try:
        worker.submit(b"first", None, default_pipeline(), 1, 1, 0, 1)
        assert started.wait(1)
        before = monotonic()
        worker.submit(b"discard", None, default_pipeline(), 2, 1, 0, 1)
        worker.submit(b"newest", None, default_pipeline(), 3, 1, 0, 1)
        assert monotonic() - before < 0.1
        assert worker.latest(3) is None
        release.set()
        deadline = monotonic() + 2
        while worker.latest(3) is None and monotonic() < deadline:
            sleep(0.01)
        assert worker.latest(3) is not None
        assert calls == [b"first", b"newest"]
        assert worker.latest(1) is None
    finally:
        release.set()
        worker.close()


def test_pipeline_editor_context_operations_import_export_and_lock(tmp_path):
    import subprocess
    import sys

    from test_capture_panel import popup_environment

    script = """
import json, sys
from copy import deepcopy
from pathlib import Path
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMenu
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node, default_pipeline
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
state = {"pipeline": default_pipeline(), "pipeline_serial": 12, "processing_preset_available": True,
         "hardware": {"brightness": {"value": 57, "available": True}}}
window.update_state(state)
assert messages == []
editor.insert("hardware", "brightness", 0)
assert editor.document["software"][0]["type"] == "source"
assert editor.document["hardware"][0]["params"]["value"] == 57
assert messages[-1]["serial"] == 13
count = len(messages)
editor.insert("hardware", "brightness", 2)
assert len(messages) == count
# Metadata-only changes cannot be mistaken for camera writes.
item = editor.document["hardware"][0]
_, controls, bypass, title, badge = editor.widgets[item["id"]]
title.click()
assert not item["expanded"]
assert messages[-1]["hardware_operation"] is False
# A queued old state must not overwrite newer edits.
window.update_state(state)
assert editor.document["hardware"][0]["params"]["value"] == 57
editor.insert("software", "gamma", 0)
editor.insert("software", "gamma", 1)
assert sum(n["type"] == "gamma" for n in editor.document["software"]) == 2
software = editor.stacks["software"]
first_id = editor.document["software"][1]["id"]
# Exercise the model move that drag/drop uses, including editor callbacks.
assert software.model().moveRows(QModelIndex(), 1, 1, QModelIndex(), 3)
app.processEvents()
assert editor.document["software"][2]["id"] == first_id
assert messages[-1]["hardware_operation"] is False
menu = QMenu()
editor.stacks["hardware"].add_menu(menu, "Add", 2)
brightness = next(a for a in menu.actions()[0].menu().actions() if a.text() == "Camera brightness")
assert not brightness.isEnabled()
# Bypassed singleton still cannot be duplicated.
item["bypass"] = True
editor.publish()
editor.insert("hardware", "brightness", 2)
assert sum(n["type"] == "brightness" for n in editor.document["hardware"]) == 1
# Only pipeline data is exported.
destination = Path(sys.argv[1]) / "saved.pipeline.json"
QFileDialog.getSaveFileName = lambda *_args: (str(destination), "")
editor.export_file()
exported = json.loads(destination.read_text())
assert set(exported) == {"version", "hardware", "software", "branches"}
assert exported == editor.document
QFileDialog.getOpenFileName = lambda *_args: (str(destination), "")
editor.clear()
assert not editor.document["hardware"] and len(editor.document["software"]) == 2
editor.import_file()
assert editor.document == exported
before = deepcopy(editor.document)
destination.write_text('{"version":99}')
warnings = []
QMessageBox.warning = lambda *args: warnings.append(args)
editor.import_file()
assert warnings and editor.document == before
# Logging cancels timers and blocks all mutations, including programmatic calls.
for _, rows, *_ in editor.widgets.values():
    for row in rows.values(): row.timer.start()
editor.update_state({"pipeline_serial": editor.edit_serial, "pipeline": editor.document, "processing_preset_available": True}, True)
count = len(messages)
for _, rows, bypass, title, badge in editor.widgets.values():
    for row in rows.values():
        assert not row.input.isEnabled() and not row.timer.isActive()
editor.clear()
editor.insert("software", "gamma", 0)
editor.remove("software", 0)
assert editor.document == before and len(messages) == count
editor.update_state({"pipeline_serial": editor.edit_serial, "pipeline": editor.document, "processing_preset_available": True}, False)
editor.clear("software")
assert len(editor.document["software"]) == 2 and len(editor.document["hardware"]) > 0
editor.clear("hardware")
assert not editor.document["hardware"]
editor.remove("software", 0)
assert editor.document["software"][0]["type"] == "source"
# Tabs edit independent stacks; A keeps its fixed output and clear honors tab scope.
assert editor.tab_bar.count() == 4
snapshot_a = deepcopy(editor.document["software"])
editor.tab_bar.setCurrentIndex(1)
assert editor.tab == "B" and len(editor.nodes("software")) == 1
assert "idle" in editor.tab_status.text()
editor.insert("software", "brightness", 1)
assert editor.document["branches"]["B"][1]["type"] == "brightness"
assert editor.document["software"] == snapshot_a
brightness_b = editor.nodes("software")[1]
brightness_row = editor.widgets[brightness_b["id"]][1]["amount"]
brightness_row.input.setValue(12.3)
assert brightness_row.timer.isActive()
editor.tab_bar.setCurrentIndex(0)
assert editor.document["branches"]["B"][1]["params"]["amount"] == 12.3
editor.insert("software", "combine", 999)
assert editor.document["software"][-2]["type"] == "combine"
assert editor.document["software"][-1]["type"] == "output"
output = editor.document["software"][-1]
editor.bypass(output, True)
editor.remove("software", len(editor.document["software"]) - 1)
assert editor.document["software"][-1] == output and not output["bypass"]
editor.tab_bar.setCurrentIndex(1)
assert "connected" in editor.tab_status.text()
editor.insert("software", "combine", 2)
combine = editor.nodes("software")[-1]
editor.change(combine, "tab", "A")
assert "cycle" in warnings[-1][-1]
assert editor.nodes("software")[-1]["params"]["tab"] == "C"
editor.clear("software")
assert len(editor.document["branches"]["B"]) == 1
assert editor.document["software"][-2]["type"] == "combine"
# Updates while B is visible must still update hidden A and C/D.
incoming = deepcopy(editor.document)
incoming["software"][-2]["params"]["opacity"] = 0.7
incoming["branches"]["D"].append(node("software", "gamma"))
editor.update_state({"pipeline": incoming, "pipeline_serial": editor.edit_serial + 1,
                     "processing_preset_available": True}, False)
assert editor.document == incoming
editor.tab_bar.setCurrentIndex(3)
assert editor.nodes("software")[1]["type"] == "gamma"
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial,
                     "processing_preset_available": True}, True)
locked_snapshot = deepcopy(editor.document)
editor.insert("software", "filter", 1)
editor.clear()
assert editor.document == locked_snapshot
editor.update_state({"pipeline": editor.document, "pipeline_serial": editor.edit_serial,
                     "processing_preset_available": True}, False)
editor.tab_bar.setCurrentIndex(0)
window.grab().save(str(Path(sys.argv[1]) / "pipeline-editor.png"))
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_pipeline_fields_preserve_sdk_calibration_bytes_and_restore_original(tmp_path, monkeypatch):
    from test_hardware_controls import Device

    from topdon_duo.hardware_controls import HardwareControls

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    device = Device()
    hw = HardwareControls(Mock(device=device))
    hw.load()
    hw.set("ambient", 22.0, True)
    hw.set("reflected", 24.2, True)
    hw.set("emissivity", 0.95, True)
    calibration = hw.read(3, 1)
    original_preview = {key: hw.read(*key) for key in ((2, 1), (2, 2), (2, 5))}
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"] += [
        node("hardware", "contrast", value=75),
        node("hardware", "detail", amount=40, enabled=True),
        node("hardware", "camera_colors", palette=22),
    ]
    controller.apply(document)
    assert hw.read(3, 1) == calibration
    assert hw.state()["detail_enabled"]["value"] == 1
    for item in document["hardware"]:
        item["bypass"] = True
    controller.apply(document)
    assert hw.read(3, 1) == calibration
    assert {key: hw.read(*key) for key in original_preview} == original_preview


def test_desktop_pipeline_command_persists_and_is_locked_while_logging(viewer, monkeypatch):
    from topdon_duo import desktop
    from topdon_duo.settings_preferences import load_settings

    document = raw_pipeline(node("software", "mirror", horizontal=True))
    document["software"][1]["expanded"] = True
    panel = Mock()
    panel.poll.return_value = [{"action": "pipeline", "document": document, "serial": 1}]
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "waitKey", lambda _delay: ord("q"))
    assert desktop.main([]) == 0
    assert load_settings()["pipeline"] == document
    panel.poll.return_value = []
    assert desktop.main([]) == 0
    assert panel.update.call_args.args[0]["pipeline"] == document
    candidate = default_pipeline()
    panel.poll.return_value = [{"action": "pipeline", "document": candidate, "serial": 2}]
    viewer.graphs.logging = True
    assert desktop.main([]) == 0
    assert load_settings()["pipeline"] == document


def test_legacy_preview_default_migrates_with_matching_colors():
    frame, _ = frame_with_preview()
    baseline = ThermalRenderer(scale=2).render_detailed(frame)
    renderer = ThermalRenderer(scale=2)
    renderer.set_pipeline(migrate_pipeline({}))
    migrated = renderer.render_detailed(frame)
    # Antialiasing now has an explicit smoothing pass; broad uniform areas retain colors.
    assert np.array_equal(migrated.image[30, 30], baseline.image[30, 30])
    assert np.array_equal(migrated.image[-30, -30], baseline.image[-30, -30])
    assert migrated.stats == baseline.stats


def test_software_pipeline_works_when_sdk_controls_are_unavailable():
    hw = fake_hardware()
    hw.original = {}
    hw.load.side_effect = CameraError("Unsupported control layout")
    document = raw_pipeline(node("software", "gamma", amount=2.0))
    controller = PipelineHardware(hw)
    controller.apply(document)
    hw.load.assert_not_called()
    assert controller.document == document
    document["software"][0]["params"]["source"] = "preview"
    document["software"] = [document["software"][0], document["software"][-1]]
    document["hardware"].append(node("hardware", "brightness", value=40))
    with pytest.raises(CameraError, match="Unsupported control layout"):
        controller.apply(document)


def test_preview_unknown_baseline_palette_reports_error_without_killing_worker():
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    document = default_pipeline()
    document["software"] = [
        document["software"][0],
        node("software", "range", low=10.0, high=60.0),
        document["software"][-1],
    ]
    worker = PipelineWorker()
    try:
        worker.submit(frame, raw.astype(np.float32), document, 1, 1, 0, 99)
        deadline = monotonic() + 2
        while worker.latest(1) is None and monotonic() < deadline:
            sleep(0.01)
        assert "Unknown baseline" in worker.latest(1)[-1]
        worker.submit(frame, raw.astype(np.float32), document, 2, 1, 0, 1)
        deadline = monotonic() + 2
        while worker.latest(2) is None and monotonic() < deadline:
            sleep(0.01)
        assert worker.latest(2)[1] is not None and not worker.latest(2)[-1]
    finally:
        worker.close()


@pytest.mark.parametrize("bypass", (False, True))
def test_old_transmission_node_moves_to_standalone_preferences(tmp_path, monkeypatch, bypass):
    from topdon_duo.pipeline import HARDWARE_NODES
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document = default_pipeline()
    document["hardware"].append(
        {
            "id": "old-transmission",
            "type": "transmission",
            "params": {"value": 95},
            "bypass": bypass,
            "expanded": True,
        }
    )
    save_settings({"pipeline": document})
    saved = load_settings()
    assert "transmission" not in HARDWARE_NODES
    assert all(item["type"] != "transmission" for item in saved["pipeline"]["hardware"])
    assert saved["hardware"].get("transmission") == (None if bypass else 95)
    # Importing a pipeline never changes the standalone correction controls.
    assert all(item["type"] != "transmission" for item in validate_pipeline(document)["hardware"])


def test_transmission_is_not_owned_or_restored_by_pipeline_nodes():
    hw = fake_hardware()
    hw.state.return_value["transmission"].update(value=95, enabled=True)
    controller = PipelineHardware(hw)
    document = default_pipeline()
    document["hardware"].append(node("hardware", "brightness", value=60))
    controller.apply(document)
    document["hardware"] = []
    controller.apply(document)
    assert hw.state.return_value["transmission"] == {
        "value": 95,
        "enabled": True,
        "available": True,
    }
    assert all(call.args[0] != "transmission" for call in hw.set.call_args_list)


def test_version_one_source_migrates_to_first_software_node_with_identity_and_settings(
    tmp_path, monkeypatch
):
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    current = raw_pipeline(node("software", "gamma", amount=2.0))
    current["hardware"].append(node("hardware", "brightness", value=65))
    source = current["software"][0]
    source["expanded"] = True
    legacy = deepcopy(current)
    legacy["version"] = 1
    legacy.pop("branches")
    legacy["software"].pop()
    legacy["hardware"].insert(0, legacy["software"].pop(0))
    upgraded = validate_pipeline(legacy)
    assert upgraded["hardware"] == current["hardware"]
    assert upgraded["software"][:-1] == current["software"][:-1]
    assert upgraded["software"][0] == source
    assert legacy["version"] == 1  # validation doesn't mutate the input file
    save_settings({"pipeline": legacy})
    assert load_settings()["pipeline"]["software"][:-1] == current["software"][:-1]
    assert np.array_equal(process(upgraded), process(current))
