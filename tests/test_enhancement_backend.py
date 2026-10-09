import subprocess
import sys
from copy import deepcopy

import cv2
import numpy as np
from test_capture_panel import popup_environment
from test_pipeline import raw_pipeline
from test_render import frame_with_preview

from topdon_duo.camera import decode_duo_frame
from topdon_duo.pipeline import node, validate_pipeline
from topdon_duo.pipeline_processing import PipelineProcessor


def test_old_enhancement_presets_gain_cpu_defaults_without_mutation():
    document = raw_pipeline(node("software", "enhance", model="acnet", passes=2))
    item = document["software"][2]
    del item["params"]["backend"]
    del item["params"]["apple_compute"]
    original = deepcopy(document)
    migrated = validate_pipeline(document)
    assert document == original
    params = migrated["software"][2]["params"]
    assert params["backend"] == "cpu" and params["apple_compute"] == "CPUAndGPU"
    assert params["passes"] == 2


def test_cpu_system_ignores_but_retains_saved_apple_preferences(monkeypatch):
    def no_apple():
        raise AssertionError("Apple helper must not load on CPU-only system")

    monkeypatch.setattr("topdon_duo.processing.branch.CoreMLUpsampler", no_apple)
    document = raw_pipeline(
        node(
            "software",
            "enhance",
            model="acnet",
            backend="coreml",
            apple_compute="ALL",
        )
    )
    original = deepcopy(document)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    processor = PipelineProcessor(apple_available=False)
    image, source = processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert source == "raw" and image.shape == (192, 256, 3)
    assert document == original
    cpu = deepcopy(document)
    cpu["software"][2]["params"]["backend"] = "cpu"
    baseline, _ = processor.process(frame, raw.astype(np.float32), cpu, scale=1)
    assert np.array_equal(image, baseline)
    assert not processor.coreml_models
    assert validate_pipeline(document)["software"][2]["params"]["backend"] == "coreml"
    processor.close()


def test_gpu_node_uses_isolated_engine_and_preserves_passes(monkeypatch):
    calls = []

    class Engine:
        error = ""

        def apply(self, image, denoise, compute, amount):
            calls.append((image.shape[:2], denoise, compute, amount))
            return cv2.resize(image, None, fx=2, fy=2)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr("topdon_duo.processing.branch.CoreMLUpsampler", Engine)
    document = raw_pipeline(
        node(
            "software",
            "enhance",
            model="acnet",
            backend="coreml",
            apple_compute="ALL",
            denoise=2,
            passes=2,
        )
    )
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    processor = PipelineProcessor(apple_available=True)
    processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert calls == [((192, 256), 2, "ALL", 1.0), ((384, 512), 2, "ALL", 1.0)]
    assert len(processor.coreml_models) == 1 and not processor.models
    document["software"][2]["params"]["backend"] = "cpu"
    processor.process(frame, raw.astype(np.float32), document, scale=1)
    assert not processor.coreml_models and "closed" in calls
    processor.close()


def test_anime_ignores_saved_apple_backend_even_on_gpu_system(monkeypatch):
    def no_apple():
        raise AssertionError("Anime4K09 must not use Core ML")

    monkeypatch.setattr("topdon_duo.processing.branch.CoreMLUpsampler", no_apple)
    document = raw_pipeline(
        node(
            "software",
            "enhance",
            model="anime4k09",
            backend="coreml",
            passes=1,
        )
    )
    original = deepcopy(document)
    frame, _ = frame_with_preview()
    _, raw, _ = decode_duo_frame(frame)
    processor = PipelineProcessor(apple_available=True)
    try:
        result, _ = processor.process(frame, raw.astype(np.float32), document, scale=1)
        assert result.shape == (192, 256, 3)
        assert document == original
        assert not processor.coreml_models
    finally:
        processor.close()


def test_gpu_controls_hidden_on_cpu_but_saved_settings_survive(tmp_path):
    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
from topdon_duo.pipeline_presets import save_presets, load_presets
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
item = node("software", "enhance", model="acnet", backend="coreml", apple_compute="ALL")
editor.document["software"].insert(1, item)
editor.rebuild()
original = deepcopy(editor.document)
for detected in (True, False, True):
    editor.update_state({"pipeline": original, "apple_acceleration": {"available": detected}}, False)
    rows = editor.widgets[item["id"]][1]
    assert rows["backend"].isHidden() is (not detected)
    assert rows["apple_compute"].isHidden() is (not detected)
    assert editor.document == original
    save_presets({"Portable Apple preference": editor.document})
    assert load_presets()["Portable Apple preference"] == original
window.close()
"""
    env = popup_environment()
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_anime_is_cpu_only_and_backend_never_changes_selected_model():
    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
item = node("software", "enhance")
editor.document["software"].insert(1, item)
editor.rebuild()
state = {"pipeline": editor.document, "apple_acceleration": {"available": True}}
editor.update_state(state, False)
rows = editor.widgets[item["id"]][1]
assert rows["backend"].isHidden() and not rows["backend"].input.isEnabled()
assert rows["apple_compute"].isHidden()
editor.change(item, "apple_compute", "ALL")
assert item["params"]["model"] == "anime4k09"
editor.change(item, "backend", "coreml")
assert item["params"]["model"] == "anime4k09"
assert item["params"]["passes"] == 3
editor.change(item, "model", "acnet")
editor.update_state({"pipeline_serial": editor.edit_serial,
                     "apple_acceleration": {"available": True}}, False)
assert item["params"]["model"] == "acnet"
assert item["params"]["passes"] == 1
assert item["params"]["apple_compute"] == "ALL"
assert rows["model"].value() == "acnet"
assert not rows["backend"].isHidden() and rows["backend"].input.isEnabled()
assert rows["apple_compute"].input.isEnabled()
editor.change(item, "model", "anime4k09")
editor.update_state({"pipeline_serial": editor.edit_serial,
                     "apple_acceleration": {"available": True}}, False)
assert rows["backend"].isHidden()
assert item["params"]["backend"] == "coreml"
assert item["params"]["apple_compute"] == "ALL"
editor.update_state(state, True)
assert not rows["backend"].input.isEnabled()
assert not rows["apple_compute"].input.isEnabled()
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
