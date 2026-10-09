import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_capture_panel import popup_environment


def test_predefined_redneck_combat_matches_original_export():
    from topdon_duo.pipeline import validate_pipeline
    from topdon_duo.pipeline_presets import predefined_presets

    exported = json.loads(
        (Path(__file__).resolve().parents[1] / "presets" / "Redneck Combat.pipeline.json").read_text()
    )
    assert predefined_presets()["Redneck Combat"] == validate_pipeline(exported)


def test_predefined_redneck_gpu_retains_current_enhancement_and_branch_settings() -> None:
    """
    Retain the tuned GPU enhancement and threshold-mask edge branch.
    """
    from topdon_duo.pipeline_presets import predefined_presets

    document = predefined_presets()["Redneck Combat GPU"]
    enhance = next(n for n in document["software"] if n["type"] == "enhance")
    expected_gpu = {
        "model": "realesr-general-x4v3",
        "backend": "coreml",
        "apple_compute": "ALL",
        "input": "native",
    }
    assert {key: enhance["params"][key] for key in expected_gpu} == expected_gpu
    combine = next(n for n in document["software"] if n["type"] == "combine")
    assert combine["params"]["mask_kind"] == "threshold"
    branch = document["branches"]["B"]
    filters = [n["params"] for n in branch if n["type"] == "filter"]
    assert filters[0]["filter"] == "scharr" and filters[0]["gain"] == 0.2
    assert filters[1]["filter"] == "sharpen" and filters[1]["amount"] == 3.0
    assert any(n["type"] == "contours" for n in branch)


def test_predefined_yautja_preserves_all_branches_and_apple_acnet():
    from topdon_duo.pipeline_presets import predefined_presets

    document = predefined_presets()["Yautja"]
    colors = next(n for n in document["software"] if n["type"] == "colors")
    assert colors["params"]["palette"] == "black_hot"
    enhance = next(n for n in document["software"] if n["type"] == "enhance")
    assert enhance["params"]["model"] == "acnet"
    assert enhance["params"]["passes"] == 1
    assert enhance["params"]["backend"] == "coreml"
    assert enhance["params"]["apple_compute"] == "ALL"
    combines = [n["params"] for n in document["software"] if n["type"] == "combine"]
    assert combines[0]["tab"] == "D" and combines[0]["mask_source"] == "C"
    assert combines[0]["mode"] == "screen"
    assert combines[1]["tab"] == "B" and combines[1]["mask_kind"] == "threshold"
    assert document["branches"]["C"][0]["params"]["source"] == "raw"
    assert document["branches"]["C"][1]["params"]["amount"] == 3.0
    assert document["branches"]["D"][1]["params"]["palette"] == "camera_19"
    assert [n["type"] for n in document["hardware"]] == ["detail", "noise", "preset"]


def test_predefined_yautja_gpu_preserves_compositing_and_realesr_settings():
    from topdon_duo.pipeline_presets import predefined_presets

    presets = predefined_presets()
    document = presets["Yautja GPU"]
    original = presets["Yautja"]
    assert document["hardware"] == original["hardware"]
    assert document["branches"] == original["branches"]
    for saved, prior in zip(document["software"], original["software"], strict=True):
        if saved["type"] != "enhance":
            assert saved == prior
    enhance = next(n for n in document["software"] if n["type"] == "enhance")
    assert enhance["params"]["model"] == "realesr-general-x4v3"
    assert enhance["params"]["backend"] == "coreml"
    assert enhance["params"]["apple_compute"] == "ALL"
    assert enhance["params"]["input"] == "native"
    assert enhance["params"]["amount"] == 1.0


def test_predefined_detail_pipeline_retains_gpu_and_edge_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Keep the captured current enhancement node and edge branch in the bundled preset.
    """
    from topdon_duo.pipeline_presets import load_presets, predefined_presets, save_presets

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    name = "Detail Enhanced GPU Upscale"
    original = predefined_presets()[name]
    assert load_presets()[name] == original
    assert not any(n["type"] == "onnx_superresolution" for n in original["software"])
    upscale = next(n for n in original["software"] if n["type"] == "enhance")
    assert upscale["params"] == {
        "model": "realesr-general-x4v3", "input": "native", "amount": 1.0,
        "backend": "coreml", "apple_compute": "ALL",
        "denoise": 0, "noise": 15, "passes": 3,
    }
    combine = next(n for n in original["software"] if n["type"] == "combine")
    assert combine["params"]["tab"] == "B"
    assert original["branches"]["B"][0]["params"]["source"] == "raw"
    preview = next(n for n in original["branches"]["B"] if n["type"] == "preview")
    assert not preview["expanded"]
    save_presets(load_presets())
    assert (tmp_path / "topdon-duo" / "pipeline-presets.json").read_text().strip() == "{}"
    assert load_presets()[name] == original
    modified = load_presets()
    modified[name]["software"][0]["expanded"] = True
    save_presets(modified)
    assert load_presets() == modified
    assert predefined_presets()[name] == original


def test_corrupt_user_presets_do_not_hide_predefined_pipeline(monkeypatch, tmp_path):
    from topdon_duo.pipeline_presets import load_presets

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    directory = tmp_path / "topdon-duo"
    directory.mkdir()
    (directory / "pipeline-presets.json").write_text("broken json")
    assert "Detail Enhanced GPU Upscale" in load_presets()


def test_user_presets_save_reload_apply_and_obey_lock(tmp_path):
    script = '''
from PySide6.QtWidgets import QApplication, QInputDialog
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
from topdon_duo.pipeline_presets import load_presets
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
assert "Detail Enhanced GPU Upscale" in editor.presets
assert "Redneck Combat" in editor.presets
assert "Redneck Combat GPU" in editor.presets
assert "Yautja" in editor.presets
assert "Yautja GPU" in editor.presets
assert editor.preset_combo.count() == 8
editor.insert("software", "enhance", 1)
enhance = editor.document["software"][1]
assert enhance["params"]["passes"] == 3
editor.change(enhance, "model", "acnet")
assert enhance["params"]["passes"] == 1
assert editor.widgets[enhance["id"]][1]["passes"].input.value() == 1
editor.remove("software", 1)
editor.document["software"].insert(1, node("software", "gamma", amount=1.7))
editor.document["branches"]["B"].append(node("software", "brightness", amount=12))
QInputDialog.getText = lambda *_: ("My test preset", True)
editor.select_preset(1)
saved = load_presets()["My test preset"]
assert saved == editor.document
assert editor.preset_combo.count() == 9
assert editor.preset_combo.currentIndex() == 0
editor.defaults()
editor.select_preset(editor.preset_combo.findText("My test preset"))
assert editor.document == saved and messages[-1]["document"] == saved
assert editor.preset_combo.currentIndex() == 0
editor.select_preset(editor.preset_combo.findText("Redneck Combat"))
assert editor.document == editor.presets["Redneck Combat"]
assert messages[-1]["document"] == editor.presets["Redneck Combat"]
editor.select_preset(editor.preset_combo.findText("Redneck Combat GPU"))
assert editor.document == editor.presets["Redneck Combat GPU"]
assert messages[-1]["document"] == editor.presets["Redneck Combat GPU"]
editor.select_preset(editor.preset_combo.findText("Yautja"))
assert editor.document == editor.presets["Yautja"]
assert messages[-1]["document"] == editor.presets["Yautja"]
editor.select_preset(editor.preset_combo.findText("Yautja GPU"))
assert editor.document == editor.presets["Yautja GPU"]
assert messages[-1]["document"] == editor.presets["Yautja GPU"]
second = ViewWindow(messages.append).pipeline_editor
assert second.presets["My test preset"] == saved
second.update_state({"pipeline": second.document, "pipeline_serial": 0}, True)
assert not second.preset_combo.isEnabled()
before = second.document
second.select_preset(second.preset_combo.findText("My test preset"))
assert second.document == before
window.close()
'''
    env = popup_environment()
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True,
        timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
