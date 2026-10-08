import pytest

from topdon_duo.enhancement_limits import acnet_pass_limit, enhancement_pass_limits
from topdon_duo.pipeline import default_pipeline, node


@pytest.mark.parametrize(
    "width,height,maximum",
    [
        (256, 192, 3),
        (512, 384, 2),
        (1024, 768, 1),
        (2048, 1536, 0),
    ],
)
def test_acnet_actual_input_budget(width, height, maximum):
    assert acnet_pass_limit(width, height) == maximum


def test_limits_follow_input_size_model_and_preceding_nodes():
    document = default_pipeline()
    scale = node("software", "interpolation", scale=2)
    ai = node("software", "enhance", model="acnet", passes=5)
    document["software"].insert(-1, scale)
    document["software"].insert(-1, ai)
    assert enhancement_pass_limits(document)[ai["id"]] == 1
    enhancement_pass_limits(document, clamp=True)
    assert ai["params"]["passes"] == 1
    ai["params"].update(input="native", passes=5)
    enhancement_pass_limits(document, clamp=True)
    assert ai["params"]["passes"] == 3
    ai["params"].update(input="preview", passes=5)
    enhancement_pass_limits(document, clamp=True)
    assert ai["params"]["passes"] == 2
    ai["params"].update(model="anime4k09", passes=5)
    assert enhancement_pass_limits(document, clamp=True)[ai["id"]] == 5
    assert ai["params"]["passes"] == 5


def test_bypassed_upstream_scales_and_thermal_range_are_accounted_for():
    document = default_pipeline()
    document["software"].insert(1, node("software", "range"))
    scale = node("software", "interpolation", scale=4)
    scale["bypass"] = True
    document["software"].insert(-1, scale)
    ai = node("software", "enhance", model="acnet", passes=5)
    document["software"].insert(-1, ai)
    assert enhancement_pass_limits(document)[ai["id"]] == 3


@pytest.mark.parametrize("model,factor", [
    ("espcn", 3), ("mewzoom-v1-2x", 2), ("realesr-general-x4v3", 4),
    ("dncnn-25", 1), ("ffdnet-gray", 1),
])
def test_regular_ai_onnx_limits_account_fixed_scale_and_ignore_hidden_passes(model, factor):
    from topdon_duo.onnx_models import MODELS

    document = default_pipeline()
    ai = node("software", "enhance", model=model, input="native", passes=5)
    following = node("software", "enhance", model="acnet", passes=5)
    document["software"].insert(-1, ai)
    document["software"].insert(-1, following)
    limits = enhancement_pass_limits(document, clamp=True)
    width, height = (512, 384) if factor == 1 else (256, 192)
    assert limits[ai["id"]] == 1
    assert ai["params"]["passes"] == 5  # Hidden and ignored, not multiplied.
    assert limits[following["id"]] == acnet_pass_limit(width * factor, height * factor)
    assert MODELS[model]["factor"] == factor


def test_preview_control_pass_bounds_update_and_save_clamped_value(tmp_path):
    import subprocess
    import sys

    from test_capture_panel import popup_environment

    script = """
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import node
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
editor = window.pipeline_editor
ai = node("software", "enhance", model="acnet", passes=5)
editor.document["software"].insert(-1, ai)
editor.rebuild()
editor.update_state({"pipeline": editor.document}, False)
row = editor.widgets[ai["id"]][1]["passes"]
assert row.input.maximum() == 2 and row.slider.maximum() == 1
assert ai["params"]["passes"] == 2
editor.change(ai, "input", "native")
editor.update_state({"pipeline_serial": editor.edit_serial}, False)
assert row.input.maximum() == 3 and row.slider.maximum() == 2
editor.change(ai, "passes", 5)
assert messages[-1]["document"]["software"][-2]["params"]["passes"] == 3
editor.change(ai, "model", "anime4k09")
editor.update_state({"pipeline_serial": editor.edit_serial}, False)
assert row.input.maximum() == 5 and row.slider.maximum() == 4
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
