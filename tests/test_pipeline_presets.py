import subprocess
import sys

from test_capture_panel import popup_environment


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
assert editor.presets == {} and editor.preset_combo.count() == 2
editor.document["software"].insert(1, node("software", "gamma", amount=1.7))
editor.document["branches"]["B"].append(node("software", "brightness", amount=12))
QInputDialog.getText = lambda *_: ("My test preset", True)
editor.select_preset(1)
saved = load_presets()["My test preset"]
assert saved == editor.document
assert editor.preset_combo.count() == 4
assert editor.preset_combo.currentIndex() == 0
editor.defaults()
editor.select_preset(3)
assert editor.document == saved and messages[-1]["document"] == saved
assert editor.preset_combo.currentIndex() == 0
second = ViewWindow(messages.append).pipeline_editor
assert second.presets["My test preset"] == saved
second.update_state({"pipeline": second.document, "pipeline_serial": 0}, True)
assert not second.preset_combo.isEnabled()
before = second.document
second.select_preset(3)
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
