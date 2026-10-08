import subprocess
import sys

import pytest
from test_capture_panel import popup_environment


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_native_placement_on_mac_and_snapping_on_linux(platform):
    script = """
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication
import topdon_duo.view_window as module
module.sys = SimpleNamespace(platform=PLATFORM)
app = QApplication([])
class Window(module.ViewWindow):
    moves = []
    def move(self, *args):
        self.moves.append(args)
        return super().move(*args)
window = Window(lambda _: None)
window.update_state({"anchor_top_right": [650, 30]})
window.show()
app.processEvents()
window._position_top_right()
app.processEvents()
if PLATFORM == "darwin":
    assert not window.moves, window.moves
    assert window._anchor_top_right is None
else:
    assert window.moves
    assert window._anchor_top_right.x() == 650
window.close()
""".replace("PLATFORM", repr(platform))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
