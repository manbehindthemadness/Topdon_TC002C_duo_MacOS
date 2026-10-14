"""
Bulk measurement visibility, mixed checkbox state and parent lock enforcement.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.measurements.schema import default_region


def test_show_all_checkbox_states_and_pending_edits(tmp_path: Path) -> None:
    """
    Exercise checked/mixed/hidden/empty/locked states without losing unsaved edits.
    """
    result = run_popup(
        """
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from based_thermals.measurements_window import MeasurementsWindow
from based_thermals.measurements.schema import default_region
app = QApplication([])
messages = []
window = MeasurementsWindow(messages.append)
state = {"sensor_size": [32, 24], "orientation": [0, False, False], "locked": False,
         "spots": [{"number": n, "name": str(n), "enabled": True, "pixel": [10, 10],
                    "region": default_region()} for n in (1, 2)]}
window.update_state(deepcopy(state))
assert window.show_all.checkState() == Qt.CheckState.Checked
assert not messages
window._dirty = True
window.name.setText("Pending")
window.show_all.click()
assert messages == [{"action": "enable_all", "enabled": False}]
assert window._dirty and window.name.text() == "Pending"
for item in state["spots"]:
    item["enabled"] = False
window.update_state(deepcopy(state))
assert window.show_all.checkState() == Qt.CheckState.Unchecked
window.show_all.click()
assert messages[-1] == {"action": "enable_all", "enabled": True}
state["spots"][0]["enabled"] = True
window.update_state(deepcopy(state))
assert window.show_all.checkState() == Qt.CheckState.PartiallyChecked
count = len(messages)
window.show_all.click()
assert messages[count:] == [{"action": "enable_all", "enabled": True}]
assert window.show_all.checkState() == Qt.CheckState.Checked
window.update_state({**deepcopy(state), "locked": True})
assert not window.show_all.isEnabled()
count = len(messages)
window.show_all.click()
assert len(messages) == count
window.update_state({**state, "spots": []})
assert not window.show_all.isEnabled()
assert window.show_all.checkState() == Qt.CheckState.Unchecked
assert len(messages) == count
window.close()
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locked", [False, True])
def test_bulk_command_preserves_geometry_and_obeys_parent_lock(viewer: Any, locked: bool) -> None:
    """
    Apply one atomic bulk operation through the controller, including exclusions.
    """
    session = desktop.DesktopSession(desktop, [])
    session.spots.create_region((10, 10), default_region(), (256, 192))
    session.spots.create_region((20, 20), {**default_region(), "exclude": True}, (256, 192))
    before = deepcopy(session.spots.region_states())
    session.spots_panel = Mock()
    session.spots_panel.poll.return_value = [{"action": "enable_all", "enabled": False}]
    viewer.graphs.logging = locked
    session.poll_spot_and_graph_commands()
    assert session.spots.disabled == (set() if locked else {1, 2})
    assert session.spots.numbers == [1, 2] and session.spots.generation == 0
    for old, new in zip(before, session.spots.region_states(), strict=True):
        assert old["region"] == new["region"] and old["pixel"] == new["pixel"]
    viewer.graphs.logging = False
    session.spots_panel.poll.return_value = [{"action": "enable_all", "enabled": True}]
    session.poll_spot_and_graph_commands()
    assert not session.spots.disabled
    restored = desktop.SampleSpots()
    restored.restore_saved_state(session.spots.saved_state())
    assert restored.region_states() == session.spots.region_states()


@pytest.mark.parametrize("enabled", [None, 0, 1, "true", [], {}])
def test_invalid_bulk_state_does_not_partially_change_measurements(enabled: object) -> None:
    """
    Reject malformed bulk states before replacing the enabled identity set.
    """
    spots = desktop.SampleSpots(pixels=[(10, 10), (20, 20)])
    spots.set_enabled(1, False)
    with pytest.raises(ValueError):
        spots.set_all_enabled(enabled)
    assert spots.disabled == {1}
