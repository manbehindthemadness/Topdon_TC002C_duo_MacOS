"""
Main toolbar access to measurement configuration without changing placement mode.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401

from based_thermals import desktop


def test_measurements_control_opens_popup_and_retains_parent_lock(
    viewer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Open editable geometry from the Qt control, including read-only logging access.
    """
    panel = Mock()
    monkeypatch.setattr("based_thermals.desktop.SpotsPanel", lambda: panel)
    session = desktop.DesktopSession(desktop, [])
    viewer_panel = Mock(closed=False)
    session.viewer = viewer_panel
    viewer_panel.poll.return_value = [{"action": "spots"}]
    assert session.poll_viewer_commands()
    panel.open.assert_called_once_with(session.spots_state())
    assert not session.spots.placing
    assert session.viewer_keys == []
    viewer.graphs.logging = True
    assert session.poll_viewer_commands()
    assert panel.open.call_args.args[0]["locked"]
