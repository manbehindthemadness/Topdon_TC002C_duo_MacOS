"""
Exercise capture transitions directly without acquiring frames or running the viewer loop.
"""

from typing import Any

import pytest
from support.desktop_recording import viewer_fixture

from topdon_duo import desktop
from topdon_duo.desktop_app.session import DesktopSession


@pytest.mark.usefixtures("viewer")
def test_repeating_record_request_cancels_pending_dialog() -> None:
    """
    A second request for the pending recording cancels its dialog before encoding starts.
    """
    session = DesktopSession(desktop, [])
    session.request_save("video")
    assert session.pending_save_kind == "video"
    assert session.save_dialog.is_open
    session.request_save("video")
    assert session.pending_save_kind is None
    assert not session.save_dialog.is_open
    assert not session.recorder.is_recording


def test_recording_mode_stays_fixed_until_matching_stop_request(viewer: Any) -> None:
    """
    A different mode request leaves an active recording intact; its own button stops it.
    """
    session = DesktopSession(desktop, [])
    session.recorder.start(viewer.dialog.selected, (192, 256, 3), "video")
    session.request_save("timelapse")
    assert session.recorder.mode == "video"
    assert not session.save_dialog.is_open
    session.request_save("video")
    assert not session.recorder.is_recording
    viewer.writer.release.assert_called_once()


@pytest.mark.usefixtures("viewer")
def test_pending_recording_freezes_timelapse_rate() -> None:
    """
    Rate changes are deferred while a recording filename is pending.
    """
    session = DesktopSession(desktop, [])
    session.set_timelapse_fpm(17)
    session.request_save("timelapse")
    session.set_timelapse_fpm(30)
    assert session.timelapse_fpm == 17
    session.request_save("timelapse")
    session.set_timelapse_fpm(30)
    assert session.timelapse_fpm == 30


__all__ = ["viewer_fixture"]
