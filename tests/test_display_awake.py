import subprocess
from unittest.mock import Mock

import pytest

from topdon_duo import display_awake
from topdon_duo.display_awake import DisplayAwake


@pytest.mark.parametrize(
    "platform,desktop,gnome",
    [
        ("darwin", "", None),
        ("linux", "ubuntu:GNOME", "/usr/bin/gnome-session-inhibit"),
        ("linux", "KDE", None),
    ],
)
def test_display_inhibitor_commands_and_cleanup(monkeypatch, platform, desktop, gnome):
    monkeypatch.setattr(display_awake.sys, "platform", platform)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
    monkeypatch.setattr(display_awake.shutil, "which", lambda _name: gnome)
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(display_awake.subprocess, "Popen", popen)
    awake = DisplayAwake()
    awake.start()
    awake.start()
    popen.assert_called_once()
    command = popen.call_args.args[0]
    if platform == "darwin":
        assert command[:3] == ["/usr/bin/caffeinate", "-d", "-w"]
    elif gnome:
        assert command[-2:] == ["--gnome", gnome]
    else:
        assert command[-2:] == ["-m", "topdon_duo.display_awake_helper"]
    assert awake.check() is None
    awake.close()
    awake.close()
    process.stdin.close.assert_called_once()
    process.wait.assert_called_once_with(timeout=1)
    assert process.terminate.call_count == (1 if platform == "darwin" else 0)


def test_request_failure_is_reported_once_without_breaking_viewer(monkeypatch):
    monkeypatch.setattr(display_awake.sys, "platform", "darwin")
    monkeypatch.setattr(
        display_awake.subprocess, "Popen", Mock(side_effect=OSError("missing helper"))
    )
    awake = DisplayAwake()
    awake.start()
    assert "missing helper" in awake.check()
    assert awake.check() is None
    awake.close()


def test_exited_helper_and_slow_cleanup_are_handled(monkeypatch):
    monkeypatch.setattr(display_awake.sys, "platform", "linux")
    process = Mock(returncode=2)
    process.poll.return_value = 2
    process.communicate.return_value = (None, b"Idle inhibition refused")
    monkeypatch.setattr(display_awake.subprocess, "Popen", Mock(return_value=process))
    awake = DisplayAwake()
    awake.start()
    assert "Idle inhibition refused" in awake.check()
    assert awake.check() is None
    awake.close()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired("helper", 1),
        subprocess.TimeoutExpired("helper", 1),
        0,
    ]
    awake.start()
    awake.close()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
