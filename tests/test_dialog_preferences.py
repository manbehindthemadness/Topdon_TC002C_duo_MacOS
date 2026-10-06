import json
from unittest.mock import Mock

import pytest

from topdon_duo.desktop import LinuxSaveDialog, MacSaveDialog
from topdon_duo.dialog_preferences import load_dialog_directory, remember_dialog_directory


@pytest.mark.parametrize("dialog_class", [MacSaveDialog, LinuxSaveDialog])
@pytest.mark.parametrize(
    "kind,suffix",
    [("", ".png"), ("video", ".mp4"), ("timelapse", ".mp4"), ("temperatures", ".csv")],
)
def test_save_dialog_remembers_accepted_directory_on_next_launch(
    monkeypatch, tmp_path, dialog_class, kind, suffix
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    directory = tmp_path / "captures with spaces"
    directory.mkdir()
    process = Mock(returncode=0)
    process.poll.return_value = 0
    process.communicate.return_value = (str(directory / ("capture" + suffix)) + "\n", "")
    popen = Mock(return_value=process)
    monkeypatch.setattr("topdon_duo.desktop.subprocess.Popen", popen)
    first = dialog_class()
    assert first.open(suffix=suffix, kind=kind)
    assert first.poll() == (True, directory / ("capture" + suffix))
    second = dialog_class()
    assert second.open(suffix=suffix, kind=kind)
    command = popen.call_args.args[0]
    if dialog_class is MacSaveDialog:
        assert command[-1] == str(directory)
    else:
        assert any(argument.startswith(f"--filename={directory}/") for argument in command)


def test_each_directory_is_independent_and_cancel_does_not_replace_it(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for kind in ("image", "video", "timelapse", "temperatures"):
        directory = tmp_path / kind
        directory.mkdir()
        remember_dialog_directory(kind, directory / "selected.file")
    process = Mock(returncode=1)
    process.poll.return_value = 1
    process.communicate.return_value = ("", "")
    monkeypatch.setattr("topdon_duo.desktop.subprocess.Popen", lambda *args, **kwargs: process)
    dialog = LinuxSaveDialog()
    dialog.open()
    assert dialog.poll() == (True, None)
    for kind in ("image", "video", "timelapse", "temperatures"):
        assert load_dialog_directory(kind) == tmp_path / kind


def test_invalid_missing_and_explicit_directories(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    remembered = tmp_path / "remembered"
    remembered.mkdir()
    remember_dialog_directory("image", remembered / "capture.png")
    assert load_dialog_directory("image", tmp_path) == tmp_path
    assert load_dialog_directory("image", tmp_path / "missing") == remembered
    remembered.rmdir()
    assert load_dialog_directory("image") is None
    path = tmp_path / "config" / "topdon-duo" / "dialog-directories" / "image.json"
    for contents in ("{", "null", json.dumps("relative/path"), json.dumps([str(tmp_path)])):
        path.write_text(contents)
        assert load_dialog_directory("image") is None
    assert not list(path.parent.glob("*.tmp"))
