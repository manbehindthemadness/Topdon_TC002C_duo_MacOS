"""
Support archives preserve client-side evidence and keep file work off acquisition.
"""

import json
import logging
from pathlib import Path
from threading import Event
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.maintenance import MaintenancePaths, MaintenancePlan
from based_thermals.maintenance import export as support
from based_thermals.maintenance.logging import SupportLogging, debug_enabled, save_debug
from based_thermals.viewer_diagnostics import ViewerDiagnostics


def paths(root: Path) -> MaintenancePaths:
    """
    Keep all support-test data in temporary directories.
    """
    return MaintenancePaths(
        (root / "config",),
        root / "models",
        root / "runtimes",
        root / "state",
        (root / "source",),
        (root / ":memory:.ses",),
        (root / "diagnostics",),
    )


def write(path: Path, content: bytes) -> Path:
    """
    Create an original evidence file for the archive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.mark.parametrize("all_data", [False, True])
def test_archive_saved_originals_live_edits_logs_and_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    all_data: bool,
) -> None:
    """
    Keep malformed saved evidence separate from current edits and omit heavyweight/source assets.
    """
    scope = paths(tmp_path)
    original = write(scope.settings[0] / "settings.json", b"{bad JSON")
    write(scope.settings[0] / "pipeline-presets.json", b'{"custom": {}}')
    write(scope.settings[0] / "camera-profiles/other.json", b"{}")
    write(scope.settings[0] / "custom_nodes/mine/__init__.py", b"original code")
    write(scope.models / "custom-models.json", b"[]")
    write(scope.models / "large.onnx", b"model")
    write(scope.state / "debug-logs/application.log", b"client error")
    write(scope.state / "debug-logs/application.log.1", b"previous error")
    write(scope.runtimes / "state.json", b'{"failed_at": 1}')
    write(scope.artifacts[0], b"session")
    write(scope.outputs[0] / "trace.jsonl", b'{"stage": "controls"}')
    write(scope.outputs[0] / "trace.stacks.log", b"stack")
    write(scope.outputs[0] / "trace.rejected-01.bin", b"rejected frame")
    write(scope.outputs[0] / "capture.png", b"image")
    cli_log = write(tmp_path / "elsewhere/custom", b"cli trace")
    write(cli_log.with_suffix(".stacks.log"), b"cli stack")
    temperature = write(tmp_path / "temperatures/user-named.csv", b"temperature log")
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(support, "load_dialog_directory", lambda kind: temperature.parent)
    monkeypatch.setattr(support, "support_metadata", lambda: {"machine": "aarch64"})
    destination = tmp_path / "report.zip"
    count, warnings = support.create_archive(
        destination,
        scope,
        {"settings": {"changed": True}, "session": {"camera": "duo"}},
        all_data=all_data,
        diagnostics=cli_log,
    )
    assert count >= 4 and not warnings
    assert original.read_bytes() == b"{bad JSON"
    with ZipFile(destination) as archive:
        names = archive.namelist()
        assert archive.read("settings/0/settings.json") == b"{bad JSON"
        assert json.loads(archive.read("current-settings.json")) == {"changed": True}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["kind"] == ("all" if all_data else "settings")
        assert not any(name.endswith((".onnx", ".png", ".py")) for name in names)
        assert ("system.json" in names) is all_data
        contents = [archive.read(name) for name in names]
        assert (b"client error" in contents) is all_data
        assert (b"previous error" in contents) is all_data
        assert (b"cli stack" in contents) is all_data
        assert (b"cli trace" in contents) is all_data
        assert (b"temperature log" in contents) is all_data
        assert (b"session" in contents) is all_data


def test_archive_skips_symlinks_and_reports_read_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preserve linked source data and report partial reads inside a usable support archive.
    """
    scope = paths(tmp_path)
    private = write(tmp_path / "original.py", b"private source")
    scope.settings[0].mkdir()
    (scope.settings[0] / "settings.json").symlink_to(private)
    blocked = write(scope.settings[0] / "pipeline-presets.json", b"blocked")
    original_open = support.os.open

    def denied(path: Path, flags: int, *args: object, **kwargs: object) -> int:
        """
        Simulate an unreadable client settings file without filesystem permission assumptions.
        """
        if Path(path) == blocked:
            raise PermissionError("test read failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(support.os, "open", denied)
    destination = tmp_path / "report.zip"
    _count, warnings = support.create_archive(
        destination,
        scope,
        {"settings": {}},
        all_data=False,
    )
    assert len(warnings) == 2
    assert private.read_bytes() == b"private source"
    with ZipFile(destination) as archive:
        assert b"private source" not in [archive.read(name) for name in archive.namelist()]
        assert json.loads(archive.read("manifest.json"))["warnings"] == warnings


def test_archive_write_failure_preserves_previous_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A failed ZIP leaves the earlier report intact and removes its temporary output.
    """
    destination = write(tmp_path / "report.zip", b"previous report")
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(support, "ZipFile", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        support.create_archive(destination, paths(tmp_path), {"settings": {}}, all_data=False)
    assert destination.read_bytes() == b"previous report"
    assert not list(tmp_path.glob(".based-support-*"))


def test_export_worker_does_not_block_and_reports_result_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Prevent overlapping export/reset work while archive I/O is blocked in a background worker.
    """
    entered, release = Event(), Event()

    def archive(*_args: object, **_kwargs: object) -> tuple[int, list[str]]:
        """
        Hold file work independently from the caller's frame loop.
        """
        entered.set()
        assert release.wait(3)
        return 2, []

    monkeypatch.setattr(support, "create_archive", archive)
    job = support.SupportExport()
    job.start(tmp_path / "report.zip", paths(tmp_path), {}, all_data=False, diagnostics=None)
    assert entered.wait(2) and job.busy and job.poll() is None
    with pytest.raises(ValueError, match="already running"):
        job.start(tmp_path / "other.zip", paths(tmp_path), {}, all_data=False, diagnostics=None)
    release.set()
    job.close()
    result = job.poll()
    assert not job.busy and result is not None and "report.zip" in result
    assert job.poll() is None


def test_debug_preference_queued_logging_and_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Persist debug selection, record verbose details and remove its data during scoped reset.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    scope = paths(tmp_path)
    scope = MaintenancePaths(
        (tmp_path / "config/based-thermals",),
        scope.models,
        scope.runtimes,
        scope.state,
        scope.bytecode,
        scope.artifacts,
        scope.outputs,
    )
    assert not debug_enabled()
    save_debug(True)
    assert debug_enabled()
    logs = SupportLogging(scope.state / "debug-logs")
    logger = logging.getLogger("based_thermals.support_test")
    logs.start(True)
    try:
        logger.debug("debug detail")
        logs.flush()
        assert "debug detail" in (logs.directory / "application.log").read_text()
        logs.set_debug(False)
        logger.debug("should not be retained")
        logger.warning("ordinary warning")
    finally:
        logs.close()
    text = (logs.directory / "application.log").read_text()
    assert "ordinary warning" in text and "should not be retained" not in text
    MaintenancePlan.from_command({"categories": ["settings", "sessions"]}, scope).execute()
    assert not debug_enabled() and not (logs.directory / "application.log").exists()


def test_dynamic_diagnostics_keeps_frame_iterator_and_cli_output(tmp_path: Path) -> None:
    """
    Toggle trace output on an existing diagnostics instance without replacing its generator.
    """
    diagnostics = ViewerDiagnostics(None)
    frames = diagnostics.frames([b"first", b"second"])
    assert next(frames) == b"first"
    path = tmp_path / "trace.jsonl"
    diagnostics.configure(path)
    diagnostics.stream({"received": 1})
    assert next(frames) == b"second"
    diagnostics.configure(None)
    diagnostics.stream({"received": 2})
    diagnostics.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [item["received"] for item in events if item["event"] == "usb_stream"] == [1]


@pytest.mark.usefixtures("viewer")
def test_controller_export_copies_live_settings_without_overwriting_saved_evidence(
    tmp_path: Path,
) -> None:
    """
    Capture unsaved edits at dispatch and lock destructive maintenance until export completes.
    """
    session = desktop.DesktopSession(desktop, [])
    session.pipeline["label"] = "current edit"
    original = desktop.load_settings()
    job = Mock(busy=False)
    session.support_export = job
    session.export_support({"kind": "settings", "path": str(tmp_path / "report.zip")})
    captured = job.start.call_args.args[2]
    assert captured["settings"]["pipeline"]["label"] == "current edit"
    session.pipeline["label"] = "later edit"
    assert captured["settings"]["pipeline"]["label"] == "current edit"
    assert desktop.load_settings() == original
    job.busy = True
    assert session.maintenance_state()["locked"]
    with pytest.raises(ValueError, match="already running"):
        session.export_support({"kind": "all", "path": str(tmp_path / "report.zip")})


def test_support_controls_signals_cancel_suffix_and_busy_state(tmp_path: Path) -> None:
    """
    Verify Qt debug updates do not echo and archive dialogs send only explicit selections.
    """
    result = run_popup(
        """
from PySide6.QtWidgets import QApplication, QFileDialog
from based_thermals.maintenance_window import MaintenanceWindow
app = QApplication([])
messages = []
window = MaintenanceWindow(messages.append)
window.update_state({'debug_logs': True})
assert window.debug.isChecked() and messages == []
window.debug.click()
assert messages == [{'action': 'debug_logs', 'enabled': False}]
QFileDialog.exec = lambda self: QFileDialog.DialogCode.Rejected
window.choose_export('all')
assert len(messages) == 1
QFileDialog.exec = lambda self: QFileDialog.DialogCode.Accepted
QFileDialog.selectedFiles = lambda self: ['/temporary/report.zip']
window.choose_export('settings')
assert messages[-1] == {'action': 'export', 'kind': 'settings', 'path': '/temporary/report.zip'}
window.update_state({'locked': True, 'export_busy': True, 'export_status': 'Creating archive'})
assert not window.export_all.isEnabled() and not window.debug.isEnabled()
assert not window.factory.isEnabled() and window.status.text() == 'Creating archive'
window.choose_export('all')
assert len(messages) == 2
window.close()
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.usefixtures("viewer")
def test_debug_controller_persists_and_preserves_explicit_cli_diagnostics(
    tmp_path: Path,
) -> None:
    """
    Keep explicitly requested diagnostic output when disabling the UI's extra debug verbosity.
    """
    session = desktop.DesktopSession(desktop, [])
    diagnostics = Mock()
    session.diagnostics = diagnostics
    logs = Mock(directory=tmp_path / "logs")
    session.support_logging = logs
    session.args.diagnostics = tmp_path / "explicit.jsonl"
    session.set_debug_logs(True)
    assert session.debug_logs and debug_enabled()
    diagnostics.configure.assert_called_with(session.args.diagnostics)
    session.set_debug_logs(False)
    assert not session.debug_logs and not debug_enabled()
    diagnostics.configure.assert_called_with(session.args.diagnostics)
    with pytest.raises(ValueError, match="enabled or disabled"):
        session.set_debug_logs("yes")
    session.support_export = Mock(busy=True)
    with pytest.raises(ValueError, match="Wait"):
        session.set_debug_logs(True)


@pytest.mark.usefixtures("viewer")
def test_unexpected_viewer_failure_is_logged_and_propagates(tmp_path: Path) -> None:
    """
    Retain traceback evidence while normal shutdown still runs and the original error propagates.
    """
    session = desktop.DesktopSession(desktop, [])
    session.tick = Mock(side_effect=ValueError("simulated client failure"))
    with pytest.raises(ValueError, match="simulated client failure"):
        session.run()
    output = tmp_path / "state/based-thermals/debug-logs/application.log"
    text = output.read_text()
    assert "Unexpected viewer failure" in text and "ValueError: simulated client failure" in text
