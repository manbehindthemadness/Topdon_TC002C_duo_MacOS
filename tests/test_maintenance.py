"""
Maintenance plans preserve source resources and run after every app writer has stopped.
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from support.desktop_recording import viewer_fixture  # noqa: F401
from support.qt_process import run_popup

from based_thermals import desktop
from based_thermals.maintenance import MaintenancePaths, MaintenancePlan
from based_thermals.maintenance.purge import CATEGORIES
from based_thermals.model_downloads import ModelDownloads


def locations(root: Path) -> MaintenancePaths:
    """
    Scope every destructive test exclusively to its temporary directory.
    """
    return MaintenancePaths(
        (root / "config",),
        root / "models",
        root / "runtimes",
        root / "state",
        (root / "source",),
        (root / ":memory:.ses",),
        (root / "captures",),
    )


def write(path: Path, data: bytes = b"generated") -> Path:
    """
    Create a fixture file without touching any application or host storage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_factory_reset_preserves_node_and_backend_sources_and_shipped_assets(
    tmp_path: Path,
) -> None:
    """
    Delete only app data/bytecode and generated outputs, preserving source and unrelated files.
    """
    paths = locations(tmp_path)
    deleted = [
        write(paths.settings[0] / "settings.json", b"{}"),
        write(paths.settings[0] / "pipeline-presets.json", b"{}"),
        write(paths.settings[0] / "desktop.ini"),
        write(paths.settings[0] / ("camera-profiles/" + "b" * 24 + ".json"), b"{}"),
        write(paths.state / ("camera-baselines/" + "a" * 32 + ".json")),
        write(paths.runtimes / "v1-cpython-312-linux-aarch64-cuda13/ready/ort.so"),
        write(paths.bytecode[0] / "camera_backends/added/__pycache__/camera.cpython-312.pyc"),
        write(paths.artifacts[0]),
        write(paths.outputs[0] / "based-thermals-20261014-093558-000100.png"),
        write(paths.outputs[0] / "saved.pipeline.json", b"{}"),
    ]
    preserved = [
        write(paths.bytecode[0] / "custom_nodes/mine/__init__.py", b"CONFIG_JSON = {}"),
        write(paths.bytecode[0] / "custom_nodes/mine/config.json", b"{}"),
        write(paths.bytecode[0] / "camera_backends/added/camera.py", b"backend"),
        write(paths.bytecode[0] / "models/bundled.onnx"),
        write(paths.bytecode[0] / ".venv/__pycache__/dependency.pyc"),
        write(paths.bytecode[0] / ".agents/skills/skill.py"),
        write(paths.settings[0] / "custom_nodes/original/__init__.py"),
        write(paths.runtimes / "user-added-camera-backend/camera.py"),
        write(paths.outputs[0] / "custom_node/__init__.py"),
        write(paths.outputs[0] / "camera_backend.py"),
        write(paths.outputs[0] / "personal-photo.png"),
    ]
    plan = MaintenancePlan.from_command({"categories": list(CATEGORIES)}, paths)
    count, size = plan.execute()
    assert count == len(deleted) and size > 0
    assert all(not path.exists() for path in deleted)
    assert all(path.exists() for path in preserved)
    assert plan.execute() == (0, 0)


def test_cleanup_does_not_follow_directory_or_file_symlinks(tmp_path: Path) -> None:
    """
    Protect original files reached through linked bytecode, runtime or output entries.
    """
    paths = locations(tmp_path / "app")
    outside = write(tmp_path / "originals/backend.py", b"keep")
    paths.bytecode[0].mkdir(parents=True)
    (paths.bytecode[0] / "linked").symlink_to(outside.parent, target_is_directory=True)
    paths.runtimes.mkdir(parents=True)
    (paths.runtimes / "v1-cpython-312-linux-aarch64-cuda13").symlink_to(
        outside.parent, target_is_directory=True
    )
    paths.outputs[0].mkdir(parents=True)
    (paths.outputs[0] / "saved.pipeline.json").symlink_to(outside)
    plan = MaintenancePlan.from_command({"categories": ["bytecode", "runtimes", "outputs"]}, paths)
    plan.execute()
    assert outside.read_bytes() == b"keep"
    assert (paths.outputs[0] / "saved.pipeline.json").is_symlink()


@pytest.mark.parametrize("categories", [None, [], "settings", ["unknown"], [True], [{}]])
def test_invalid_plan_cannot_choose_arbitrary_paths(tmp_path: Path, categories: object) -> None:
    """
    Reject malformed UI data before touching any filesystem targets.
    """
    with pytest.raises(ValueError):
        MaintenancePlan.from_command(
            {"categories": categories, "path": str(tmp_path)}, locations(tmp_path)
        )


def test_confirmations_name_user_data_loss_and_default_to_cancel(tmp_path: Path) -> None:
    """
    Exercise actual Qt factory/settings warnings and graph-toolbar position/portrait movement.
    """
    result = run_popup(
        """
from PySide6.QtWidgets import QApplication, QMessageBox
from based_thermals.maintenance_window import MaintenanceWindow
from based_thermals.viewer_ui.toolbar import ViewerToolbar
app = QApplication([])
messages, dialogs = [], []
window = MaintenanceWindow(messages.append)
window.update_state({'locked': False, 'paths': {'settings': ['/isolated/config']}})
def confirmation(*args):
    dialogs.append(args)
    return QMessageBox.StandardButton.Cancel
QMessageBox.warning = confirmation
window.choices['settings'].setChecked(True)
window.clear_selected()
assert not messages
text = dialogs[-1][2]
assert 'saved custom pipelines' in text and 'custom-node entries' in text
assert 'camera backend source files are preserved' in text
assert '/isolated/config' in text
assert dialogs[-1][-1] == QMessageBox.StandardButton.Cancel
window.factory_reset()
assert 'permanently deleted' in dialogs[-1][2]
QMessageBox.warning = lambda *args: QMessageBox.StandardButton.Yes
window.factory_reset()
assert messages[-1]['action'] == 'cleanup'
assert set(messages[-1]['categories']) == set(window.choices)
window.update_state({'locked': True})
count = len(messages)
window.factory_reset()
assert len(messages) == count
bar = ViewerToolbar(messages.append)
bar.update_state({'show_graph': True})
assert bar.graph_widgets[-1].defaultWidget() is bar.buttons['maintenance']
bar.buttons['maintenance'].click()
assert messages[-1] == {'action': 'maintenance'}
bar.update_state({'show_graph': True, 'viewer_layout': 'portrait'})
assert bar.graph_toolbar.actions()[-1].defaultWidget() is bar.buttons['maintenance']
window.close()
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_cleanup_runs_after_shutdown_and_recreates_fresh_settings(
    viewer: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Stop every writer before cleanup and start the next session from the first-install preset.
    """
    paths = locations(tmp_path)
    paths = MaintenancePaths(
        (tmp_path / "config/based-thermals",),
        paths.models,
        paths.runtimes,
        paths.state,
        paths.bytecode,
        paths.artifacts,
        paths.outputs,
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    session = desktop.DesktopSession(desktop, [])
    session.maintenance_paths = paths
    view_panel = Mock()
    spots_panel = Mock()
    session.view_panel = view_panel
    session.spots_panel = spots_panel
    panel = Mock()
    panel.poll.return_value = [{"action": "cleanup", "categories": ["settings"]}]
    session.maintenance_panel = panel
    original = cast(Callable[[MaintenancePlan], tuple[int, int]], MaintenancePlan.execute)
    deleted = []

    def execute(plan: MaintenancePlan) -> tuple[int, int]:
        """
        Assert shutdown order at the actual deletion boundary.
        """
        assert viewer.camera.close.called and view_panel.close.called
        assert panel.close.called and spots_panel.close.called
        deleted.append(True)
        result = original(plan)
        return result

    monkeypatch.setattr(MaintenancePlan, "execute", execute)
    assert session.run() == 0
    assert deleted and not desktop.load_settings()
    next_session = desktop.DesktopSession(desktop, [])
    assert next_session.pipeline == desktop.predefined_presets()["Detail Enhanced CPU Upscale"]
    next_session.graphs.close()


@pytest.mark.usefixtures("viewer")
def test_reset_refuses_deletion_when_downloads_cannot_stop() -> None:
    """
    Fail explicitly rather than removing cache files underneath a live installer.
    """
    session = desktop.DesktopSession(desktop, [])
    plan = Mock()
    session.maintenance_plan = plan
    api = SimpleNamespace(**vars(desktop))
    api.MODEL_DOWNLOADS = Mock(stop=Mock(side_effect=TimeoutError("installer active")))
    session.api = api
    assert session.finish_maintenance() == 1
    plan.execute.assert_not_called()


def test_download_shutdown_cancels_progress_and_joins_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Cancel active progress, block new requests and reopen only after installers have exited.
    """
    from threading import Event

    from based_thermals import model_downloads

    manager = ModelDownloads()
    entered, proceed = Event(), Event()

    def download(_model: str, *, progress: Callable[[int, int], None]) -> Path:
        """
        Hold a fake download until maintenance sets the cancellation event.
        """
        entered.set()
        assert proceed.wait(3)
        progress(1, 1)
        return tmp_path / "unused.onnx"

    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(model_downloads, "download_model", download)
    # noinspection PyUnresolvedReferences
    monkeypatch.setattr(
        model_downloads, "model_path", lambda *args, **kwargs: tmp_path / "missing.onnx"
    )
    assert manager.request("espcn") is False
    assert entered.wait(2)
    manager.stopping.set()
    proceed.set()
    manager.stop(timeout=3)
    assert not manager.threads
    assert manager.request("espcn") is False
    manager.begin()
    assert not manager.stopping.is_set() and not manager.jobs


def test_full_model_reset_removes_generated_exports_but_keeps_registered_originals(
    tmp_path: Path,
) -> None:
    """
    Clear converter output/provenance and registrations while retaining selected original files.
    """
    import json

    from based_thermals.onnx_models import MODELS, model_path

    paths = locations(tmp_path)
    name = next(name for name, spec in MODELS.items() if spec.get("local_export"))
    export = paths.models / model_path(name).name
    generated = [
        write(export),
        write(export.with_suffix(".sha256")),
        write(export.with_suffix(".json")),
        write(paths.models / f".{name}.export.lock"),
    ]
    original = write(paths.models / model_path("espcn").name, b"original local model")
    registry = write(paths.models / "custom-models.json", json.dumps([str(original)]).encode())
    plan = MaintenancePlan.from_command({"categories": ["models", "settings"]}, paths)
    plan.execute()
    assert all(not path.exists() for path in generated) and not registry.exists()
    assert original.read_bytes() == b"original local model"


def test_settings_cleanup_removes_model_registration_without_deleting_weights(
    tmp_path: Path,
) -> None:
    """
    Treat the local-model index as settings rather than deleting the registered model itself.
    """
    paths = locations(tmp_path)
    model = write(paths.models / "personal.onnx", b"keep")
    registry = write(paths.models / "custom-models.json", b"[]")
    MaintenancePlan.from_command({"categories": ["settings"]}, paths).execute()
    assert not registry.exists() and model.read_bytes() == b"keep"


@pytest.mark.parametrize("host", ["linux", "darwin"])
def test_discovery_uses_platform_roots_and_explicit_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    """
    Resolve Linux/Spark and macOS directories without assuming one host's storage layout.
    """
    import sys

    monkeypatch.setattr(sys, "platform", host)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("BASED_THERMALS_RUNTIME_DIR", str(tmp_path / "runtime"))
    paths = MaintenancePaths.discover()
    assert paths.settings[0] == tmp_path / "config/based-thermals"
    assert paths.models == tmp_path / "models" and paths.runtimes == tmp_path / "runtime"
    assert paths.state == tmp_path / "state/based-thermals"
    if host == "darwin":
        assert tmp_path / "home/Library/Preferences/based-thermals" in paths.settings


def test_partial_cleanup_failure_is_reported_and_preserves_unremoved_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Report partial deletion explicitly rather than labeling it a complete factory reset.
    """
    from based_thermals.maintenance import purge

    paths = locations(tmp_path)
    kept = write(paths.settings[0] / "settings.json", b"locked")
    removed = write(paths.settings[0] / "pipeline-presets.json", b"{}")
    original = purge.remove_target

    def remove(path: Path) -> tuple[int, int]:
        """
        Simulate one protected preferences file while other selected files can be removed.
        """
        if path == kept:
            raise PermissionError("simulated permission denial")
        result = original(path)
        return result

    monkeypatch.setattr(purge, "remove_target", remove)
    with pytest.raises(OSError, match="some items may already be removed"):
        MaintenancePlan.from_command({"categories": ["settings"]}, paths).execute()
    assert kept.read_bytes() == b"locked" and not removed.exists()


@pytest.mark.usefixtures("viewer")
def test_closed_maintenance_panel_does_not_build_state_every_frame() -> None:
    """
    Keep administrative path/state work out of the normal frame loop when the popup is closed.
    """
    session = desktop.DesktopSession(desktop, [])
    session.maintenance_panel = Mock(is_open=False)
    session.maintenance_panel.poll.return_value = []
    state = Mock(side_effect=AssertionError("No closed-popup state computation"))
    session.maintenance_state = state
    assert session.poll_maintenance_commands()
    state.assert_not_called()


@pytest.mark.usefixtures("viewer")
def test_open_refreshes_output_locations_before_displaying_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Include folders selected during the current session in the next maintenance confirmation.
    """
    session = desktop.DesktopSession(desktop, [])
    paths = locations(tmp_path)
    monkeypatch.setattr(MaintenancePaths, "discover", lambda output: paths)
    panel = Mock()
    session.maintenance_panel = panel
    session.open_maintenance()
    assert session.maintenance_paths is paths
    assert panel.open.call_args.args[0]["paths"]["outputs"] == [str(paths.outputs[0])]
