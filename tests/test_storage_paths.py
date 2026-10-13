"""
Keep current application storage distinct from obsolete product directories.
"""

import sys
from pathlib import Path

import pytest
from support.qt_process import run_popup

from based_thermals import onnx_models, runtime_setup, window_preferences


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("xdg", [False, True])
def test_platform_storage_paths(
    platform: str, xdg: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Resolve platform defaults and XDG overrides without touching installed user data.
    """
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for variable in ("BASED_THERMALS_MODEL_DIR", "BASED_THERMALS_RUNTIME_DIR",
                     "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(variable, raising=False)
    if xdg:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    config = (tmp_path / "config" if xdg else tmp_path / "Library/Application Support"
              if platform == "darwin" else tmp_path / ".config")
    cache = (tmp_path / "Library/Caches" if platform == "darwin" else
             tmp_path / "cache" if xdg else tmp_path / ".cache")
    assert window_preferences._path() == config / "based-thermals/main-window.json"
    assert onnx_models.model_path("espcn").parent == cache / "based-thermals/models"
    assert runtime_setup.cache_root() == cache / "based-thermals/runtimes"


def test_qt_imaging_settings_use_application_name(tmp_path: Path) -> None:
    """
    Preserve geometry storage under the current application namespace in isolated Qt.
    """
    result = run_popup(
        '''
import sys
from pathlib import Path
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from based_thermals.view_window import ViewWindow
app = QApplication([])
root = Path(sys.argv[1])
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root))
window = ViewWindow(lambda message: None)
assert window._settings.organizationName() == "based-thermals"
assert Path(window._settings.fileName()) == root / "based-thermals/desktop.ini"
window.close()
''',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
