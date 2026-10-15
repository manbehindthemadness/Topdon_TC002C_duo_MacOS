"""
Imaging uses sufficient window width instead of horizontal scrolling or clipped controls.
"""

from pathlib import Path

import pytest
from support.qt_process import run_popup


@pytest.mark.parametrize("saved_width", [None, 520, 700, 1200])
def test_imaging_width_fits_content_on_open_and_after_expansion(
    tmp_path: Path, saved_width: int | None,
) -> None:
    """
    Cover initial defaults, narrow/wide saved sizes, expanding tools and real pipeline controls.
    """
    saved_width_text = "None" if saved_width is None else str(saved_width)
    result = run_popup('''
from copy import deepcopy
from PySide6.QtCore import QSettings, QSize
from PySide6.QtWidgets import QApplication, QScrollArea
from based_thermals.view_window import ViewWindow

app = QApplication([])
settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                     "based-thermals", "desktop")
saved_width = ''' + saved_width_text + '''
if saved_width is not None:
    settings.setValue("camera/window_size", QSize(saved_width, 650))
settings.sync()
window = ViewWindow(lambda _message: None)
window.show()
scroll = window.findChild(QScrollArea)

def check_width() -> None:
    for _ in range(4):
        app.processEvents()
    assert scroll.horizontalScrollBar().maximum() == 0, (
        window.width(), scroll.viewport().width(), scroll.widget().minimumSizeHint().width())
    assert scroll.widget().width() <= scroll.viewport().width()
    assert scroll.widget().width() >= scroll.widget().minimumSizeHint().width()

check_width()
if saved_width == 1200:
    assert window.width() == 1200
window.environment_button.click()
window.calibrations_button.click()
check_width()
assert scroll.verticalScrollBar().maximum() > 0
editor = window.pipeline_editor
for preset in editor.presets.values():
    editor.document = deepcopy(preset)
    for nodes in (editor.document["hardware"], editor.document["software"]):
        for item in nodes:
            item["expanded"] = True
    editor.rebuild()
    check_width()
window.environment_button.click()
window.calibrations_button.click()
check_width()
window.close()
reopened = ViewWindow(lambda _message: None)
reopened.show()
for _ in range(4):
    app.processEvents()
assert reopened.findChild(QScrollArea).horizontalScrollBar().maximum() == 0
reopened.close()
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
