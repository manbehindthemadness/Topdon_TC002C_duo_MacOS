"""
Visible drag placement and matching reorder boundaries for pipeline stacks.
"""

import subprocess
import sys

from test_capture_panel import popup_environment


def test_placement_bar_matches_drop_and_respects_endpoints_and_locks() -> None:
    """
    Exercise stack geometry, accepted moves, rejection and marker cleanup offscreen.
    """
    script = """
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QDragLeaveEvent
from PySide6.QtWidgets import QApplication
from topdon_duo.pipeline import node
from topdon_duo.view_window import ViewWindow
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
editor.create_new_pipeline()
editor.insert("software", "gamma", 1)
editor.insert("software", "brightness", 2)
editor.insert("software", "contrast", 3)
app.processEvents()
listing = editor.stacks["software"]
listing.setCurrentRow(1)
moving_id = editor.document["software"][1]["id"]
class DragEvent:
    '''
    Supply drag source and pointer position to the native event handlers.
    '''
    def __init__(self, source: object, position: QPoint) -> None:
        self.origin, self.point = source, position
        self.accepted = False
        self.action = None
    def source(self) -> object:
        '''Return the stack that owns the dragged node.'''
        return self.origin
    def position(self) -> QPointF:
        '''Return the viewport pointer position.'''
        return QPointF(self.point)
    def accept(self) -> None:
        '''Record acceptance.'''
        self.accepted = True
    def ignore(self) -> None:
        '''Record rejection.'''
        self.accepted = False
    def setDropAction(self, action: Qt.DropAction) -> None:
        '''Record the move action.'''
        self.action = action
target_rect = listing.visualItemRect(listing.item(3))
target = QPoint(target_rect.center().x(), target_rect.bottom())
event = DragEvent(listing, target)
listing.dragMoveEvent(event)
assert event.accepted and event.action == Qt.DropAction.MoveAction
assert not listing.placement_bar.isHidden()
assert listing.drop_row(target) == 4
bar = listing.placement_bar.geometry()
output_top = listing.visualItemRect(listing.item(4)).top()
assert bar.top() < output_top and bar.height() == 4
listing.dropEvent(event)
app.processEvents()
assert event.accepted and listing.placement_bar.isHidden()
assert editor.document["software"][3]["id"] == moving_id
assert messages[-1]["document"] == editor.document
# The gaps between cards use the same boundary as their neighboring halves.
listing.setCurrentRow(3)
gap = QPoint(20, listing.visualItemRect(listing.item(2)).top() - 2)
assert listing.drop_row(gap) == 2
event = DragEvent(listing, gap)
listing.dragMoveEvent(event)
assert not listing.placement_bar.isHidden()
listing.dragLeaveEvent(QDragLeaveEvent())
assert listing.placement_bar.isHidden()
for point in (
    listing.visualItemRect(listing.item(0)).topLeft(),
    listing.visualItemRect(listing.item(4)).bottomLeft(),
    listing.visualItemRect(listing.item(3)).center(),
):
    event = DragEvent(listing, point)
    listing.dragMoveEvent(event)
    assert not event.accepted and listing.placement_bar.isHidden()
event = DragEvent(editor.stacks["hardware"], gap)
listing.dragMoveEvent(event)
assert not event.accepted and listing.placement_bar.isHidden()
for row in (0, 4):
    listing.setCurrentRow(row)
    assert listing.drop_row(gap) is None
listing.setCurrentRow(3)
editor.locked = True
before = editor.document.copy()
event = DragEvent(listing, gap)
listing.dragMoveEvent(event)
listing.dropEvent(event)
assert not event.accepted and editor.document == before
editor.locked = False
# Hardware has no fixed endpoint; branch B allows insertion after its last node.
editor.insert("hardware", "brightness", 0)
editor.insert("hardware", "contrast", 1)
hardware = editor.stacks["hardware"]
hardware.setCurrentRow(0)
app.processEvents()
end = QPoint(20, hardware.visualItemRect(hardware.item(1)).bottom() + 10)
event = DragEvent(hardware, end)
hardware.dragMoveEvent(event)
assert event.accepted and not hardware.placement_bar.isHidden()
hardware.dropEvent(event)
app.processEvents()
assert [n["type"] for n in editor.document["hardware"]] == ["contrast", "brightness"]
editor.select_tab(1)
editor.insert("software", "gamma", 1)
editor.insert("software", "brightness", 2)
app.processEvents()
listing.setCurrentRow(1)
end = QPoint(20, listing.visualItemRect(listing.item(2)).bottom() + 10)
event = DragEvent(listing, end)
listing.dragMoveEvent(event)
assert event.accepted and not listing.placement_bar.isHidden()
listing.dropEvent(event)
app.processEvents()
assert [n["type"] for n in editor.document["branches"]["B"]] == ["source", "brightness", "gamma"]
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
