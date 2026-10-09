"""Preview nodes inspect their position without changing or activating pipelines."""

import base64
import subprocess
import sys
from copy import deepcopy
from time import monotonic, sleep

import cv2
import numpy as np
from support.desktop_recording import viewer_fixture
from test_capture_panel import popup_environment
from test_render import frame_with_preview

from topdon_duo.pipeline import default_pipeline, node
from topdon_duo.pipeline_processing import PipelineProcessor, PipelineWorker
from topdon_duo.processing.branch import BranchProcessor


def preview_document():
    document = default_pipeline()
    before, after = node("software", "preview"), node("software", "preview")
    before["expanded"] = after["expanded"] = True
    document["software"] = [
        document["software"][0],
        before,
        node("software", "brightness", amount=20),
        after,
        document["software"][-1],
    ]
    document["software"][0]["params"]["source"] = "raw"
    return document, before, after


def decode(payload):
    return cv2.imdecode(np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_COLOR)


def test_node_snapshots_are_at_their_position_and_preserve_final_image():
    document, before, after = preview_document()
    frame, _ = frame_with_preview()
    processor = PipelineProcessor()
    processor.collect_previews = True
    try:
        image, _ = processor.process(frame, None, document, scale=1)
        first = decode(processor.last_previews[before["id"]])
        second = decode(processor.last_previews[after["id"]])
        assert first.shape == second.shape == (192, 256, 3)
        assert not np.array_equal(first, second)
        assert np.array_equal(image, second)
        no_previews = deepcopy(document)
        no_previews["software"] = [n for n in no_previews["software"] if n["type"] != "preview"]
        unmodified, _ = processor.process(frame, None, no_previews, scale=1)
        assert np.array_equal(image, unmodified)
    finally:
        processor.close()


def test_collapsed_bypassed_and_closed_window_previews_do_no_thumbnail_work(monkeypatch):
    import topdon_duo.processing.branch as processing

    document, before, after = preview_document()
    frame, _ = frame_with_preview()
    processor = PipelineProcessor()

    def forbidden(*_args):
        raise AssertionError("disabled preview generated a thumbnail")

    monkeypatch.setattr(processing, "encode_thumbnail", forbidden)
    try:
        processor.process(frame, None, document)
        assert processor.last_previews == {}
        processor.collect_previews = True
        before["expanded"] = False
        after["bypass"] = True
        processor.process(frame, None, document)
        assert processor.last_previews == {}
    finally:
        processor.close()


def test_expanded_preview_activates_disconnected_tab_and_stops_when_collapsed():
    document = default_pipeline()
    thumbnail = node("software", "preview")
    thumbnail["expanded"] = True
    document["branches"]["B"].append(thumbnail)
    frame, _ = frame_with_preview()
    processor = PipelineProcessor()
    processor.collect_previews = True
    try:
        processor.process(frame, None, document)
        assert set(processor.executors) == {"A", "B"}
        assert thumbnail["id"] in processor.last_previews
        thumbnail["expanded"] = False
        processor.process(frame, None, document)
        assert processor.last_previews == {} and set(processor.executors) == {"A"}
        document["software"].insert(-1, node("software", "combine", tab="B"))
        thumbnail["expanded"] = True
        processor.process(frame, None, document)
        image = decode(processor.last_previews[thumbnail["id"]])
        assert image.shape[0] <= 240 and image.shape[1] <= 320
        assert set(processor.executors) == {"A", "B"}
    finally:
        processor.close()


def test_preview_flags_persist_in_pipeline_preferences(tmp_path, monkeypatch):
    from topdon_duo.settings_preferences import load_settings, save_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    document, before, after = preview_document()
    before["expanded"] = False
    after["bypass"] = True
    save_settings({"pipeline": document})
    assert load_settings()["pipeline"] == document


def test_worker_previews_are_revision_guarded_throttled_and_cleared_on_close(monkeypatch):
    worker = PipelineWorker()
    frame, _ = frame_with_preview()
    document, before, after = preview_document()
    calls = []
    original = worker.processor.process

    def counting(*args):
        calls.append(worker.processor.collect_previews)
        return original(*args)

    worker.processor.process = counting

    def wait_result(revision):
        deadline = monotonic() + 3
        while worker.latest(revision) is None and monotonic() < deadline:
            sleep(0.01)
        assert worker.latest(revision) is not None

    try:
        worker.enable_previews(True)
        worker.submit(frame, None, document, 1, 1, 0, 1)
        wait_result(1)
        assert set(worker.latest_previews(1)) == {before["id"], after["id"]}
        assert worker.latest_previews(2) == {}
        assert set(worker.latest_preview_timings(1)) == {before["id"], after["id"]}
        assert all(value >= 0 for value in worker.latest_preview_timings(1).values())
        assert worker.latest_preview_timings(2) == {}
        with worker.condition:
            worker.last_preview_at = (
                monotonic() + 60
            )  # Deterministically suppress same-revision refresh.
        worker.submit(frame, None, document, 2, 1, 0, 1)
        wait_result(2)  # New revisions still update immediately.
        assert calls[-1] is True and worker.latest_previews(1) == {}
        with worker.condition:
            worker.last_preview_at = monotonic() + 60
        worker.submit(frame, None, document, 2, 1, 0, 1)
        deadline = monotonic() + 3
        while len(calls) < 3 and monotonic() < deadline:
            sleep(0.01)
        assert calls[-1] is False
        worker.enable_previews(False)
        assert worker.latest_previews(2) == {}
    finally:
        worker.close()
    assert worker.latest_previews(2) == {}
    assert worker.latest_preview_timings(2) == {}


def test_preview_node_ui_expands_collapses_and_shows_aspect_preserving_images(tmp_path):
    script = """
import base64, sys
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from topdon_duo.view_window import ViewWindow
from topdon_duo.pipeline import default_pipeline, node
app = QApplication([])
messages = []
window = ViewWindow(messages.append)
window.show()
editor = window.pipeline_editor
editor.insert("software", "preview", 1)
item = editor.document["software"][1]
assert not item["expanded"] and item["type"] == "preview"
editor.widgets[item["id"]][3].click()
assert item["expanded"]
image = QImage(256, 192, QImage.Format_RGB32)
image.fill(Qt.red)
data = QByteArray()
buffer = QBuffer(data)
buffer.open(QIODevice.WriteOnly)
image.save(buffer, "PNG")
payload = bytes(data.toBase64()).decode("ascii")
state = {"pipeline": editor.document, "pipeline_serial": editor.edit_serial,
         "pipeline_previews": {item["id"]: payload}, "pipeline_preview_timings": {item["id"]: 12.34}, "processing_preset_available": True}
editor.update_state(state, False)
app.processEvents()
preview = editor.preview_widgets[item["id"]]
assert not preview.image.isNull()
assert preview.height() == 200 and preview.isVisible()
editor.resize(900, editor.height())
app.processEvents()
pixmap = preview.pixmap()
assert preview.zoomed and pixmap.width() == preview.contentsRect().width()
assert editor.preview_timing_widgets[item["id"]].text() == "12.3 ms"
editor.widgets[item["id"]][3].click()
assert not item["expanded"] and not preview.isVisible()
assert messages[-1]["document"]["software"][1]["expanded"] is False
editor.widgets[item["id"]][3].click()
assert item["expanded"] and preview.isVisible()
frozen = preview.image.toImage()
editor.bypass(item, True)
state.update(pipeline_serial=editor.edit_serial, pipeline_previews={})
editor.update_state(state, False)
assert not preview.image.isNull() and preview.image.toImage() == frozen
# The frozen image survives widget rebuilds while switching tabs.
editor.tab_bar.setCurrentIndex(1)
editor.tab_bar.setCurrentIndex(0)
preview = editor.preview_widgets[item["id"]]
assert preview.image.toImage() == frozen
editor.update_state(state, False)
assert preview.image.toImage() == frozen
assert preview.elapsed_ms == 12.34
assert editor.preview_timing_widgets[item["id"]].text() == "12.3 ms"
editor.bypass(item, False)
state.update(pipeline_serial=editor.edit_serial, pipeline_previews={item["id"]: "invalid png"})
editor.update_state(state, False)
assert preview.image.isNull()
assert editor.preview_timing_widgets[item["id"]].text() == "— ms"
editor.update_state(state, True)
assert not editor.widgets[item["id"]][3].isEnabled()
window.grab().save(sys.argv[1] + "/preview-node.png")
window.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=popup_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_preview_only_dependencies_stop_after_last_preview_closes_and_keep_cache_between_updates():
    document = default_pipeline()
    first, last = node("software", "preview"), node("software", "preview")
    first["expanded"] = last["expanded"] = True
    document["branches"]["B"] += [node("software", "combine", tab="C"), first, last]
    frame, _ = frame_with_preview()
    processor = PipelineProcessor()
    processor.preview_active = processor.collect_previews = True
    try:
        processor.process(frame, None, document)
        assert set(processor.executors) == {"A", "B", "C"}
        cache = processor.branches["B"]
        processor.collect_previews = False
        processor.process(frame, None, document)
        assert (
            processor.branches["B"] is cache
        )  # No teardown/model reload on frames between previews.
        assert processor.last_previews == {}
        first["expanded"] = False
        processor.process(frame, None, document)
        assert set(processor.executors) == {"A", "B", "C"}
        last["expanded"] = False
        processor.process(frame, None, document)
        assert set(processor.executors) == {"A"}
    finally:
        processor.close()


def test_closing_camera_stops_only_preview_branches():
    document = default_pipeline()
    for tab in "BCD":
        thumbnail = node("software", "preview")
        thumbnail["expanded"] = True
        document["branches"][tab].append(thumbnail)
    document["software"].insert(-1, node("software", "combine", tab="B"))
    frame, _ = frame_with_preview()
    processor = PipelineProcessor()
    processor.preview_active = processor.collect_previews = True
    try:
        processor.process(frame, None, document)
        assert set(processor.executors) == set("ABCD")
        processor.preview_active = processor.collect_previews = False
        processor.process(frame, None, document)
        assert set(processor.executors) == {"A", "B"}
        assert processor.last_previews == {}
    finally:
        processor.close()


def test_disconnected_preview_failure_does_not_break_viewer(monkeypatch):
    document = default_pipeline()
    preview = node("software", "preview")
    preview["expanded"] = True
    document["branches"]["B"].append(preview)
    source_id = document["branches"]["B"][0]["id"]
    original = BranchProcessor.process

    def fail_preview(self, *args):
        if args[2]["software"][0]["id"] == source_id:
            raise ValueError("Preview branch failed")
        return original(self, *args)

    monkeypatch.setattr(BranchProcessor, "process", fail_preview)
    processor = PipelineProcessor()
    processor.collect_previews = True
    frame, _ = frame_with_preview()
    try:
        image, _ = processor.process(frame, None, document)
        assert image.shape == (576, 768, 3)
        assert processor.last_previews == {}
        assert processor.last_preview_errors == {preview["id"]: "Preview branch failed"}
    finally:
        processor.close()


def test_preview_failure_propagates_to_preview_consumers_without_blocking_A(monkeypatch):
    document = default_pipeline()
    preview = node("software", "preview")
    preview["expanded"] = True
    document["branches"]["B"] += [node("software", "combine", tab="C"), preview]
    source_id = document["branches"]["C"][0]["id"]
    original = BranchProcessor.process

    def fail_preview(self, *args):
        if args[2]["software"][0]["id"] == source_id:
            raise ValueError("bad input")
        return original(self, *args)

    monkeypatch.setattr(BranchProcessor, "process", fail_preview)
    processor = PipelineProcessor()
    processor.collect_previews = True
    frame, _ = frame_with_preview()
    try:
        image, _ = processor.process(frame, None, document)
        assert image.shape == (576, 768, 3)
        assert "bad input" in processor.last_preview_errors[preview["id"]]
    finally:
        processor.close()


def test_camera_open_and_reopen_collapse_all_previews(viewer, monkeypatch):
    from unittest.mock import Mock

    from topdon_duo import desktop
    from topdon_duo.settings_preferences import save_settings

    document, *_ = preview_document()
    another = node("software", "preview")
    another["expanded"] = True
    document["branches"]["D"].append(another)
    save_settings({"pipeline": document})
    opened, updates = [], []
    panel = Mock(is_open=False)
    panel.poll.side_effect = [[], [{"action": "pipeline", "document": document, "serial": 1}], []]
    panel.update.side_effect = lambda state: updates.append(deepcopy(state["pipeline"]))

    def open_panel(state):
        opened.append(deepcopy(state["pipeline"]))
        panel.is_open = True

    panel.open.side_effect = open_panel
    monkeypatch.setattr(desktop, "ViewPanel", lambda: panel)
    monkeypatch.setattr(desktop.cv2, "getWindowProperty", lambda *_args: 1)
    keys = iter("vvq")

    def key(_delay):
        value = next(keys)
        if len(opened) == 1:
            panel.is_open = False
        return ord(value)

    monkeypatch.setattr(desktop.cv2, "waitKey", key)
    assert desktop.main([]) == 0
    assert len(opened) == 2
    assert any(
        n["expanded"] for state in updates for n in state["software"] if n["type"] == "preview"
    )
    assert all(
        not n["expanded"]
        for state in opened
        for nodes in (state["software"], *state["branches"].values())
        for n in nodes
        if n["type"] == "preview"
    )


def test_preview_right_click_fit_width_and_drag_pan_preserve_view_across_updates():
    script = """
import base64
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPoint, Qt
from PySide6.QtGui import QColor, QContextMenuEvent, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from topdon_duo.pipeline_editor import PipelinePreview
app = QApplication([])
preview = PipelinePreview()
preview.resize(640, 200)
preview.show()
image = QImage(256, 192, QImage.Format_RGB32)
painter = QPainter(image)
for y in range(192):
    painter.setPen(QColor(y, 0, 255-y))
    painter.drawLine(0, y, 255, y)
painter.end()
array = QByteArray()
buffer = QBuffer(array)
buffer.open(QIODevice.WriteOnly)
image.save(buffer, "PNG")
payload = bytes(array.toBase64()).decode("ascii")
preview.show_image(payload, "waiting")
app.processEvents()
assert preview.zoomed and preview.pixmap().width() == preview.contentsRect().width()
QTest.mouseClick(preview, Qt.RightButton)
assert not preview.zoomed
normal = preview.pixmap().size()
assert abs(normal.width() / normal.height() - 4/3) < .02
QTest.mouseClick(preview, Qt.RightButton)
assert preview.zoomed and preview.pixmap().width() == preview.contentsRect().width()
center = preview.pixmap().toImage().pixelColor(10, 10)
QTest.mousePress(preview, Qt.LeftButton, pos=QPoint(300, 100))
QTest.mouseMove(preview, QPoint(300, 150))
QTest.mouseRelease(preview, Qt.LeftButton, pos=QPoint(300, 150))
assert preview.pan.y() < .5 and preview.drag_anchor is None
assert preview.pixmap().toImage().pixelColor(10, 10) != center
# Large drags clamp to the image edge rather than leaving blank space.
QTest.mousePress(preview, Qt.LeftButton, pos=QPoint(300, 20))
QTest.mouseMove(preview, QPoint(300, 199))
QTest.mouseMove(preview, QPoint(300, 1999))
QTest.mouseRelease(preview, Qt.LeftButton, pos=QPoint(300, 1999))
assert 0 <= preview.pan.y() <= 1
assert preview.pixmap().height() == preview.contentsRect().height()
pan = preview.pan.y()
# Every live frame must keep zoom and pan; resizing must also clamp the viewport.
preview.show_image(payload + "\\n", "waiting")  # Invalid payload clears the image safely.
assert preview.image.isNull()
preview.show_image(payload, "waiting")
assert preview.zoomed and preview.pan.y() == pan
preview.resize(780, 200)
app.processEvents()
assert preview.pixmap().width() == preview.contentsRect().width()
context = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(50, 50))
QApplication.sendEvent(preview, context)
assert context.isAccepted()
QTest.mouseClick(preview, Qt.RightButton)
assert not preview.zoomed and preview.pan.y() == .5
assert abs(preview.pixmap().width() / preview.pixmap().height() - 4/3) < .02
preview.close()
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


def test_preview_timings_measure_prefix_work_and_exclude_all_thumbnail_encoding(monkeypatch):
    import topdon_duo.processing.branch as processing

    clock = [0.0]
    original_decode = processing.decode_duo_frame
    original_brightness = processing.map_luminance
    original_thumbnail = processing.encode_thumbnail

    def decoding(*args, **kwargs):
        clock[0] += 0.003
        return original_decode(*args, **kwargs)

    def brightness(*args, **kwargs):
        clock[0] += 0.020
        return original_brightness(*args, **kwargs)

    def thumbnail(*args, **kwargs):
        clock[0] += 0.200
        return original_thumbnail(*args, **kwargs)

    monkeypatch.setattr(processing, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(processing, "decode_duo_frame", decoding)
    monkeypatch.setattr(processing, "map_luminance", brightness)
    monkeypatch.setattr(processing, "encode_thumbnail", thumbnail)
    document, before, after = preview_document()
    frame, _ = frame_with_preview()
    processor = PipelineProcessor()
    processor.collect_previews = True
    try:
        processor.process(frame, None, document, scale=1)
        np.testing.assert_allclose(processor.last_preview_timings[before["id"]], 3, atol=1e-6)
        np.testing.assert_allclose(processor.last_preview_timings[after["id"]], 23, atol=1e-6)
        processor.collect_previews = False
        processor.process(frame, None, document, scale=1)
        assert processor.last_preview_timings == {}
    finally:
        processor.close()


__all__ = ["viewer_fixture"]
