import json
from unittest.mock import Mock

from topdon_duo import viewer_diagnostics


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_diagnostics_reports_slow_phase_and_measurement_transitions(tmp_path, monkeypatch):
    thread = Mock()
    monkeypatch.setattr(viewer_diagnostics.threading, "Thread", lambda **_kwargs: thread)
    path = tmp_path / "ui.jsonl"
    diagnostics = viewer_diagnostics.ViewerDiagnostics(path)
    try:
        diagnostics.stage("render")
        diagnostics._since -= 3
        diagnostics.stage("waitKey")
        diagnostics.measurements("Auto calibrating")
        diagnostics.measurements("Auto calibrating")
        diagnostics.measurements("")
        assert list(diagnostics.frames([b"first", b"second"])) == [b"first", b"second"]
    finally:
        diagnostics.close()
    rows = records(path)
    slow = [r for r in rows if r["event"] == "slow_stage"]
    assert slow[0]["stage"] == "render"
    assert slow[0]["seconds"] >= 3
    assert [r["status"] for r in rows if r["event"] == "measurement_status"] == [
        "Auto calibrating", "valid"
    ]
    assert rows[-1]["event"] == "stop"
    assert rows[-1]["frames"] == 2
    thread.join.assert_called_once()


def test_stall_writes_thread_stack_once_for_same_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer_diagnostics.threading, "Thread", lambda **_kwargs: Mock())
    path = tmp_path / "ui.jsonl"
    diagnostics = viewer_diagnostics.ViewerDiagnostics(path)
    try:
        diagnostics.stage("camera_wait")
        diagnostics._since -= 3
        diagnostics._stop.wait = Mock(side_effect=[False, False, True])
        diagnostics._watch()
    finally:
        diagnostics.close()
    assert len([r for r in records(path) if r["event"] == "stall"]) == 1
    stacks = path.with_suffix(".stacks.log").read_text()
    assert "stage=camera_wait" in stacks
    assert "test_stall_writes_thread_stack_once_for_same_phase" in stacks


def test_disabled_diagnostics_passes_frames_through_without_files():
    diagnostics = viewer_diagnostics.ViewerDiagnostics(None)
    diagnostics.stage("render")
    diagnostics.measurements("")
    assert list(diagnostics.frames([b"frame"])) == [b"frame"]
    diagnostics.close()
    assert diagnostics._thread is None
