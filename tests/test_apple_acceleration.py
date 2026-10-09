import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from topdon_duo import apple_acceleration as module


@pytest.fixture(autouse=True)
def reset_cache():
    module.apple_acceleration.cache_clear()
    yield
    module.apple_acceleration.cache_clear()


def test_linux_never_loads_apple_libraries_or_runs_subprocess(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    native = Mock(side_effect=AssertionError("Native Apple probe on Linux"))
    monkeypatch.setattr(module, "_metal_available", native)
    monkeypatch.setattr(module.subprocess, "run", native)
    assert not module.apple_acceleration()["available"]
    assert not module._probe()["available"]
    native.assert_not_called()


@pytest.mark.parametrize(
    "metal,providers,available",
    [
        (False, ["CoreMLExecutionProvider"], False),
        (True, ["CPUExecutionProvider"], False),
        (True, ["CoreMLExecutionProvider"], True),
    ],
)
def test_probe_requires_both_metal_and_coreml(monkeypatch, metal, providers, available):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(module, "_metal_available", lambda: metal)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: providers,
        ),
    )
    assert module._probe()["available"] is available


def test_missing_runtime_reports_install_command(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(module, "_metal_available", lambda: True)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    result = module._probe()
    assert result["metal"] and not result["available"]
    assert "uv sync" in result["reason"]


def test_successful_check_is_cached(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    expected = {"available": True, "metal": True, "coreml": True, "reason": "Ready"}
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(expected)))
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.apple_acceleration() == expected
    assert module.apple_acceleration() == expected
    run.assert_called_once()
    assert run.call_args.kwargs["timeout"] == 8


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(returncode=-11, stdout=""),
        SimpleNamespace(returncode=0, stdout="not JSON"),
        SimpleNamespace(returncode=0, stdout='{"available": true}'),
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"available": True, "metal": False, "coreml": True, "reason": "Invalid"}
            ),
        ),
    ],
)
def test_crashed_or_invalid_probe_preserves_cpu_path(monkeypatch, response):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(module.subprocess, "run", Mock(return_value=response))
    assert not module.apple_acceleration()["available"]


def test_probe_timeout_preserves_cpu_path(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        Mock(
            side_effect=subprocess.TimeoutExpired("probe", 8),
        ),
    )
    assert not module.apple_acceleration()["available"]


def test_editor_enables_apple_node_without_rewriting_pipeline():
    from test_capture_panel import popup_environment

    script = """
from copy import deepcopy
from PySide6.QtWidgets import QApplication
from topdon_duo.pipeline import node
from topdon_duo.view_window import ViewWindow
app = QApplication([])
window = ViewWindow(lambda _: None)
editor = window.pipeline_editor
enhance = node("software", "enhance", model="acnet", backend="coreml")
editor.document["software"].insert(1, enhance)
editor.rebuild()
original = deepcopy(editor.document)
for available in (False, True, False):
    capability = {"available": available, "metal": available, "coreml": available,
                  "reason": "Ready" if available else "No Metal device available"}
    editor.update_state({"pipeline": original, "apple_acceleration": capability}, False)
    controls = editor.widgets[enhance["id"]][1]
    for key in ("backend", "apple_compute"):
        assert controls[key].isHidden() is (not available)
        assert controls[key].input.isEnabled() is available
    assert editor.apple_status.text() == ("Apple acceleration ready · Metal + Core ML" if available else capability["reason"])
    assert editor.document == original
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
