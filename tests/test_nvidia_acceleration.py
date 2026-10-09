"""
Contain optional NVIDIA startup failures and reject unusable CUDA sessions.
"""

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from topdon_duo import nvidia_acceleration as module
from topdon_duo.onnx_upsampling import ONNXRuntime


@pytest.fixture(autouse=True)
def clear_probe_cache() -> Any:
    """
    Keep capability results independent between tests.
    """
    module.nvidia_acceleration.cache_clear()
    yield
    module.nvidia_acceleration.cache_clear()


def test_other_platforms_do_not_load_nvidia_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Avoid native GPU imports and subprocesses on macOS.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    run = Mock(side_effect=AssertionError("unexpected probe"))
    monkeypatch.setattr(module.subprocess, "run", run)
    assert not module.nvidia_acceleration()["available"]
    assert not module._probe()["available"]
    run.assert_not_called()


@pytest.mark.parametrize("usable", [False, True])
def test_probe_requires_successful_inference(
    monkeypatch: pytest.MonkeyPatch,
    usable: bool,
) -> None:
    """
    An advertised provider alone must not enable GPU controls.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider"],
        ),
    )
    infer = Mock(return_value=np.zeros((16, 16), np.uint8))
    if not usable:
        infer.side_effect = RuntimeError("CUDA library missing")
    monkeypatch.setattr(ONNXRuntime, "apply", infer)
    result = module._probe()
    assert result["available"] is usable
    assert result["cuda"] is usable
    assert infer.call_args.args[1:] == ("acnet-legacy-hdn0", "cuda")


def test_cpu_runtime_never_attempts_cuda_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Explain missing GPU runtime without creating a native session.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )
    infer = Mock(side_effect=AssertionError("unexpected inference"))
    monkeypatch.setattr(ONNXRuntime, "apply", infer)
    assert "JetPack-compatible" in module._probe()["reason"]
    infer.assert_not_called()


def test_successful_probe_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Spawn at most one capability subprocess during viewer startup.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    expected = {"available": True, "cuda": True, "reason": "ready"}
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(expected)))
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.nvidia_acceleration() == expected
    assert module.nvidia_acceleration() == expected
    run.assert_called_once()
    assert run.call_args.kwargs["timeout"] == 20


@pytest.mark.parametrize(
    "reply",
    [
        SimpleNamespace(returncode=-11, stdout=""),
        SimpleNamespace(returncode=0, stdout="garbage"),
        SimpleNamespace(returncode=0, stdout="null"),
        SimpleNamespace(returncode=0, stdout='{"available": true}'),
        SimpleNamespace(returncode=0, stdout='{"available": true, "cuda": false, "reason": "bad"}'),
        SimpleNamespace(returncode=0, stdout='{"available": 1, "cuda": true, "reason": "bad"}'),
        subprocess.TimeoutExpired("probe", 20),
        OSError("cannot spawn"),
    ],
)
def test_probe_crashes_and_invalid_replies_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
    reply: Any,
) -> None:
    """
    Keep startup usable after native crashes, timeouts, and malformed JSON.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    run = Mock(side_effect=reply) if isinstance(reply, BaseException) else Mock(return_value=reply)
    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.nvidia_acceleration()
    assert not result["available"] and not result["cuda"]
    assert result["reason"]
