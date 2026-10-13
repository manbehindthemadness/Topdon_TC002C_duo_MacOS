"""
Check automatic conversion with fake process boundaries and verified temporary model caches.
"""

import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from based_thermals import model_exports as exports
from based_thermals.model_exports import run_export


def install_cached_model(path: Path, data: bytes = b"verified model") -> None:
    """
    Write cache/checksum/provenance fixtures without invoking model code or external services.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest())
    path.with_suffix(".json").write_text('{"license":"MIT"}')


@pytest.mark.parametrize("model", exports.EXPORT_MODELS)
def test_preparation_is_isolated_verified_and_reused_offline(
    model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Resolve the installed interpreter, strip native runtime overrides and reuse verified artifacts.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/staged/gpu/runtime")
    monkeypatch.setenv("BASED_THERMALS_ORT_PATH", "/staged/gpu/runtime")
    monkeypatch.setenv("VIRTUAL_ENV", "/viewer/venv")
    monkeypatch.setattr("based_thermals.runtime_setup.install.find_uv", lambda: "/tools/uv")
    path = exports.model_path(model)
    calls: list[list[str]] = []

    def run(command: list[str], environment: dict[str, str], directory: Path) -> None:
        """
        Assert process isolation and publish only a checksum-valid result.
        """
        calls.append(command)
        assert command[:6] == ["/tools/uv", "run", "--no-project", "--no-python-downloads",
                               "--python", sys.executable]
        assert command[6] == "--script" and command[-1] == model
        assert Path(command[7]).is_file()
        assert "PYTHONPATH" not in environment and "VIRTUAL_ENV" not in environment
        assert "BASED_THERMALS_ORT_PATH" not in environment
        assert environment["BASED_THERMALS_MODEL_DIR"] == str(tmp_path.resolve())
        assert Path(environment["BASED_THERMALS_EXPORT_SOURCE_ROOT"]).is_dir()
        assert directory == tmp_path
        install_cached_model(path)

    monkeypatch.setattr(exports, "run_export", run)
    progress: list[str] = []
    assert exports.prepare_model(model, progress.append) == path
    assert progress and len(calls) == 1
    assert exports.prepare_model(model) == path and len(calls) == 1
    assert os.environ["PYTHONPATH"] == "/staged/gpu/runtime"


def test_cached_models_need_no_uv_and_corrupt_cache_is_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Permit offline reuse without installer availability and repair a bad checksum through conversion.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    path = exports.model_path("dncnn-25")
    install_cached_model(path)
    uv = Mock(return_value=None)
    monkeypatch.setattr("based_thermals.runtime_setup.install.find_uv", uv)
    assert exports.prepare_model("dncnn-25") == path
    uv.assert_not_called()
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="needs uv"):
        exports.prepare_model("dncnn-25")
    uv.return_value = "/tools/uv"
    run = Mock(side_effect=lambda *_args: install_cached_model(path))
    monkeypatch.setattr(exports, "run_export", run)
    assert exports.prepare_model("dncnn-25") == path
    run.assert_called_once()


def test_successful_process_must_produce_verified_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Do not declare readiness merely because an exporter exits successfully.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("based_thermals.runtime_setup.install.find_uv", lambda: "/tools/uv")
    monkeypatch.setattr(exports, "run_export", Mock())
    with pytest.raises(ValueError, match="Model not installed"):
        exports.prepare_model("ffdnet-gray")


def test_waiting_for_another_process_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Bound preparation when a different viewer holds the cache lock indefinitely.
    """
    monkeypatch.setenv("BASED_THERMALS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(exports.fcntl, "flock", Mock(side_effect=BlockingIOError))
    monkeypatch.setattr(exports.time, "monotonic", Mock(side_effect=[0, exports.EXPORT_TIMEOUT]))
    with pytest.raises(ValueError, match="waiting for another viewer"):
        exports.prepare_model("dncnn-25")


@pytest.mark.parametrize("outcome", [0, 1, "timeout"])
def test_conversion_process_failures_and_timeout_cleanup(
    outcome: int | str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Capture bounded diagnostics and terminate the dedicated process group after a timeout.
    """
    communicate = Mock(return_value=("converter output", None))
    if outcome == "timeout":
        communicate.side_effect = [subprocess.TimeoutExpired(["uv"], 900), ("stopped", None)]
    process = SimpleNamespace(communicate=communicate, returncode=outcome, pid=12345)
    popen = Mock(return_value=process)
    kill = Mock()
    monkeypatch.setattr(exports.subprocess, "Popen", popen)
    monkeypatch.setattr(exports.os, "killpg", kill)
    if outcome == 0:
        run_export(["uv"], {}, tmp_path)
    else:
        with pytest.raises(ValueError, match="timed out" if outcome == "timeout" else "converter output"):
            run_export(["uv"], {}, tmp_path)
    assert popen.call_args.kwargs["start_new_session"]
    assert communicate.call_args_list[0].kwargs == {"timeout": exports.EXPORT_TIMEOUT}
    if outcome == "timeout":
        kill.assert_called_once_with(process.pid, signal.SIGKILL)
        assert communicate.call_count == 2
    else:
        kill.assert_not_called()


def test_unknown_models_never_launch_conversion(tmp_path: Path) -> None:
    """
    Restrict code conversion to application-owned model recipes.
    """
    with pytest.raises(ValueError, match="not supported"):
        exports.prepare_model("custom-something")
