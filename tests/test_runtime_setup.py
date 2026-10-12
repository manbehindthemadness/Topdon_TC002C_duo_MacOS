"""
Exercise startup provisioning using fake drivers, installers and runtime probes only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from topdon_duo import runtime_setup as setup
from topdon_duo.runtime_setup import install as installer
from topdon_duo.runtime_setup import probes


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """
    Provide a writable runtime cache with no access to real native libraries or downloads.
    """
    monkeypatch.setattr(setup, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(setup, "find_uv", lambda: "/fake/uv")
    monkeypatch.setattr(setup.sys, "platform", "linux")
    monkeypatch.setattr(setup.platform, "machine", lambda: "aarch64")
    install = Mock(return_value=True)
    query = Mock(return_value={"available": False, "reason": "missing provider"})
    activate = Mock()
    monkeypatch.setattr(setup, "install_packages", install)
    monkeypatch.setattr(setup, "query", query)
    monkeypatch.setattr(setup, "activate", activate)
    return SimpleNamespace(root=tmp_path, install=install, query=query, activate=activate)


@pytest.mark.parametrize("system,machine,version,expected", [
    ("darwin", "arm64", 0, ["coreml", "cpu"]),
    ("linux", "x86_64", 13000, ["cuda13", "cuda12", "cpu"]),
    ("linux", "aarch64", 13020, ["cuda13", "cuda12", "cpu"]),
    ("linux", "amd64", 12080, ["cuda12", "cpu"]),
    ("linux", "arm64", 12000, ["cuda12", "cpu"]),
    ("linux", "x86_64", 11080, ["cpu"]),
    ("linux", "x86_64", 0, ["cpu"]),
    ("linux", "riscv64", 13000, ["cpu"]),
])
def test_host_selection(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, version: int, expected: list[str],
) -> None:
    """
    Cover ordinary NVIDIA and ARM hosts without JetPack-specific identification.
    """
    monkeypatch.setattr(setup.sys, "platform", system)
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    query = Mock(return_value={"cuda": version, "available": True})
    monkeypatch.setattr(setup, "query", query)
    assert setup.candidates() == expected
    if machine == "riscv64":
        query.assert_not_called()


@pytest.mark.parametrize("selection", ["cpu", "coreml", "cuda12", "cuda13"])
def test_healthy_environment_requires_no_download(isolated: SimpleNamespace, selection: str) -> None:
    """
    Preserve a working runtime, including compatible vendor builds.
    """
    isolated.query.return_value["available"] = True
    assert setup.prepare(selection, retry=False)
    isolated.install.assert_not_called()
    isolated.activate.assert_not_called()


@pytest.mark.parametrize("selection", ["cpu", "coreml", "cuda13"])
def test_verified_runtime_reused_after_uv_sync(isolated: SimpleNamespace, selection: str) -> None:
    """
    Publish only verified packages and prefer the cached runtime over a replaced project dependency.
    """
    isolated.query.side_effect = [
        {"available": False, "reason": "CPU-only project install"},
        {"available": True, "reason": "verified"},
    ]
    assert setup.prepare(selection, retry=False)
    ready = isolated.activate.call_args.args[0]
    (ready / "onnxruntime").mkdir()
    (ready / "onnxruntime" / "__init__.py").touch()
    assert ready.is_dir()
    isolated.query.side_effect = [
        {"available": False, "reason": "uv sync restored CPU"},
        {"available": True, "reason": "verified cache"},
    ]
    isolated.install.reset_mock()
    assert setup.prepare(selection, retry=False)
    isolated.install.assert_not_called()
    assert isolated.query.call_args.args == (setup.backend(selection), ready)
    assert isolated.activate.call_args.args == (ready,)


def test_missing_system_cuda_downloads_declared_libraries(isolated: SimpleNamespace) -> None:
    """
    Add only CUDA/cuDNN dependencies after the system-library inference check fails.
    """
    def staged_wheel(_uv: str, path: Path, _requirements: list[str], **_kwargs: bool) -> bool:
        """
        Supply GPU extras metadata as a fake wheel installer would.
        """
        metadata = path / "onnxruntime_gpu-1.30.0.dist-info"
        metadata.mkdir(exist_ok=True)
        (metadata / "METADATA").write_text(
            'Requires-Dist: numpy>=1.21\n'
            'Requires-Dist: nvidia-cuda-runtime~=13.0; extra == "cuda"\n'
            'Requires-Dist: nvidia-cudnn-cu13~=9.0; extra == "cudnn"\n', encoding="utf-8",
        )
        return True

    isolated.install.side_effect = staged_wheel
    isolated.query.side_effect = [
        {"available": False, "reason": "CPU-only runtime"},
        {"available": False, "reason": "missing cuDNN"},
        {"available": True, "reason": "verified CUDA"},
    ]
    assert setup.prepare("cuda13", retry=False)
    extra = isolated.install.call_args
    assert extra.args[2] == ["nvidia-cuda-runtime~=13.0", "nvidia-cudnn-cu13~=9.0"]
    assert extra.kwargs == {"dependencies": True}


def test_failed_install_preserves_environment_and_retries(isolated: SimpleNamespace) -> None:
    """
    Remove incomplete staging, limit retry loops, and allow an explicit retry.
    """
    isolated.install.return_value = False
    assert not setup.prepare("coreml", retry=False)
    isolated.activate.assert_not_called()
    assert not list(isolated.root.glob("*/runtime-*"))
    assert not setup.prepare("coreml", retry=False)
    assert isolated.install.call_count == 1
    assert not setup.prepare("coreml", retry=True)
    assert isolated.install.call_count == 2


def test_failed_inference_never_activates_runtime(isolated: SimpleNamespace) -> None:
    """
    A resolved wheel is insufficient; fail native inference without changing imports.
    """
    assert not setup.prepare("coreml", retry=False)
    isolated.activate.assert_not_called()
    assert not list(isolated.root.glob("*/runtime-*"))


def test_missing_uv_leaves_environment_intact(
    isolated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Leave setup optional when a packaged deployment has no installer.
    """
    monkeypatch.setattr(setup, "find_uv", lambda: None)
    assert not setup.prepare("cuda13", retry=False)
    isolated.install.assert_not_called()


@pytest.mark.parametrize("name", [None, 12, "../runtime-outside", "/runtime-outside", "unknown"])
def test_untrusted_cache_pointer_rejected(tmp_path: Path, name: object) -> None:
    """
    Cache metadata must not inject an arbitrary Python import path.
    """
    assert setup.cached_path(tmp_path, {"ready": name}) is None


def test_symlink_cache_pointer_rejected(tmp_path: Path) -> None:
    """
    Keep cached runtime generations within their selected cache directory.
    """
    target = tmp_path / "outside"
    target.mkdir()
    (tmp_path / "runtime-link").symlink_to(target, target_is_directory=True)
    assert setup.cached_path(tmp_path, {"ready": "runtime-link"}) is None


def test_parallel_startup_does_not_install_twice(isolated: SimpleNamespace) -> None:
    """
    Use a nonblocking process lock while another launch provisions the same backend.
    """
    import fcntl

    tag = f"v1-{setup.sys.implementation.cache_tag}-linux-aarch64-cuda13"
    root = isolated.root / tag
    root.mkdir()
    with (root / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not setup.prepare("cuda13", retry=False)
    isolated.install.assert_not_called()


def test_interrupted_download_is_cleaned_on_retry(isolated: SimpleNamespace) -> None:
    """
    Reclaim abandoned staging files after a killed installer without touching verified generations.
    """
    tag = f"v1-{setup.sys.implementation.cache_tag}-linux-aarch64-coreml"
    abandoned = isolated.root / tag / "staging-interrupted"
    abandoned.mkdir(parents=True)
    (abandoned / "download-partial").touch()
    isolated.install.return_value = False
    assert not setup.prepare("coreml", retry=True)
    assert not abandoned.exists()
    assert not list(isolated.root.glob("*/staging-*"))


def test_activation_propagates_to_subprocesses_and_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Select the same runtime in the parent, spawned workers and capability probes.
    """
    monkeypatch.setattr(setup.sys, "path", ["/old/runtime", "/project"])
    monkeypatch.setenv("PYTHONPATH", "/old/runtime:/project")
    monkeypatch.setenv("BASED_THERMALS_ORT_PATH", "/old/runtime")
    monkeypatch.setattr(setup, "apple_acceleration", Mock())
    monkeypatch.setattr(setup, "nvidia_acceleration", Mock())
    setup.activate(tmp_path)
    assert setup.sys.path == [str(tmp_path), "/project"]
    assert setup.os.environ["PYTHONPATH"] == f"{tmp_path}:/project"
    assert setup.os.environ["BASED_THERMALS_ORT_PATH"] == str(tmp_path)
    assert probes.environment()["PYTHONPATH"] == "/project"
    setup.apple_acceleration.cache_clear.assert_called_once()
    setup.nvidia_acceleration.cache_clear.assert_called_once()


def test_setup_falls_back_through_supported_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preserve CPU operation after missing wheels, native failures, or read-only cache errors.
    """
    monkeypatch.delenv("BASED_THERMALS_RUNTIME_SETUP")
    monkeypatch.delitem(setup.sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(setup, "candidates", lambda: ["cuda13", "cuda12", "cpu"])
    prepare = Mock(side_effect=[PermissionError("read-only"), False, True])
    monkeypatch.setattr(setup, "prepare", prepare)
    setup.ensure_runtime()
    assert [call.args[0] for call in prepare.call_args_list] == ["cuda13", "cuda12", "cpu"]


def test_disabled_setup_never_discovers_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Support explicitly offline deployments and deterministic application tests.
    """
    discover = Mock()
    monkeypatch.setattr(setup, "candidates", discover)
    setup.ensure_runtime()
    discover.assert_not_called()


@pytest.mark.parametrize("reply", [
    SimpleNamespace(returncode=-11, stdout="", stderr="native crash"),
    SimpleNamespace(returncode=0, stdout="null", stderr=""),
    SimpleNamespace(returncode=0, stdout='{"available": 1, "reason": "bad"}', stderr=""),
    subprocess.TimeoutExpired("probe", 45),
])
def test_native_probe_failures_are_contained(
    monkeypatch: pytest.MonkeyPatch, reply: object,
) -> None:
    """
    Invalid native probes cannot trigger activation or escape into acquisition.
    """
    def run(command: list[str], **_kwargs: Any) -> object:
        """
        Deliver malformed replies through the same file boundary as the native probe.
        """
        if isinstance(reply, BaseException):
            raise reply
        response = cast(SimpleNamespace, reply)
        Path(command[-1]).write_text(response.stdout, encoding="utf-8")
        return reply

    monkeypatch.setattr(probes.subprocess, "run", run)
    assert probes.query("coreml")["available"] is False


def test_native_stdout_and_teardown_cannot_corrupt_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preserve a valid capability reply despite buffered C output and process-exit chatter.
    """
    actual_run = subprocess.run
    script = """
import atexit
import ctypes
import os
import sys
from pathlib import Path
from topdon_duo.runtime_setup import probes
os.write(1, b'native startup chatter\\n')
ctypes.CDLL(None).printf(b'buffered native stdout\\n')
atexit.register(lambda: os.write(1, b'native teardown chatter\\n'))
probes.runtime = lambda _: {'available': True, 'reason': 'CoreML verified'}
probes.write_result('coreml', Path(sys.argv[1]))
"""

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """
        Replace live GPU inference with a real child process producing native stdout noise.
        """
        result = actual_run(
            [sys.executable, "-c", script, command[-1]], env=kwargs["env"],
            capture_output=True, text=True, check=False, timeout=45,
        )
        assert "buffered native stdout" in result.stdout
        assert "native teardown chatter" in result.stdout
        return result

    monkeypatch.setattr(probes.subprocess, "run", run)
    assert probes.query("coreml") == {"available": True, "reason": "CoreML verified"}


def test_target_installer_keeps_venv_packages_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Never uninstall or overwrite uv's shared CPU/GPU import files during automatic setup.
    """
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(installer.subprocess, "run", run)
    assert installer.install("/fake/uv", tmp_path, ["onnxruntime>=1.23"])
    command = run.call_args.args[0]
    assert "--target" in command and str(tmp_path) in command
    assert "--no-deps" in command
    assert "uninstall" not in command
    assert run.call_args.kwargs["timeout"] == 600


def test_malformed_state_is_ignored(tmp_path: Path) -> None:
    """
    Recover from incomplete or corrupt state writes rather than breaking startup.
    """
    path = tmp_path / "state.json"
    for value in ("null", "invalid", "[]"):
        path.write_text(value, encoding="utf-8")
        assert setup.read_state(path) == {}
    setup.write_state(path, {"ready": "runtime-good"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ready": "runtime-good"}


def test_subprocess_imports_selected_runtime(tmp_path: Path) -> None:
    """
    Verify a fresh interpreter resolves the selected package instead of uv's environment copy.
    """
    package = tmp_path / "onnxruntime"
    package.mkdir()
    (package / "__init__.py").write_text('selection = "cached runtime"\n', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", "import onnxruntime; print(onnxruntime.selection)"],
        env=probes.environment(tmp_path), capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cached runtime"


@pytest.mark.parametrize("reply", [
    SimpleNamespace(returncode=1, stderr="no wheel for host"),
    OSError("offline"),
    subprocess.TimeoutExpired("installer", 600),
])
def test_installer_failures_are_optional(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: object,
) -> None:
    """
    Missing wheels, network errors and timed-out downloads leave the viewer able to fall back.
    """
    monkeypatch.setattr(installer.subprocess, "run", Mock(side_effect=[reply]))
    assert not installer.install("/fake/uv", tmp_path, ["onnxruntime-gpu>=1.27"])


def test_desktop_prepares_runtime_before_creating_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep package setup ahead of any camera backend constructor or acquisition.
    """
    from topdon_duo import desktop

    actions = []
    monkeypatch.setattr("topdon_duo.desktop.ensure_runtime", lambda: actions.append("runtime"))

    def stop_at_camera(*_args: object, **_kwargs: object) -> None:
        """
        Record camera creation without constructing or claiming hardware.
        """
        actions.append("camera")
        raise RuntimeError("camera boundary")

    monkeypatch.setattr("topdon_duo.desktop.create_camera", stop_at_camera)
    with pytest.raises(RuntimeError, match="camera boundary"):
        desktop.main([])
    assert actions == ["runtime", "camera"]


def test_web_prepares_runtime_before_creating_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Apply the same pre-acquisition setup to the web entry point.
    """
    from topdon_duo import web

    actions = []
    monkeypatch.setattr("topdon_duo.web.ensure_runtime", lambda: actions.append("runtime"))

    def stop_at_camera(*_args: object, **_kwargs: object) -> None:
        """
        Stop at the fake camera boundary before Flask or acquisition starts.
        """
        actions.append("camera")
        raise RuntimeError("camera boundary")

    monkeypatch.setattr("topdon_duo.web.create_camera", stop_at_camera)
    with pytest.raises(RuntimeError, match="camera boundary"):
        web.main([])
    assert actions == ["runtime", "camera"]


@pytest.mark.parametrize("viewer", ["desktop", "web"])
def test_help_never_installs_runtime(monkeypatch: pytest.MonkeyPatch, viewer: str) -> None:
    """
    Read-only CLI requests must exit before driver discovery or downloading packages.
    """
    from topdon_duo import desktop, web

    module = desktop if viewer == "desktop" else web
    prepare = Mock(side_effect=AssertionError("installation during help"))
    target = (
        "topdon_duo.desktop.ensure_runtime" if viewer == "desktop"
        else "topdon_duo.web.ensure_runtime"
    )
    monkeypatch.setattr(target, prepare)
    with pytest.raises(SystemExit) as result:
        module.main(["--help"])
    assert result.value.code == 0
    prepare.assert_not_called()
