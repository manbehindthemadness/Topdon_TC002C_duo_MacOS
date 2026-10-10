"""Verify optional CUDA installation without modifying Python environments."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from topdon_duo import cuda_setup as setup


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(setup.sys, "platform", "linux")
    monkeypatch.setattr(setup.sys, "prefix", "/fake/venv")
    monkeypatch.setattr(setup.sys, "base_prefix", "/fake/python")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/fake/uv")
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout='{"available": true}'))
    monkeypatch.setattr(setup.subprocess, "run", run)
    return run


@pytest.mark.parametrize("cuda,requirement", [
    ("12", "onnxruntime-gpu[cuda,cudnn]>=1.23,<1.27"),
    ("13", "onnxruntime-gpu[cuda,cudnn]>=1.27,<1.31"),
])
def test_resolve_install_and_verify(commands: Mock, cuda: str, requirement: str) -> None:
    assert setup.main(["--cuda", cuda]) == 0
    calls = commands.call_args_list
    assert requirement in calls[0].args[0] and "--dry-run" in calls[0].args[0]
    assert calls[1].args[0][1:3] == ["pip", "uninstall"]
    assert "onnxruntime" in calls[1].args[0] and "onnxruntime-gpu" in calls[1].args[0]
    assert requirement in calls[2].args[0]
    assert calls[3].kwargs["timeout"] == 30


def test_resolution_failure_leaves_runtime_intact(commands: Mock) -> None:
    commands.return_value.returncode = 1
    assert setup.main(["--cuda", "12"]) == 1
    assert commands.call_count == 1


def test_dry_run_and_system_library_mode(commands: Mock) -> None:
    assert setup.main(["--cuda", "13", "--dry-run", "--system-libraries"]) == 0
    assert commands.call_count == 1
    assert "onnxruntime-gpu>=1.27,<1.31" in commands.call_args.args[0]


def test_install_failure_restores_cpu(commands: Mock) -> None:
    commands.side_effect = [SimpleNamespace(returncode=code) for code in (0, 0, 1, 0)]
    assert setup.main(["--cuda", "12"]) == 1
    assert "--reinstall" in commands.call_args.args[0]
    assert "onnxruntime>=1.23" in commands.call_args.args[0]


@pytest.mark.parametrize("reply", [
    SimpleNamespace(returncode=0, stdout='{"available": false}'),
    SimpleNamespace(returncode=0, stdout="invalid"),
    SimpleNamespace(returncode=0, stdout="null"),
    SimpleNamespace(returncode=-11, stdout=""),
    subprocess.TimeoutExpired("probe", 30),
])
def test_failed_gpu_verification_reports_cpu_fallback(commands: Mock, reply: object) -> None:
    commands.side_effect = [SimpleNamespace(returncode=0)] * 3 + [reply]
    assert setup.main(["--cuda", "13"]) == 1


@pytest.mark.parametrize("platform,machine", [("darwin", "arm64"), ("linux", "riscv64")])
def test_unsupported_platform_does_not_install(
    commands: Mock, monkeypatch: pytest.MonkeyPatch, platform: str, machine: str,
) -> None:
    monkeypatch.setattr(setup.sys, "platform", platform)
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    assert setup.main(["--cuda", "13"]) == 2
    commands.assert_not_called()


def test_system_python_does_not_install(commands: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.sys, "prefix", setup.sys.base_prefix)
    assert setup.main(["--cuda", "13"]) == 2
    commands.assert_not_called()
