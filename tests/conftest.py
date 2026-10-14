"""
Keep optional NVIDIA hardware discovery out of deterministic application tests.
"""

from collections.abc import Iterator
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from based_thermals.export_jobs import EXPORT_JOBS


@pytest.fixture(autouse=True)
def no_live_model_conversion(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    Keep automatic dependency/model downloads outside deterministic tests unless explicitly faked.
    """
    EXPORT_JOBS.begin()

    def unavailable(
        _command: list[str],
        _environment: dict[str, str],
        _directory: Path,
        *,
        cancellation: Event | None = None,
    ) -> None:
        """
        Refuse launching real conversion installers from ordinary application tests.
        """
        del cancellation
        raise AssertionError("Live model conversion disabled in tests")

    monkeypatch.setattr("based_thermals.model_exports.run_export", unavailable)
    yield
    EXPORT_JOBS.close()


@pytest.fixture(autouse=True)
def no_runtime_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep startup provisioning and external package installation out of application tests.
    """
    monkeypatch.setenv("BASED_THERMALS_RUNTIME_SETUP", "0")


@pytest.fixture(autouse=True)
def no_live_viewer_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep GUI processes out of unit tests using the injectable HighGUI boundary.

    Qt viewer integration tests explicitly install their bridge or use a subprocess.
    """
    monkeypatch.setattr("based_thermals.desktop.ViewerPanel", lambda: None)


@pytest.fixture(autouse=True)
def no_live_nvidia_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fake startup capabilities; CUDA tests opt in with explicit availability or mocks.
    """

    def unavailable() -> dict[str, Any]:
        """
        Describe a CPU test host without loading native GPU libraries.
        """
        return {"available": False, "cuda": False, "reason": "NVIDIA probe disabled in tests"}

    for target in (
        "based_thermals.desktop.nvidia_acceleration",
        "based_thermals.processing.branch.nvidia_acceleration",
        "based_thermals.processing.scheduler.nvidia_acceleration",
    ):
        monkeypatch.setattr(target, unavailable)
