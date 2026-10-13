"""
Keep optional NVIDIA hardware discovery out of deterministic application tests.
"""

from typing import Any

import pytest


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
