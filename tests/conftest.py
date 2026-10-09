"""
Keep optional NVIDIA hardware discovery out of deterministic application tests.
"""

from typing import Any

import pytest


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
        "topdon_duo.desktop.nvidia_acceleration",
        "topdon_duo.processing.branch.nvidia_acceleration",
        "topdon_duo.processing.scheduler.nvidia_acceleration",
    ):
        monkeypatch.setattr(target, unavailable)
