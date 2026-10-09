"""
Present effective compute devices without overwriting saved Apple preferences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..view_window import ControlRow


def update_compute_devices(row: ControlRow, params: dict[str, Any], available: bool) -> None:
    """
    Lock CPU execution to CPU only and restore the saved choice for Core ML.
    """
    cpu_only = params["backend"] == "cpu"
    if cpu_only:
        row.timer.stop()
    help_text = (
        "CPU execution uses CPU only. Select Apple Core ML to choose compute devices."
        if cpu_only
        else "Choose the compute devices used by Apple Core ML."
    )
    row.setToolTip(help_text)
    row.input.setToolTip(help_text)
    row.update_state("CPUOnly" if cpu_only else params["apple_compute"], available and not cpu_only)
