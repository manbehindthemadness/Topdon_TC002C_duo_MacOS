"""
Keep optical hardware state and change detection independent of thermal settings.
"""

from __future__ import annotations

from typing import Any

from ..desktop_app.dual_camera import optical_document
from ..pipeline_hardware import desired_hardware


def stack_state(state: dict[str, Any], stack: str) -> dict[str, Any]:
    """
    Present one camera's cached capability metadata without probing either device.
    """
    if stack != "visible_hardware":
        return state
    return {
        **state,
        "camera_capabilities": state.get("visible_camera_capabilities", {}),
        "hardware": state.get("visible_hardware", {}),
    }


def desired_camera_hardware(document: dict[str, Any]) -> tuple[Any, Any]:
    """
    Detect hardware operations in both persisted camera stacks.
    """
    return desired_hardware(document), desired_hardware(optical_document(document), "visible")
