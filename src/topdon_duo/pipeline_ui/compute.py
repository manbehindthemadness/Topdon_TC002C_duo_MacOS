"""
Present effective compute devices without overwriting saved Apple preferences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..acceleration import select_backend
from ..onnx_models import MODELS as ONNX_MODELS

if TYPE_CHECKING:
    from ..view_window import ControlRow


def update_compute_devices(
    row: ControlRow,
    params: dict[str, Any],
    available: bool,
    backend: str | None = None,
) -> None:
    """
    Enable Apple device choices only for Core ML and retain the saved preference.
    """
    cpu_only = (params["backend"] if backend is None else backend) != "coreml"
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


def state_backend(params: dict[str, Any], state: dict[str, Any]) -> str:
    """
    Resolve the effective execution backend using the viewer's capability state.
    """
    backend = select_backend(
        params["backend"],
        bool(state.get("apple_acceleration", {}).get("available")),
        bool(state.get("nvidia_acceleration", {}).get("available")),
    )
    return backend


def backend_status(item: dict[str, Any], state: dict[str, Any]) -> str:
    """
    Explain GPU or CPU fallback while keeping the saved preference visible.
    """
    supported = item["type"] in ("onnx_superresolution", "onnx_denoise", "onnx_style") or (
        item["type"] == "enhance"
        and (item["params"]["model"] == "acnet" or item["params"]["model"] in ONNX_MODELS)
    )
    if not supported:
        return ""
    params = item["params"]
    backend = state_backend(params, state)
    if backend == params["backend"]:
        return ""
    label = {"cpu": "CPU", "cuda": "NVIDIA CUDA", "coreml": "Apple Core ML"}[backend]
    status = f"{label} execution · saved GPU settings retained"
    return status


def configure_backend_row(
    row: ControlRow,
    item: dict[str, Any],
    key: str,
    state: dict[str, Any],
    available: bool,
) -> bool:
    """
    Expose supported accelerators while retaining unavailable saved selections.
    """
    supported = item["type"] in ("onnx_superresolution", "onnx_denoise", "onnx_style") or (
        item["type"] == "enhance"
        and (item["params"]["model"] == "acnet" or item["params"]["model"] in ONNX_MODELS)
    )
    apple = bool(state.get("apple_acceleration", {}).get("available"))
    nvidia = bool(state.get("nvidia_acceleration", {}).get("available"))
    visible = supported and (
        (apple or nvidia)
        if key == "backend"
        else apple and state_backend(item["params"], state) != "cuda"
    )
    row.setVisible(visible)
    if key == "backend":
        for index in range(row.input.count()):
            selected = row.input.itemData(index)
            row.input.model().item(index).setEnabled(
                selected == "cpu" or selected == "coreml" and apple or selected == "cuda" and nvidia
            )
        help_text = (
            "Execution backend for display enhancement. Anime4K09 uses CPU. "
            "Unavailable saved GPU preferences try the other GPU before CPU and remain saved."
        )
        row.setToolTip(help_text)
        row.input.setToolTip(help_text)
    return available and visible
