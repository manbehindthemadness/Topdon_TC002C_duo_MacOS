"""
Apply independently owned thermal and visible-light pipeline settings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def optical_document(document: dict[str, Any]) -> dict[str, Any]:
    """
    Route the saved optical stack to its owner without duplicating node identities.
    """
    result = deepcopy(document)
    result["hardware"] = result.pop("visible_hardware", [])
    return result


def apply_camera_hardware(
    session: Any,
    document: dict[str, Any],
    *,
    previous_document: dict[str, Any] | None = None,
) -> None:
    """
    Apply both owners and restore the thermal pipeline if optical application fails.
    """
    optical: Any = getattr(session, "visible_pipeline_hardware", None)
    thermal = session.pipeline_hardware
    previous = previous_document if previous_document is not None else thermal.document
    if optical is not None:
        optical.hardware.load()
    if previous is None:
        previous = deepcopy(document)
        previous["hardware"] = []
    thermal.apply(document, previous_document=previous_document)
    if optical is None:
        return
    try:
        optical.apply(optical_document(document))
    except (session.api.CameraError, ValueError) as exc:
        if previous is not None:
            try:
                thermal.apply(previous)
            except (session.api.CameraError, ValueError) as rollback:
                raise session.api.CameraError(
                    f"{exc}; thermal rollback failed: {rollback}"
                ) from exc
        raise
