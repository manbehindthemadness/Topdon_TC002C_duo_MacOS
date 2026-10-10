"""Apply pipeline-owned fields in audited order; never restore calibration fields."""

from copy import deepcopy
from typing import Any

from .camera import CameraError
from .pipeline import active_nodes, preview_required, validate_pipeline

PIPELINE_FIELDS = {
    "brightness",
    "contrast",
    "palette",
    "detail",
    "detail_enabled",
    "noise_mode",
    "noise_general",
    "noise_spatial",
    "noise_temporal",
}


def desired_hardware(document: dict[str, Any]) -> tuple[dict[str, Any], str, int, int, bool]:
    """
    Resolve preview-processing controls without owning permanent calibration fields.
    """
    controls = {}
    preset, gamma, boost, fixed = "balanced", 50, 0, False
    thermal = not preview_required(document)
    for item in active_nodes(document, "hardware"):
        kind, p = item["type"], item["params"]
        if thermal:
            continue
        if kind in ("brightness", "contrast"):
            controls[kind] = p["value"]
        elif kind == "camera_colors":
            controls["palette"] = p["palette"]
        elif kind == "noise":
            controls.update(p)
        elif kind == "detail":
            controls.update(detail=p["amount"], detail_enabled=int(p["enabled"]))
            fixed = p["fixed"]
        elif kind == "preset":
            preset = p["value"]
        elif kind == "gamma":
            gamma = int(p["value"])
        elif kind == "boost":
            boost = p["value"]
    return controls, preset, gamma, boost, fixed


class PipelineHardware:
    def __init__(self, hardware):
        self.hardware = hardware
        self.document = None
        self._desired = None

    @property
    def applied_state(self) -> tuple | None:
        """
        Return a snapshot of the last successfully applied hardware configuration.
        """
        return deepcopy(self._desired)

    def invalidate_applied_state(self) -> None:
        """
        Require hardware application after an external restoration.
        """
        self._desired = None

    def apply(self, document, *, previous_document=None):
        candidate = validate_pipeline(document)
        if (
            not self.hardware.original
            and previous_document is not None
            and candidate["hardware"] == previous_document["hardware"]
        ):
            # An image-source/software edit must not depend on an unavailable
            # SDK handshake. Preserve the last actual hardware state so future
            # hardware edits still require validation and readback.
            self.document = candidate
            return
        desired = desired_hardware(candidate)
        if desired == self._desired:
            self.document = candidate
            return  # Reordering/collapse/software edits perform no USB operations.
        if not self.hardware.original and desired == ({}, "balanced", 50, 0, False):
            # Software remains usable when SDK controls are temporarily unavailable.
            self.document = candidate
            self._desired = desired
            return
        previous = self._desired
        try:
            self._apply(desired)
        except (CameraError, ValueError, TypeError) as exc:
            try:
                self._apply(previous or ({}, "balanced", 50, 0, False), force=True)
            except (CameraError, ValueError, TypeError) as rollback:
                raise CameraError(
                    f"Pipeline rejected: {exc}; restoration failed: {rollback}"
                ) from exc
            raise
        self.document = deepcopy(candidate)
        self._desired = desired

    def _apply(
        self, desired: tuple[dict[str, Any], str, int, int, bool], force: bool = False,
    ) -> None:
        """
        Apply only pipeline-owned fields, preserving standalone calibration settings.
        """
        controls, preset, gamma, boost, fixed = desired
        hw = self.hardware
        if not hw.original:
            hw.load()
        current = hw.state()
        unavailable = set(controls) - set(current)
        if unavailable:
            raise CameraError(f"Unavailable camera controls: {sorted(unavailable)}")
        changed = [
            name
            for name in PIPELINE_FIELDS
            if (
                name in controls
                and (not current[name]["enabled"] or current[name]["value"] != controls[name])
            )
            or (name not in controls and current.get(name, {}).get("enabled", False))
        ]
        if hw.fixed_range and (
            force
            or changed
            or not fixed
            or hw.processing_preset != preset
            or hw.gamma != gamma
            or hw.boost != boost
        ):
            hw.restore_fixed_range()
        # Restore/rebuild tone only when needed; all block writes precede the final LUT.
        tone_changes = (
            force
            or hw.gamma != gamma
            or hw.boost != boost
            or bool(changed)
            or hw.processing_preset != preset
        )
        if tone_changes and (hw.gamma != 50 or hw.boost):
            hw.restore_tone()
        order = (
            "brightness",
            "contrast",
            "noise_mode",
            "noise_general",
            "noise_spatial",
            "noise_temporal",
            "palette",
            "detail",
            "detail_enabled",
        )
        for name in order:
            if name in changed:
                hw.set(name, controls.get(name, current[name]["value"]), name in controls)
        if force or hw.processing_preset != preset:
            hw.set_processing_preset(preset)
        if tone_changes and (gamma != 50 or boost):
            hw.set_tone(gamma, boost)
        if fixed and not hw.fixed_range:
            hw.set_fixed_range(True)
