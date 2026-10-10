"""Apply pipeline-owned fields in audited order; never restore calibration fields."""

from copy import deepcopy
from typing import Any

from .camera import CameraError
from .camera_backends.controls import DeviceControls
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
        elif kind == "device_control" and p["control"]:
            controls[p["control"]] = p["value"]
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

    def apply(
        self, document: dict[str, Any], *, previous_document: dict[str, Any] | None = None,
    ) -> None:
        """
        Apply changed hardware settings and roll back rejected configurations.
        """
        candidate = validate_pipeline(
            document, hardware_profile=self.hardware.profile.id
            if isinstance(self.hardware, DeviceControls) else "duo",
        )
        if isinstance(self.hardware, DeviceControls):
            self._apply_device(candidate)
            return
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
        applicable = deepcopy(candidate)
        applicable["hardware"] = [item for item in candidate["hardware"]
                                  if item["type"] != "device_control"]
        desired = desired_hardware(applicable)
        preset_active = preview_required(candidate) and any(
            item["type"] == "preset" for item in active_nodes(candidate, "hardware")
        )
        previous_preset_active = (
            self.document is not None
            and preview_required(self.document)
            and any(item["type"] == "preset" for item in active_nodes(self.document, "hardware"))
        )
        verify_preset = preset_active and (self._desired is None or not previous_preset_active)
        if desired == self._desired and not verify_preset:
            self.document = candidate
            return  # Reordering/collapse/software edits perform no USB operations.
        if not self.hardware.original and desired == ({}, "balanced", 50, 0, False):
            # Software remains usable when SDK controls are temporarily unavailable.
            self.document = candidate
            self._desired = desired
            return
        previous = self._desired
        try:
            self._apply(desired, verify_preset=verify_preset)
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

    def _apply_device(self, document: dict[str, Any]) -> None:
        """
        Apply supported display settings with target rules and restore only pipeline ownership.
        """
        hw = self.hardware
        hw.validate_pipeline(document)
        applicable = deepcopy(document)
        nodes = hw.capabilities()["nodes"]
        applicable["hardware"] = [item for item in document["hardware"]
                                  if item["type"] == "device_control"
                                  or nodes.get(item["type"], {}).get("supported")]
        desired = desired_hardware(applicable)
        controls = desired[0]
        specs = hw.specs
        selected = {name: value for name, value in controls.items()
                    if name in specs and specs[name].supported and specs[name].scope == "device"}
        if any(specs[name].effect != "preview" for name in selected):
            raise ValueError("Measurement corrections belong to Camera settings, outside pipelines")
        for name, value in selected.items():
            specs[name].validate(value)
        if desired == self._desired:
            self.document = deepcopy(document)
            return
        previous = self._desired[0] if self._desired is not None else {}
        try:
            for name in set(previous) - set(selected):
                if name in hw.enabled:
                    hw.set(name, hw.original[name], False)
            for name, value in selected.items():
                if name not in hw.enabled or hw.values[name] != value:
                    hw.set(name, value, True)
        except CameraError:
            for name in set(selected) | set(previous):
                if name in specs and name in hw.original:
                    hw.set(name, previous.get(name, hw.original[name]), name in previous)
            raise
        self.document = deepcopy(document)
        self._desired = ({**selected}, *desired[1:])

    def _apply(
        self, desired: tuple[dict[str, Any], str, int, int, bool], force: bool = False,
        verify_preset: bool = False,
    ) -> None:
        """
        Apply only pipeline-owned fields, preserving standalone calibration settings.
        """
        controls, preset, gamma, boost, fixed = desired
        hw = self.hardware
        if set(controls) - PIPELINE_FIELDS:
            raise ValueError("Unsupported display control or measurement correction in pipeline")
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
        preset_changes = force or verify_preset or hw.processing_preset != preset
        if hw.fixed_range and (
            force
            or verify_preset
            or changed
            or not fixed
            or hw.processing_preset != preset
            or hw.gamma != gamma
            or hw.boost != boost
        ):
            hw.restore_fixed_range()
        if preset_changes:
            hw.capture_processing_preset()
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
        if preset_changes:
            hw.set_processing_preset(preset)
        if tone_changes and (gamma != 50 or boost):
            hw.set_tone(gamma, boost)
        if fixed and not hw.fixed_range:
            hw.set_fixed_range(True)
