"""
Capability state and a reversible control owner for independently implemented backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..camera import CameraError
from .contracts import CameraProfile, ControlSpec, ControlValue

DUO_NODES = {
    "brightness": ("brightness",), "contrast": ("contrast",),
    "camera_colors": ("palette",),
    "noise": ("noise_mode", "noise_general", "noise_spatial", "noise_temporal"),
    "detail": ("detail", "detail_enabled"),
}


def capability_state(
    specs: dict[str, ControlSpec], states: dict[str, Any], *,
    features: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Describe supported settings separately from temporary availability and enable state.
    """
    features = features or {}
    controls = {
        name: {**spec.as_dict(), "available": bool(states.get(name, {}).get("available")),
               "reason": spec.reason or states.get(name, {}).get("reason", "")}
        for name, spec in specs.items()
    }
    nodes = {}
    for kind, names in DUO_NODES.items():
        supported = bool(features.get("duo_nodes")) and all(
            name in specs and specs[name].supported for name in names
        )
        available = supported and all(controls[name]["available"] for name in names)
        nodes[kind] = {"supported": supported, "available": available,
                       "reason": "" if available else (
                           "Camera settings are not ready" if supported
                           else "Not supported by this camera")}
    for kind in ("preset", "gamma", "boost"):
        supported = bool(features.get(kind))
        available = supported and bool(features.get("ready"))
        nodes[kind] = {"supported": supported, "available": available,
                       "reason": "" if available else (
                           "Camera settings are not ready" if supported
                           else "Not supported by this camera")}
    return {"controls": controls, "nodes": nodes, "features": features}


class DeviceControls:
    """
    Snapshot verified volatile settings before overrides; backends own read/write operations.

    Spot controls use separate identities and never silently become whole-device settings.
    """

    def __init__(
        self, profile: CameraProfile, specs: dict[str, ControlSpec],
        read: Callable[[str, str | None], ControlValue],
        write: Callable[[str, ControlValue, str | None], None],
        *, validate_pipeline: Callable[[dict[str, Any]], None] | None = None,
        spot_ids: tuple[str, ...] = (),
    ) -> None:
        """
        Attach target-compatible commands without issuing any transfers.
        """
        if (any(not isinstance(name, str) or not name or not isinstance(spec, ControlSpec)
                for name, spec in specs.items())
                or any(not isinstance(spot, str) or not spot for spot in spot_ids)
                or len(set(spot_ids)) != len(spot_ids)):
            raise ValueError("Invalid camera control or spot identity")
        self.profile, self.specs = profile, specs.copy()
        self._read, self._write = read, write
        self._validate_pipeline = validate_pipeline
        self.original: dict[str, ControlValue] = {}
        self.values: dict[str, ControlValue] = {}
        self.enabled: set[str] = set()
        self._spot_original: dict[tuple[str, str], ControlValue] = {}
        self._spot_enabled: set[tuple[str, str]] = set()
        self.spot_ids = spot_ids
        self._spot_values: dict[tuple[str, str], ControlValue] = {}
        self._loaded = False
        self.error = ""
        self.gamma, self.boost, self.processing_preset = 50, 0, "balanced"
        self.fixed_range = False
        self.tone_busy = False
        self.tone_progress = 0

    def load(self) -> None:
        """
        Read a complete supported device baseline before any override is permitted.
        """
        if self._loaded:
            return
        original = {}
        spot_original = {}
        for name, spec in self.specs.items():
            if spec.supported and spec.scope == "device":
                value = self._read(name, None)
                spec.validate(value)
                original[name] = value
            elif spec.supported and spec.scope == "spot":
                for spot_id in self.spot_ids:
                    value = self._read(name, spot_id)
                    spec.validate(value)
                    spot_original[name, spot_id] = value
        self.original = original
        self.values = original.copy()
        self._spot_original = spot_original
        self._spot_values = spot_original.copy()
        self._loaded = True

    def state(self) -> dict[str, Any]:
        """
        Report each setting independently, keeping unsupported and unready states explicit.
        """
        return {name: {"value": self.values.get(name, spec.default),
                       "enabled": name in self.enabled,
                       "available": spec.supported and name in self.original,
                       "reason": spec.reason or ("" if name in self.original
                                                  else "Camera settings are not ready")}
                for name, spec in self.specs.items()}

    def capabilities(self) -> dict[str, Any]:
        """
        Publish metadata without invoking commands from UI or document validation.
        """
        return capability_state(self.specs, self.state(), features={"ready": self._loaded})

    def spot_state(self) -> dict[str, Any]:
        """
        Publish known spot settings from saved readbacks without UI-side device queries.
        """
        return {spot_id: {name: {"value": self._spot_values.get((name, spot_id), spec.default),
                                "available": spec.supported and self._loaded,
                                "enabled": (name, spot_id) in self._spot_enabled}
                         for name, spec in self.specs.items() if spec.scope == "spot"}
                for spot_id in self.spot_ids}

    def set(self, name: str, value: ControlValue, enabled: bool) -> None:
        """
        Override or restore one whole-device field with verified readback and rollback.
        """
        if type(enabled) is not bool:
            raise TypeError("Control enabled state must be a boolean")
        spec = self.specs.get(name)
        if spec is None or not spec.supported or spec.scope != "device":
            raise CameraError("Camera control is unsupported or requires a spot")
        spec.validate(value)
        self.load()
        previous = self._read(name, None)
        target = value if enabled else self.original[name]
        try:
            self._write(name, target, None)
            if self._read(name, None) != target:
                raise CameraError(f"Camera did not apply {spec.title}")
        except CameraError as exc:
            try:
                self._write(name, previous, None)
                if self._read(name, None) != previous:
                    raise CameraError("Rollback readback did not match")
            except CameraError as rollback:
                self.enabled.add(name)
                raise CameraError(f"{exc}; rollback failed: {rollback}") from exc
            raise
        self.values[name] = value if enabled else self.original[name]
        if enabled:
            self.enabled.add(name)
        else:
            self.enabled.discard(name)

    def set_spot(self, name: str, value: ControlValue, spot_id: str, enabled: bool) -> None:
        """
        Apply a spot-scoped correction without overwriting another spot's baseline.
        """
        spec = self.specs.get(name)
        if (spec is None or not spec.supported or spec.scope != "spot"
                or not isinstance(spot_id, str) or not spot_id or type(enabled) is not bool):
            raise CameraError("Unsupported spot control")
        spec.validate(value)
        key = name, spot_id
        self.load()
        if spot_id not in self.spot_ids:
            raise CameraError("Unknown camera spot")
        previous = self._read(name, spot_id)
        self._spot_original.setdefault(key, previous)
        target = value if enabled else self._spot_original[key]
        try:
            self._write(name, target, spot_id)
            if self._read(name, spot_id) != target:
                raise CameraError(f"Camera did not apply spot {spec.title}")
        except CameraError as exc:
            try:
                self._write(name, previous, spot_id)
                if self._read(name, spot_id) != previous:
                    raise CameraError("Spot rollback readback did not match")
            except CameraError as rollback:
                self._spot_enabled.add(key)
                raise CameraError(f"{exc}; rollback failed: {rollback}") from exc
            raise
        if enabled:
            self._spot_enabled.add(key)
        else:
            self._spot_enabled.discard(key)
        self._spot_values[key] = target

    def validate_pipeline(self, document: dict[str, Any]) -> None:
        """
        Evaluate target-specific dependencies before writes, independently of document syntax.
        """
        if self._validate_pipeline is not None:
            self._validate_pipeline(document)

    def invalidate_device(self) -> None:
        """
        Forget baselines after an unowned device replacement, requiring fresh reads.
        """
        if self.enabled or self._spot_enabled:
            raise CameraError("Restore owned settings before replacing the camera")
        self.original.clear()
        self.values.clear()
        self._spot_original.clear()
        self._spot_values.clear()
        self._loaded = False

    @property
    def preview_active(self) -> bool:
        """
        Identify owned settings affecting display pixels.
        """
        return any(self.specs[name].effect == "preview" for name in self.enabled)

    @property
    def measurement_active(self) -> bool:
        """
        Identify corrections affecting native measurements.
        """
        return (any(self.specs[name].effect == "measurement" for name in self.enabled)
                or any(self.specs[name].effect == "measurement" for name, _ in self._spot_enabled))

    def restore(self) -> None:
        """
        Restore only owned fields, retaining ownership when a restore fails.
        """
        for name in tuple(self.enabled):
            self.set(name, self.original[name], False)
        for name, spot_id in tuple(self._spot_enabled):
            self.set_spot(name, self._spot_original[name, spot_id], spot_id, False)
