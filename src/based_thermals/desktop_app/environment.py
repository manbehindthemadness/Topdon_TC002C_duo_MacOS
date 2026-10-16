"""
Coordinate combined environment/calibration presets outside the pipeline ownership boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..environment_presets import (
    load_presets,
    measurement_specs,
    save_presets,
    validate_name,
    validate_preset,
)

if TYPE_CHECKING:
    from .session import DesktopSession


class EnvironmentPresets:
    """
    Own user preset persistence and transactional application for one camera session.
    """

    def __init__(self, session: DesktopSession) -> None:
        """
        Restore the user collection and its last selected name without camera writes.
        """
        self.session = session
        self.presets = load_presets(session.camera_profile, session.hardware.specs)
        name = session.saved_settings.get("environment_preset")
        self.current_name = name if isinstance(name, str) and name in self.presets else None

    def locked(self) -> bool:
        """
        Prevent snapshots of temporary probes and changes during logging or calibration tools.
        """
        session = self.session
        locked = bool(
            session.graphs.logging or session.pending_save_kind == "graph_log"
            or session.hardware.tone_busy or session.emissivity_calibration.active
            or session.reflected_calibration.active or session.distance_calibration.selecting
        )
        return locked

    def snapshot(self) -> dict[str, Any]:
        """
        Capture effective hardware values and saved references in Celsius and meters.
        """
        session = self.session
        state = session.hardware.state()
        specs = measurement_specs(session.hardware.specs)
        if not specs or any(not state.get(name, {}).get("available") for name in specs):
            raise ValueError("Wait for all environmental settings to become available")
        document = {
            "version": 1, "camera": session.camera_profile.settings_key,
            "hardware": {name: state[name]["value"] for name in specs},
            "calibrations": {
                "distance": session.distance_calibration.reference.as_dict()
                if session.distance_calibration.reference else None,
                "emissivity": session.emissivity_calibration.reference,
                "reflected": session.reflected_calibration.reference,
            },
        }
        result = validate_preset(document, session.camera_profile, session.hardware.specs)
        return result

    def state(self) -> dict[str, Any]:
        """
        Publish names and acknowledged selection without file or hardware I/O during refresh.
        """
        ready = True
        modified = False
        try:
            current = self.snapshot()
            if self.current_name is None and not self.locked():
                self.current_name = next(
                    (name for name in sorted(self.presets, key=str.casefold)
                     if self.presets[name] == current), None,
                )
            if self.current_name is not None:
                modified = self.presets[self.current_name] != current
        except (ValueError, TypeError, OverflowError):
            ready = False
        result = {"names": sorted(self.presets, key=str.casefold), "current": self.current_name,
                  "modified": modified, "ready": ready, "locked": self.locked()}
        return result

    def command(self, command: dict[str, Any]) -> None:
        """
        Save, rename, delete or load one combined preset after checking session locks.
        """
        if self.locked():
            raise ValueError("Stop logging and close calibration tools before using presets")
        session = self.session
        operation = command.get("operation")
        name = validate_name(command.get("name"))
        if operation == "load":
            if name not in self.presets:
                raise ValueError("Choose a saved environment + calibration preset")
            self._apply(self.presets[name])
            self.current_name = name
            session.persist_settings()
            session.hardware.error = ""
            session.notify(f'Environment + calibration preset "{name}" applied')
            session.refresh_camera_measurements(command)
            return
        candidate = dict(self.presets)
        selected = self.current_name
        if operation == "save":
            if name in candidate and command.get("replace") is not True:
                raise ValueError("Confirm replacement of the existing preset")
            candidate[name] = self.snapshot()
            selected = name
        elif operation == "rename":
            target = validate_name(command.get("new_name"))
            if name not in candidate:
                raise ValueError("Choose a saved preset to rename")
            if target != name and target in candidate and command.get("replace") is not True:
                raise ValueError("Confirm replacement of the existing preset")
            document = candidate.pop(name)
            candidate[target] = document
            if selected in (name, target):
                selected = target
        elif operation == "delete":
            if name not in candidate:
                raise ValueError("Choose a saved preset to delete")
            del candidate[name]
            if selected == name:
                selected = None
        else:
            raise ValueError("Unknown environment preset operation")
        save_presets(candidate, session.camera_profile, session.hardware.specs)
        self.presets, self.current_name = candidate, selected
        session.persist_settings()
        session.notify(f'Environment + calibration preset "{name}": {operation} complete')

    def _apply(self, document: dict[str, Any]) -> None:
        """
        Validate before writes, roll back partial failures, then replace saved references together.
        """
        session = self.session
        hardware = session.hardware
        document = validate_preset(document, session.camera_profile, hardware.specs)
        baseline = self.snapshot()
        features = hardware.capabilities().get("features", {})
        references = document["calibrations"]
        for name, reference in references.items():
            if reference is not None and not features.get(f"host_{name}_calibration"):
                raise ValueError(f"This camera does not support saved {name} calibration")
        distance = session.api.DistanceCalibrator(references["distance"])
        emissivity = session.api.EmissivityCalibrator(hardware, references["emissivity"])
        reflected = session.api.ReflectedCalibrator(hardware, references["reflected"])
        previous = hardware.state()
        attempted = []
        try:
            for name, value in document["hardware"].items():
                attempted.append(name)
                hardware.set(name, value, True)
        except (session.api.CameraError, ValueError, TypeError) as exc:
            failures = []
            for name in reversed(attempted):
                try:
                    hardware.set(name, baseline["hardware"][name], previous[name]["enabled"])
                except (session.api.CameraError, ValueError, TypeError) as rollback:
                    failures.append(f"{name}: {rollback}")
            session.renderer.request_measurement_restart()
            if failures:
                raise session.api.CameraError(
                    f"Preset rejected: {exc}; restoration failed: {'; '.join(failures)}"
                ) from exc
            raise
        session.distance_calibration = distance
        session.emissivity_calibration = emissivity
        session.reflected_calibration = reflected
        session.remembered_hardware.update(document["hardware"])
        session.ambient_input_celsius = document["hardware"].get("ambient")
