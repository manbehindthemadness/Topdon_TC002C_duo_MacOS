"""
Dispatch camera commands and coordinate hardware initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class ControlsController(SessionState):
    """
    Own desktop controls operations and transitions.
    """

    def initialize_hardware(self: DesktopSession) -> bool:
        """
        Initialize hardware.
        """
        try:
            self.hardware.load()
            self.hardware.error = ""
            for name, value in self.remembered_hardware.items():
                if name in self.api.PIPELINE_FIELDS:
                    continue
                if self.camera_profile.id != "duo" and not self.hardware.state().get(
                    name, {},
                ).get("available"):
                    continue
                try:
                    self.hardware.set(name, value, True)
                except (self.api.CameraError, ValueError, TypeError) as exc:
                    self.hardware.error = f"Could not restore saved {name}: {exc}"
                    self.api.LOG.warning("%s", self.hardware.error)
        except self.api.HardwareProtocolError as exc:
            # Hardware controls are optional. An unknown protocol layout will
            # not become compatible by retrying it every second, and those
            # control transfers can interfere with Darwin's newly started UVC
            # stream. Leave live capture running with the settings unavailable.
            self.hardware.error = str(exc)
            self.api.LOG.warning("Hardware controls unavailable: %s", exc)
            return True
        except self.api.CameraError as exc:
            self.hardware.error = str(exc)
            self.api.LOG.warning("Could not read hardware settings: %s", exc)
            return False
        features = self.hardware.capabilities().get("features", {})
        if self.camera_profile.id != "duo":
            for spot_id, values in self.remembered_spot_hardware.items():
                for name, value in values.items():
                    try:
                        self.hardware.set_spot(name, value, spot_id, True)
                    except (self.api.CameraError, ValueError, TypeError) as exc:
                        self.hardware.error = f"Could not restore spot {spot_id} {name}: {exc}"
                        self.api.LOG.warning("%s", self.hardware.error)
        if features.get("auto_calibrate"):
            try:
                self.hardware.set_auto_calibrate(self.auto_calibrate)
                self.calibration_available = bool(features.get("calibrate_now"))
            except self.api.CameraError as exc:
                self.hardware.error = f"Could not set Auto calibrate: {exc}"
                self.api.LOG.warning("%s", self.hardware.error)
        try:
            self.pipeline_hardware.apply(self.pipeline)
            if self.hardware.tone_busy:
                self.tone_previous_pipeline = self.api.validate_pipeline(self.pipeline)
                previous_tone = self.tone_previous_pipeline
                assert previous_tone is not None
                for item in previous_tone["hardware"]:
                    if item["type"] in ("gamma", "boost"):
                        item["bypass"] = True
        except (self.api.CameraError, ValueError, TypeError) as exc:
            self.hardware.error = f"Could not restore pipeline: {exc}"
            self.api.LOG.warning("%s", self.hardware.error)
        return True

    def poll_spot_and_graph_commands(self: DesktopSession) -> None:
        """
        Poll spot and graph commands for the current desktop session.
        """
        for command in self.spots_panel.poll():
            if command.get("action") == "error":
                self.notify(f"Spot menu failed: {command.get('message', '')}")
                continue
            if self.spots_state()["locked"]:
                self.notify("Stop logging or close calibration before changing measuring spots.")
                continue
            try:
                action = command["action"]
                if action == "calibrate_now":
                    if not self.spots_state()["calibration_available"]:
                        raise ValueError("Camera calibration is unavailable or already in progress")
                    self.hardware.calibrate_now()
                    self.renderer.request_measurement_restart()
                    self.notify("Calibration requested")
                elif action == "placing":
                    if type(command["enabled"]) is not bool:
                        raise ValueError("Placement state must be a boolean")
                    self.spots.placing = command["enabled"]
                elif action == "enable":
                    self.spots.set_enabled(command["spot"], command["enabled"])
                elif action == "rename":
                    self.spots.rename(command["spot"], command["name"])
                elif action == "clear":
                    self.spots.clear(command["spot"])
                elif action == "clear_all":
                    self.spots.clear()
                else:
                    raise ValueError("Unknown spot menu operation")
            except (self.api.CameraError, KeyError, ValueError, TypeError) as exc:
                self.notify(f"Spot change rejected: {exc}")
        for command in self.graph_panel.poll():
            try:
                if command.get("action") == "error":
                    self.notify(f"Graph configuration failed: {command.get('message', '')}")
                elif command.get("action") == "settings":
                    if self.graph_config_state()["locked"]:
                        raise ValueError("Stop logging before changing graph settings")
                    settings = self.api.validate_graph_settings(command["settings"])
                    self.graphs.configure(settings)
                    self.graph_settings = settings
                    self.persist_settings()
            except (ValueError, TypeError, KeyError) as exc:
                self.notify(f"Graph settings rejected: {exc}")

    def poll_camera_commands(self: DesktopSession) -> None:
        """
        Poll camera commands for the current desktop session.
        """
        for command in self.view_panel.poll():
            if (
                self.graphs.logging
                or self.emissivity_calibration.running
                or self.reflected_calibration.running
            ) and command.get("action") in (
                "pipeline",
                "clear_model_cache",
                "setting",
                "hardware",
                "advanced_auto",
                "auto_calibrate",
                "fixed_range",
                "processing_preset",
                "tone",
                "cancel_tone",
                "restore_hardware",
                "reset",
                "distance_calibration",
            ):
                self.notify("Stop logging or calibration measurement before changing settings.")
                continue
            if (self.graphs.logging or self.reflected_calibration.running) and command.get(
                "action"
            ) == "emissivity_calibration":
                self.notify("Stop temperature logging before emissivity calibration.")
                continue
            if command.get("action") == "pipeline":
                try:
                    command["hardware_operation"] = (
                        self.api.desired_hardware(self.api.validate_pipeline(
                            command["document"], hardware_profile=self.camera_profile.id,
                        ))
                        != self.pipeline_hardware.applied_state
                    )
                except (ValueError, TypeError, KeyError):
                    command["hardware_operation"] = False
            operation_title = self.api.camera_operation_title(command)
            operation_started = self.api.time.monotonic()
            operation_error = None
            if operation_title:
                self.camera_operation_busy = True
                self.camera_operation = f"Applying {operation_title.lower()}… Image and graph updates may pause briefly."
                self.camera_operation_until = float("inf")
                self.api.LOG.info("Camera operation started: %s", command)
                self.view_panel.update(self.view_state())
            try:
                if self.camera_profile.id != "duo":
                    required = {
                        "auto_calibrate": "auto_calibrate", "processing_preset": "preset",
                        "tone": "gamma", "cancel_tone": "gamma", "fixed_range": "fixed_detail",
                        "distance_calibration": "host_distance_calibration",
                        "emissivity_calibration": "host_emissivity_calibration",
                        "reflected_calibration": "host_reflected_calibration",
                    }.get(str(command.get("action", "")))
                    if required and not self.hardware.capabilities()["features"].get(required):
                        raise ValueError("Camera operation is not supported by this device")
                if self.reflected_calibration.active and command.get("action") in (
                    "hardware",
                    "distance_calibration",
                    "emissivity_calibration",
                    "restore_hardware",
                    "reset",
                ):
                    self.reflected_calibration.cancel()
                if self.hardware.tone_busy and command.get("action") not in (
                    "tone",
                    "cancel_tone",
                    "restore_hardware",
                ):
                    raise ValueError("Wait for the camera tone update, or cancel it")
                if command.get("action") in ("pipeline", "pipeline_refresh"):
                    self.pipeline_serial = command.get("serial", self.pipeline_serial)
                    if command["action"] == "pipeline":
                        self.set_pipeline(command["document"])
                        self.hardware.error = ""
                elif command.get("action") == "clear_model_cache":
                    try:
                        count, size = self.api.MODEL_DOWNLOADS.clear_cache()
                    except OSError as exc:
                        raise ValueError(f"Could not clear downloaded model cache: {exc}") from exc
                    freed_mb = float(size) / 1_000_000
                    self.notify(f"Model cache cleared: {count} files, {freed_mb:.1f} MB freed")
                elif command.get("action") == "setting":
                    self.set_view_setting(command["name"], command["value"])
                elif command.get("action") in (
                    "tone",
                    "processing_preset",
                    "fixed_range",
                    "cancel_tone",
                ):
                    candidate = self.api.validate_pipeline(self.pipeline)

                    def camera_node(kind: Any, hardware_document: Any = candidate) -> Any:
                        """
                        Camera node.
                        """
                        camera_item = next(
                            (n for n in hardware_document["hardware"] if n["type"] == kind), None
                        )
                        if camera_item is None:
                            camera_item = self.api.node("hardware", kind)
                            hardware_document["hardware"].append(camera_item)
                        return camera_item

                    action = command["action"]
                    if action == "tone":
                        for key in ("gamma", "boost"):
                            if key in command:
                                item = camera_node(key)
                                item["params"]["value"] = command[key]
                                item["bypass"] = False
                    elif action == "cancel_tone":
                        self.hardware.restore_tone()
                        self.tone_previous_pipeline = None
                        self.pipeline_hardware.invalidate_applied_state()
                        for item in candidate["hardware"]:
                            if item["type"] in ("gamma", "boost"):
                                item["bypass"] = True
                    elif action == "processing_preset":
                        camera_node("preset")["params"]["value"] = command["value"]
                    elif action == "fixed_range":
                        item = camera_node("detail")
                        item["params"]["fixed"] = command["value"]
                        if command["value"]:
                            item["params"]["enabled"] = True
                    self.set_pipeline(candidate)
                    self.remembered_gamma, self.remembered_boost = (
                        self.hardware.gamma,
                        self.hardware.boost,
                    )
                    self.remembered_processing_preset = self.hardware.processing_preset
                    self.remembered_fixed_range = (
                        command["value"] if action == "fixed_range" else self.hardware.fixed_range
                    )
                    self.persist_settings()
                    self.hardware.error = ""
                elif command.get("action") == "auto_calibrate":
                    if self.emissivity_calibration.active:
                        self.emissivity_calibration.cancel()
                    if self.reflected_calibration.active:
                        self.reflected_calibration.cancel()
                    self.hardware.set_auto_calibrate(command["value"])
                    self.auto_calibrate = command["value"]
                    self.calibration_available = True
                    self.persist_settings()
                    self.hardware.error = ""
                elif command.get("action") == "hardware":
                    if command.get("spot_id") is not None:
                        spot_id = command["spot_id"]
                        self.hardware.set_spot(
                            command["name"], command["value"], spot_id, command["enabled"],
                        )
                        values = self.remembered_spot_hardware.setdefault(spot_id, {})
                        if command["enabled"]:
                            values[command["name"]] = command["value"]
                        else:
                            values.pop(command["name"], None)
                        self.persist_settings()
                        self.renderer.reset_measurement_average()
                        self.hardware.error = ""
                        continue
                    if self.emissivity_calibration.active:
                        self.emissivity_calibration.cancel()
                    value = command["value"]
                    if command["name"] == "ambient" and self.camera_profile.id == "duo":
                        spec = self.api.HARDWARE_CONTROLS["ambient"]
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not spec.minimum <= value <= spec.maximum
                        ):
                            raise ValueError("Ambient temperature is outside the supported range")
                        value = round(
                            spec.minimum + round((value - spec.minimum) / spec.step) * spec.step,
                            2,
                        )
                    self.hardware.set(command["name"], value, command["enabled"])
                    if command["name"] == "detail_enabled":
                        self.remembered_fixed_range = False
                    if command["name"] == "ambient":
                        self.ambient_input_celsius = (
                            command["value"] if command["enabled"] else None
                        )
                    if command["name"] == "palette" and command["enabled"]:
                        self.renderer.set_view_setting("palette_source", "camera")
                    if command["enabled"]:
                        self.remembered_hardware[command["name"]] = self.hardware.state()[
                            command["name"]
                        ]["value"]
                    else:
                        self.remembered_hardware.pop(command["name"], None)
                    self.persist_settings()
                    self.hardware.error = ""
                    self.renderer.reset_measurement_average()
                elif command.get("action") == "distance_calibration":
                    if self.emissivity_calibration.active:
                        self.emissivity_calibration.cancel()
                    operation = command["operation"]
                    if operation in ("select", "measure"):
                        self.distance_calibration.begin(
                            "reference" if operation == "select" else "measure"
                        )
                        self.notify(self.distance_calibration.message)
                    elif operation == "cancel":
                        self.distance_calibration.cancel()
                    elif operation == "save":
                        self.distance_calibration.save_reference(
                            command["side_m"], command["distance_m"]
                        )
                        self.persist_settings()
                        self.notify(self.distance_calibration.message)
                    elif operation == "apply":
                        estimated = self.distance_calibration.estimated_m
                        if estimated is None or not 0.3 <= estimated <= 99:
                            raise ValueError(
                                "Measure a valid square within the camera's distance range first"
                            )
                        value = round(estimated, 2)
                        self.hardware.set("distance", value, True)
                        self.distance_calibration.corners = []
                        self.remembered_hardware["distance"] = self.hardware.state()["distance"][
                            "value"
                        ]
                        self.renderer.reset_measurement_average()
                        self.hardware.error = ""
                        self.persist_settings()
                        self.notify("Estimated distance applied to the camera")
                    elif operation == "clear":
                        self.distance_calibration.clear()
                        self.persist_settings()
                    else:
                        raise ValueError("Unknown distance calibration operation")
                elif command.get("action") == "reflected_calibration":
                    operation = command["operation"]
                    if (
                        self.graphs.logging
                        or self.emissivity_calibration.running
                        or self.pending_save_kind == "graph_log"
                    ):
                        raise ValueError(
                            "Stop logging or emissivity fitting before reflected-temperature calibration"
                        )
                    if operation == "cancel":
                        self.reflected_calibration.cancel()
                    elif operation == "select":
                        self.emissivity_calibration.cancel()
                        self.distance_calibration.cancel()
                        self.reflected_calibration.begin()
                        self.picker.x = self.picker.y = None
                    elif operation == "measure":
                        self.reflected_calibration.start(self.api.time.monotonic())
                    elif operation == "apply":
                        self.remembered_hardware["reflected"] = self.reflected_calibration.apply()
                        self.persist_settings()
                    else:
                        raise ValueError("Unknown reflected-temperature calibration operation")
                    self.renderer.reset_measurement_average()
                    self.notify(self.reflected_calibration.message)
                elif command.get("action") == "emissivity_calibration":
                    operation = command["operation"]
                    if operation == "cancel":
                        self.emissivity_calibration.cancel()
                    elif operation == "select":
                        if self.pending_save_kind == "graph_log":
                            raise ValueError("Finish the pending logging dialog first")
                        self.distance_calibration.cancel()
                        self.emissivity_calibration.begin()
                        self.picker.x = self.picker.y = None
                    elif operation == "fit":
                        if self.pending_save_kind == "graph_log":
                            raise ValueError("Finish the pending logging dialog first")
                        self.emissivity_calibration.start(
                            command["known_celsius"], self.api.time.monotonic()
                        )
                    elif operation == "apply":
                        self.remembered_hardware["emissivity"] = self.emissivity_calibration.apply()
                        self.persist_settings()
                    else:
                        raise ValueError("Unknown emissivity calibration operation")
                    self.renderer.reset_measurement_average()
                    self.hardware.error = ""
                    self.notify(self.emissivity_calibration.message)
                elif command.get("action") == "advanced_auto":
                    if not isinstance(command["value"], bool):
                        raise ValueError("Advanced / Auto must be a boolean")
                    self.advanced_auto = command["value"]
                    self.persist_settings()
                elif command.get("action") == "restore_hardware":
                    if self.emissivity_calibration.active:
                        self.emissivity_calibration.cancel()
                    self.hardware.restore()
                    self.pipeline_hardware.invalidate_applied_state()
                    for item in self.pipeline["hardware"]:
                        item["bypass"] = True
                    self.renderer.set_pipeline(self.pipeline)
                    self.pipeline_revision += 1
                    self.remembered_gamma, self.remembered_boost = 50, 0
                    self.remembered_processing_preset = self.hardware.processing_preset
                    self.remembered_fixed_range = False
                    self.remembered_hardware.clear()
                    self.remembered_spot_hardware.clear()
                    self.ambient_input_celsius = None
                    self.persist_settings()
                    self.hardware.error = ""
                    self.renderer.reset_measurement_average()
                elif command.get("action") == "reset":
                    self.set_pipeline(self.api.default_pipeline())
                    for name, value in self.api.VIEW_DEFAULTS.items():
                        setattr(self.renderer, name, value)
                    self.persist_settings()
                elif command.get("action") == "error":
                    self.notify(f"Camera window failed: {command.get('message', '')}")
            except (KeyError, ValueError, TypeError, self.api.CameraError) as exc:
                self.hardware.error = f"Camera setting rejected: {exc}"
                operation_error = self.hardware.error
                self.notify(self.hardware.error)
            finally:
                if operation_title:
                    self.camera_operation_busy = False
                    self.camera_operation = operation_error or f"{operation_title} updated"
                    self.camera_operation_until = self.api.time.monotonic() + 8
                    self.api.LOG.info(
                        "Camera operation finished: %s in %.3fs; %s",
                        operation_title,
                        self.api.time.monotonic() - operation_started,
                        self.camera_operation,
                    )
                    self.view_panel.update(self.view_state())
