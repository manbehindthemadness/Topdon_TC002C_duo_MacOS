"""
Desktop view settings, persisted preferences, and popup state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class SettingsController(SessionState):
    """
    Own desktop settings operations and transitions.
    """

    def persist_settings(self: DesktopSession) -> None:
        """
        Persist settings.
        """
        self.last_saved_spots = self.spots.saved_state(
            self.renderer.rotation, self.renderer.mirror_horizontal, self.renderer.mirror_vertical
        )
        try:
            self.api.save_settings(
                {
                    "pipeline": self.pipeline,
                    "display": {
                        name: getattr(self.renderer, name) for name in self.api.VIEW_DEFAULTS
                    },
                    "hardware": self.remembered_hardware,
                    "spot_hardware": self.remembered_spot_hardware,
                    "ambient_input_celsius": self.ambient_input_celsius,
                    "spots": self.last_saved_spots,
                    "rotation": self.renderer.rotation,
                    "advanced_auto": self.advanced_auto,
                    "auto_calibrate": self.auto_calibrate,
                    "fixed_range": self.remembered_fixed_range,
                    "processing_preset": self.remembered_processing_preset,
                    "camera_gamma": self.remembered_gamma,
                    "camera_boost": self.remembered_boost,
                    "show_graph": self.show_graph,
                    "graph_interval": self.graph_interval,
                    "graph_settings": self.graph_settings,
                    "capture_cursor": self.capture_cursor,
                    "capture_graphs": self.capture_graphs,
                    "timelapse_fpm": self.timelapse_fpm,
                    "distance_calibration": (
                        self.distance_calibration.reference.as_dict()
                        if self.distance_calibration.reference
                        else None
                    ),
                    "emissivity_calibration": self.emissivity_calibration.reference,
                    "reflected_calibration": self.reflected_calibration.reference,
                },
                **({"profile": self.camera_profile} if self.camera_profile.id != "duo" else {}),
            )
        except OSError as exc:
            self.api.LOG.warning("Could not save Camera settings: %s", exc)

    def current_window_size(self: DesktopSession) -> tuple[int, int] | None:
        """
        Use native Cocoa content bounds or the existing Linux/OpenCV size detector.
        """
        size = self.api.window_resize_size(self.api.WINDOW_NAME)
        if size is None and self.api.sys.platform == "darwin":
            size = self.pointer_monitor.window_size(self.requested_window_size)
        return size

    def toggle_graph(self: DesktopSession) -> None:
        """
        Toggle graph.
        """
        if self.graphs.logging or self.pending_save_kind == "graph_log":
            self.notify("Stop graph logging before hiding graphs.")
            return
        size = self.current_window_size() or self.last_window_size or self.requested_window_size
        if size is None:
            return
        width, height = size
        self.spot_drag.cancel()
        self.show_graph = not self.show_graph
        if not self.show_graph:
            self.graph_interval_editor.text = None
            self.graphs.pause()
        self.requested_window_size = (
            width * 2 if self.show_graph else max(1, round(width / 2)),
            height,
        )
        self.api.cv2.resizeWindow(
            self.api.WINDOW_NAME, *cast(tuple[int, int], self.requested_window_size)
        )
        self.last_window_size = self.requested_window_size
        self.picker.x = self.picker.y = None
        self.persist_settings()

    def notify(self: DesktopSession, message: str) -> None:
        """
        Notify.
        """
        self.status_message = message
        self.status_until = self.api.time.monotonic() + 8.0
        self.api.LOG.info("%s", message)

    def rotate_view(self: DesktopSession) -> None:
        """
        Rotate view.
        """
        if (
            self.graphs.logging
            or self.emissivity_calibration.running
            or self.reflected_calibration.running
        ):
            self.notify("Stop logging or calibration measurement before changing settings.")
            return
        if self.emissivity_calibration.active:
            self.emissivity_calibration.cancel()
        if self.reflected_calibration.active:
            self.reflected_calibration.cancel()
        self.spot_drag.cancel()
        # Rotate the stored sensor coordinates with the image, preserving samples.
        sensor_width, sensor_height = self.spots.native_size
        if self.renderer.rotation in (90, 270):
            sensor_width, sensor_height = sensor_height, sensor_width
        self.spots.mirror(
            sensor_width,
            sensor_height,
            self.renderer.mirror_horizontal,
            self.renderer.mirror_vertical,
        )
        self.spots.rotate_clockwise(sensor_height)
        self.renderer.rotate_clockwise()
        self.pipeline_revision += 1
        self.renderer.mirror_horizontal, self.renderer.mirror_vertical = self.api.geometry(
            self.pipeline, self.renderer.rotation
        )
        self.spots.mirror(
            sensor_height,
            sensor_width,
            self.renderer.mirror_horizontal,
            self.renderer.mirror_vertical,
        )
        if self.distance_calibration.corners or self.distance_calibration.selecting:
            self.distance_calibration.cancel()
        self.persist_settings()
        self.picker.x = self.picker.y = None
        self.last_selected = None

    def set_view_setting(self: DesktopSession, name: str, value: object) -> None:
        """
        Set view setting.
        """
        if (
            self.graphs.logging
            or self.emissivity_calibration.running
            or self.reflected_calibration.running
        ):
            self.notify("Stop logging or calibration measurement before changing settings.")
            return
        if name in ("mirror_horizontal", "mirror_vertical", "image_filter", "image_source"):
            candidate = self.api.validate_pipeline(self.pipeline)
            if name == "image_source":
                candidate["software"][0]["params"]["source"] = (
                    "raw" if value == "analyze" else value
                )
            else:
                kind = "mirror" if name.startswith("mirror_") else "filter"
                item = next((n for n in candidate["software"] if n["type"] == kind), None)
                if item is None:
                    item = self.api.node("software", kind)
                    candidate["software"].insert(len(candidate["software"]) - 1, item)
                key = name.removeprefix("mirror_") if kind == "mirror" else "filter"
                if kind == "mirror" and self.renderer.rotation in (90, 270):
                    key = "vertical" if key == "horizontal" else "horizontal"
                selected_node = cast(dict[str, Any], item)
                selected_node["params"][key] = value
            self.set_pipeline(candidate)
            setattr(self.renderer, name, value)
            self.persist_settings()
            return
        if name == "palette_source" and value == "camera":
            palette = self.hardware.state().get("palette", {})
            if not palette.get("available", False):
                raise ValueError("Camera palette control is unavailable")
            self.hardware.set("palette", palette["value"], True)
            self.remembered_hardware["palette"] = self.hardware.state()["palette"]["value"]
        self.renderer.set_view_setting(name, value)
        if name == "color_palette":
            self.renderer.set_view_setting("palette_source", "app")
        self.persist_settings()

    def set_pipeline(self: DesktopSession, document: Any) -> Any:
        """
        Set pipeline.
        """
        candidate = self.api.validate_pipeline(document, hardware_profile=self.camera_profile.id)
        previous_pipeline = self.pipeline
        self.pipeline_hardware.apply(candidate, previous_document=previous_pipeline)
        if self.hardware.tone_busy and self.tone_previous_pipeline is None:
            self.tone_previous_pipeline = previous_pipeline
        previous_mirrors = self.renderer.mirror_horizontal, self.renderer.mirror_vertical
        next_mirrors = self.api.geometry(candidate, self.renderer.rotation)
        if previous_mirrors != next_mirrors:
            width, height = (
                self.spots.native_size
                if self.renderer.rotation in (0, 180)
                else self.spots.native_size[::-1]
            )
            self.spots.mirror(
                width,
                height,
                previous_mirrors[0] != next_mirrors[0],
                previous_mirrors[1] != next_mirrors[1],
            )
            self.spot_drag.cancel()
            self.distance_calibration.cancel()
            if self.emissivity_calibration.active:
                self.emissivity_calibration.cancel()
            if self.reflected_calibration.active:
                self.reflected_calibration.cancel()
            self.picker.x = self.picker.y = None
            self.last_selected = None

        def signature(pipeline_document: Any) -> Any:
            """
            Signature.
            """
            tabs = self.api.software_tabs(pipeline_document)
            return (
                {
                    tab: [
                        (n["type"], n["params"], n["bypass"])
                        for n in tabs[tab]
                        if n["type"] != "preview"
                    ]
                    for tab in self.api.execution_dependencies(pipeline_document)
                },
                sorted(
                    (n["type"], self.api.json.dumps(n["params"], sort_keys=True), n["bypass"])
                    for n in pipeline_document["hardware"]
                ),
            )

        changed_image = signature(candidate) != signature(self.pipeline)
        self.pipeline = candidate
        self.renderer.set_pipeline(self.pipeline)
        self.actual_image_source = "raw" if self.api.thermal_source(self.pipeline) else "preview"
        if changed_image:
            self.pipeline_revision += 1
        for name in tuple(self.remembered_hardware):
            if name in self.api.PIPELINE_FIELDS:
                self.remembered_hardware.pop(name)
        for name, setting in self.hardware.state().items():
            if name in self.api.PIPELINE_FIELDS and setting["enabled"]:
                self.remembered_hardware[name] = setting["value"]
        self.persist_settings()

    def spots_state(self: DesktopSession) -> Any:
        """
        Spots state.
        """
        return {
            "spots": self.spots.state(),
            "placing": self.spots.placing,
            "calibration_available": self.calibration_available
            and not self.renderer.measurement_status,
            "locked": bool(
                self.graphs.logging
                or self.pending_save_kind == "graph_log"
                or self.emissivity_calibration.active
                or self.reflected_calibration.active
                or self.distance_calibration.selecting
            ),
        }

    def toggle_spots(self: DesktopSession) -> None:
        """
        Toggle spots.
        """
        if self.spots_state()["locked"]:
            self.notify("Stop logging or close calibration before changing measuring spots.")
            return
        self.spots.toggle()

    def toggle_temperature_unit(self: DesktopSession) -> None:
        """
        Toggle temperature unit.
        """
        self.set_view_setting(
            "temperature_unit", "F" if self.renderer.temperature_unit == "C" else "C"
        )

    def view_state(self: DesktopSession) -> dict:
        """
        View state.
        """
        message = "Camera preview" if self.actual_image_source == "preview" else "Raw thermal image"
        if (
            self.pipeline["software"][0]["params"]["source"] == "preview"
            and self.actual_image_source == "raw"
        ):
            message += " · Camera-style thermal recoloring (approximate palette)"
        message += f" · Pipeline {self.pipeline_elapsed_ms:.0f} ms"
        if self.pipeline_elapsed_ms > 500:
            message += " · Slow pipeline: image updates exceed 500 ms; sampling continues"
        if self.pipeline_error:
            message += f" · Pipeline error: {self.pipeline_error}"
        if self.hardware.error:
            message += f" · {self.hardware.error}"
        if self.renderer.measurement_status:
            message += f" · {self.renderer.measurement_status}"
        if self.graphs.logging:
            message += " · Settings locked while logging"
        elif self.reflected_calibration.running:
            message += " · Settings locked while measuring reflected temperature"
        elif self.emissivity_calibration.running:
            message += " · Settings locked while fitting emissivity"
        ui_hardware = self.hardware.state()
        if self.ambient_input_celsius is not None and "ambient" in ui_hardware:
            ui_hardware = {
                **ui_hardware,
                "ambient": {**ui_hardware["ambient"], "value": self.ambient_input_celsius},
            }
        return {
            **self.renderer.view_settings(),
            "camera_profile": self.camera_profile.as_dict(),
            "camera_capabilities": self.hardware.capabilities(),
            "spot_hardware": (
                self.hardware.spot_state() if self.camera_profile.id != "duo" else {}
            ),
            "reported_readings": (
                [self.api.asdict(reading) for reading in self.rendered.reported_readings]
                if hasattr(self, "rendered") else []
            ),
            "status_error": bool(self.pipeline_error or self.hardware.error),
            "pipeline": self.pipeline,
            "apple_acceleration": self.apple_capability,
            "nvidia_acceleration": self.nvidia_capability,
            "model_download_status": self.api.MODEL_DOWNLOADS.status(),
            "pipeline_serial": self.pipeline_serial,
            "pipeline_previews": self.pipeline_worker.latest_previews(self.pipeline_revision)
            if self.pipeline_worker is not None
            else {},
            "pipeline_preview_timings": self.pipeline_worker.latest_preview_timings(
                self.pipeline_revision
            )
            if self.pipeline_worker is not None
            else {},
            "pipeline_preview_errors": self.pipeline_worker.latest_preview_errors(
                self.pipeline_revision
            )
            if self.pipeline_worker is not None
            else {},
            "hardware": ui_hardware,
            "color_source": (
                "camera"
                if self.actual_image_source == "preview"
                and self.renderer.camera_preview
                and self.renderer.camera_color
                else "app"
            ),
            "advanced_auto": self.advanced_auto,
            "auto_calibrate": self.auto_calibrate,
            "fixed_range": self.hardware.fixed_range,
            "processing_preset": self.hardware.processing_preset,
            "processing_preset_available": bool(self.hardware.original),
            "camera_gamma": self.hardware.gamma,
            "camera_boost": self.hardware.boost,
            "tone_busy": self.hardware.tone_busy,
            "camera_operation": self.camera_operation
            if self.api.time.monotonic() < self.camera_operation_until
            else "",
            "camera_operation_busy": self.camera_operation_busy,
            "tone_progress": self.hardware.tone_progress,
            "actual_image_source": self.actual_image_source,
            "settings_locked": self.graphs.logging
            or self.emissivity_calibration.running
            or self.reflected_calibration.running,
            "distance_calibration": self.distance_calibration.state(),
            "emissivity_calibration": self.emissivity_calibration.state(),
            "reflected_calibration": self.reflected_calibration.state(),
            "status": message,
        }

    def open_view(self: DesktopSession) -> None:
        """
        Open view.
        """
        if not self.view_panel.is_open:
            self.api.collapse_previews(self.pipeline)
            self.persist_settings()
        try:
            try:
                self.hardware.load()
            except self.api.CameraError as exc:
                self.hardware.error = str(exc)
            state = self.view_state()
            try:
                x, y, width, height = self.api.cv2.getWindowImageRect(self.api.WINDOW_NAME)
                if self.api.sys.platform != "darwin" and width > 0 and height > 0:
                    state["anchor_top_right"] = [x + width - 1, y]
            except self.api.cv2.error:
                pass
            self.view_panel.open(state)
        except OSError as exc:
            self.notify(f"Could not open Camera: {exc}")

    def graph_config_state(self: DesktopSession) -> Any:
        """
        Graph config state.
        """
        return {
            "settings": self.graph_settings,
            "locked": self.graphs.logging or self.pending_save_kind == "graph_log",
        }
