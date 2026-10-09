"""
Receive frames, present measurements, and handle desktop interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class PresentationController(SessionState):
    """
    Own desktop presentation operations and transitions.
    """

    def receive_frame(self: DesktopSession) -> bool | None:
        """
        Receive frame for the current desktop session.
        """
        if self.hardware.tone_busy:
            try:
                if self.hardware.advance_tone():
                    self.tone_previous_pipeline = None
                    self.notify("Camera tone update complete")
                    self.camera_operation = "Camera tone update complete"
                    self.camera_operation_until = self.api.time.monotonic() + 8
            except self.api.CameraError as exc:
                if self.tone_previous_pipeline is not None:
                    previous_pipeline = self.tone_previous_pipeline
                    self.tone_previous_pipeline = None
                    self.pipeline_hardware.invalidate_applied_state()
                    try:
                        self.set_pipeline(previous_pipeline)
                    except (self.api.CameraError, ValueError) as restore_error:
                        self.api.LOG.error("Pipeline tone rollback failed: %s", restore_error)
                self.remembered_gamma, self.remembered_boost = (
                    self.hardware.gamma,
                    self.hardware.boost,
                )
                self.hardware.error = str(exc)
                self.camera_operation = f"Camera tone update failed: {exc}"
                self.camera_operation_until = self.api.time.monotonic() + 8
                self.persist_settings()
        self.fresh_frame = self.frame is not None
        if self.fresh_frame:
            self.last_frame = self.frame
            self.last_frame_at = self.api.time.monotonic()
        elif self.last_frame is None:
            waiting = self.api.np.zeros((480, 640, 3), self.api.np.uint8)
            self.api.cv2.putText(
                waiting,
                "Waiting for camera...",
                (20, 40),
                self.api.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                1,
            )
            self.api.cv2.imshow(self.api.WINDOW_NAME, waiting)
            return self.api.cv2.waitKey(1) & 0xFF not in (ord("q"), 27)
        else:
            self.frame = self.last_frame
        if self.hardware_setup_pending:
            now = self.api.time.monotonic()
            if self.hardware_retry_deadline is None:
                self.hardware_retry_deadline = now + 30
            if now >= self.next_hardware_retry:
                self.diagnostics.stage("hardware_setup_retry")
                self.hardware_setup_pending = not self.initialize_hardware()
                self.next_hardware_retry = now + 1
                if not self.hardware_setup_pending:
                    self.notify("Camera settings loaded after startup")
                elif (
                    self.hardware_retry_deadline is not None and now >= self.hardware_retry_deadline
                ):
                    self.hardware_setup_pending = False
                    self.notify(
                        "Camera settings unavailable after startup retries; reconnect camera."
                    )
                self.diagnostics.stage("controls")
        display_error = self.display_awake.check()
        if display_error:
            self.notify(display_error)
        return None

    def render_frame(self: DesktopSession) -> None:
        """
        Render frame for the current desktop session.
        """
        assert self.pipeline_worker is not None
        assert self.frame is not None
        self.pipeline_worker.enable_previews(self.view_panel.is_open)
        self.renderer.camera_preview = self.hardware.preview_active
        self.renderer.camera_color = "palette" in self.hardware.enabled
        self.renderer.hardware_settings = self.hardware.state() if self.hardware.original else {}
        ambient = self.renderer.hardware_settings.get("ambient", {})
        self.renderer.ambient_celsius = ambient.get("value") if ambient.get("available") else None
        self.diagnostics.stage("render")
        self.rendered = self.renderer.render_detailed(
            self.frame, update_measurements=self.fresh_frame, image_processing=False
        )
        image_frame = (
            self.frame if self.rendered.measurements_valid else self.renderer.last_valid_frame
        )
        if image_frame is not None:
            palette = self.renderer.hardware_settings.get("palette", {}).get("value", 1)
            self.pipeline_worker.submit(
                image_frame,
                self.renderer.averaged_raw_counts,
                self.pipeline,
                self.pipeline_revision,
                self.renderer.scale,
                self.renderer.rotation,
                palette,
            )
        processed = self.pipeline_worker.latest(self.pipeline_revision)
        if processed is not None:
            previous_error = self.pipeline_error
            _, image, source, self.pipeline_elapsed_ms, self.pipeline_error = processed
            if self.pipeline_error and self.pipeline_error != previous_error:
                self.api.LOG.warning("Image pipeline failed: %s", self.pipeline_error)
                self.notify(f"Image pipeline: {self.pipeline_error}")
            if image is not None:
                self.rendered = self.api.replace(self.rendered, image=image, image_source=source)
        if (
            not self.fresh_frame
            and self.last_frame_at is not None
            and self.api.time.monotonic() - self.last_frame_at >= 0.5
        ):
            self.rendered = self.api.replace(
                self.rendered,
                measurements_valid=False,
                measurement_status="Waiting for camera frame; readings held",
            )
            self.renderer.measurement_status = self.rendered.measurement_status
            self.renderer.request_measurement_restart()
        self.diagnostics.measurements(self.rendered.measurement_status)
        self.diagnostics.stage("calibration_and_layout")
        if (
            self.startup_calibration_pending
            and self.calibration_available
            and self.rendered.measurements_valid
        ):
            self.startup_calibration_pending = False
            try:
                self.hardware.calibrate_now()
                self.renderer.request_measurement_restart()
                self.notify("Startup calibration requested")
            except self.api.CameraError as exc:
                self.hardware.error = f"Could not run startup calibration: {exc}"
                self.api.LOG.warning("%s", self.hardware.error)
                self.notify(self.hardware.error)
        self.actual_image_source = self.rendered.image_source
        try:
            if self.emissivity_calibration.active and self.rendered.measurements_valid:
                previous_reference = self.emissivity_calibration.reference
                self.emissivity_calibration.update(
                    self.api.raw_temperatures(self.rendered.raw_counts, offset=50),
                    self.api.time.monotonic(),
                )
                if previous_reference != self.emissivity_calibration.reference:
                    self.persist_settings()
        except (self.api.CameraError, ValueError) as exc:
            self.notify(f"Emissivity calibration failed: {exc}")
            try:
                self.emissivity_calibration.cancel()
            except self.api.CameraError as restore_exc:
                self.hardware.error = f"Could not restore emissivity: {restore_exc}"
        try:
            previous_reference = self.reflected_calibration.reference
            if self.rendered.measurements_valid:
                self.reflected_calibration.update(
                    self.api.raw_temperatures(self.rendered.raw_counts, offset=50),
                    self.api.time.monotonic(),
                )
            if previous_reference != self.reflected_calibration.reference:
                self.persist_settings()
                self.renderer.reset_measurement_average()
                self.notify(self.reflected_calibration.message)
        except (self.api.CameraError, ValueError) as exc:
            self.hardware.error = f"Reflected-temperature calibration failed: {exc}"
            self.notify(self.hardware.error)
            try:
                self.reflected_calibration.cancel()
            except self.api.CameraError as restore_exc:
                self.hardware.error = (
                    f"Could not restore reflector measurement settings: {restore_exc}"
                )
        if (
            self.rendered.measurements_valid
            and not self.graphs.logging
            and self.distance_calibration.update(self.rendered.temperatures_celsius)
        ):
            self.notify(self.distance_calibration.message)
        self.layout = self.api.toolbar_layout(self.rendered.image.shape[1])
        if not self.initial_window_size_set:
            width, height = self.saved_window_size or (
                self.rendered.image.shape[1] * (2 if self.show_graph else 1),
                self.rendered.image.shape[0] + self.layout.height,
            )
            self.requested_window_size = (width, height)
            self.api.cv2.resizeWindow(self.api.WINDOW_NAME, width, height)
            self.initial_window_size_set = True
        self.event_viewport = self.api.mouse_viewport_size()
        current_size = self.api.window_resize_size(self.api.WINDOW_NAME)
        if current_size is not None:
            self.last_window_size = current_size
        # A popup covering the window center can temporarily prevent native
        # size detection. Keep the canvas and mouse mapping at the last size.
        self.graph_layout = (
            self.api.GraphWindowLayout.fit(
                (self.rendered.image.shape[1], self.rendered.image.shape[0] + self.layout.height),
                self.last_window_size,
            )
            if self.show_graph
            else None
        )
        self.viewport_size = (
            self.graph_layout.camera_viewport(self.event_viewport)
            if self.graph_layout
            else self.event_viewport
        )
        self.spot_drag.update(
            self.picker,
            self.spots,
            self.rendered.image.shape,
            self.renderer.scale,
            self.viewport_size,
            self.layout.height,
            locked=self.spots_state()["locked"],
        )
        pointer_toolbar_height = self.layout.height
        pointer_image_height = self.rendered.image.shape[0]
        if self.graph_layout:
            pointer_toolbar_height = round(
                self.layout.height
                * self.graph_layout.camera_size[1]
                / (self.rendered.image.shape[0] + self.layout.height)
            )
            pointer_image_height = self.graph_layout.canvas_size[1] - pointer_toolbar_height
        self.display, selected = (
            (self.rendered.image, None)
            if self.emissivity_calibration.active or self.reflected_calibration.active
            else self.api.draw_picker(
                self.rendered,
                self.picker,
                self.renderer.scale,
                viewport_size=self.viewport_size,
                toolbar_height=self.layout.height,
                temperature_unit=self.renderer.temperature_unit,
                spots=self.spots,
                dragging_spot=self.spot_drag.number is not None,
                pointer_over_image=self.pointer_monitor.over_image(
                    pointer_image_height, pointer_toolbar_height
                ),
            )
        )
        if selected is not None:
            self.last_selected = selected

    def draw_status(self: DesktopSession) -> None:
        """
        Draw status for the current desktop session.
        """
        if self.rendered.measurement_status:
            self.status = self.rendered.measurement_status
        elif self.reflected_calibration.active:
            self.display = self.api.draw_reflector_target(
                self.display,
                self.reflected_calibration,
                self.renderer.scale,
                self.renderer.temperature_unit,
            )
        elif not self.emissivity_calibration.active:
            self.display = self.api.draw_sample_spots(
                self.display,
                self.rendered,
                self.spots,
                self.renderer.scale,
                self.renderer.temperature_unit,
            )
            self.display = self.api.draw_distance_selection(
                self.display, self.distance_calibration, self.renderer.scale
            )
        else:
            self.display = self.api.draw_emissivity_point(
                self.display,
                self.emissivity_calibration,
                self.renderer.scale,
                self.renderer.temperature_unit,
            )
        if self.show_instructions:
            self.display = self.api.draw_control_instructions(self.display)
        if self.recorder.is_recording:
            recording_mode = self.recorder.mode
            assert recording_mode is not None
            self.status = (
                f"REC {recording_mode} | {self.recorder.elapsed_seconds():.1f}s | "
                f"{self.recorder.frames_written} frames"
            )
            if self.recorder.mode == "timelapse":
                self.status += f" | {self.timelapse_fpm}/min"
        elif self.pending_save_kind:
            self.status = (
                "Choose a filename for temperature logging..."
                if self.pending_save_kind == "graph_log"
                else f"Choose a filename for {self.pending_save_kind}..."
            )
        elif self.reflected_calibration.active:
            self.status = self.reflected_calibration.message
        elif self.emissivity_calibration.active:
            self.status = self.emissivity_calibration.message
        elif self.distance_calibration.selecting:
            self.status = self.distance_calibration.message
        elif self.api.time.monotonic() < self.status_until:
            self.status = self.status_message
        elif self.api.MODEL_DOWNLOADS.status():
            self.status = self.api.MODEL_DOWNLOADS.status()
        else:
            self.status = "Ready"
        self.capture_panel.update(self.capture_state())
        self.display = self.api.draw_toolbar(
            self.display,
            self.renderer.ambient_celsius,
            self.renderer.temperature_unit,
            placing_spots=self.spots.placing,
            recording_mode=self.recorder.mode,
            pending_recording=self.pending_save_kind
            if self.pending_save_kind in ("video", "timelapse")
            else None,
            status=self.status,
            stats=self.rendered.stats,
            show_graph=self.show_graph,
            graph_locked=self.graphs.logging or self.pending_save_kind == "graph_log",
            settings_locked=self.graphs.logging
            or self.emissivity_calibration.running
            or self.reflected_calibration.running,
            spots_locked=self.spots_state()["locked"],
        )

        # Capture spot values in this frame's orientation, before click actions can rotate
        # coordinates or add/clear spots. The next frame supplies any changed selection.
        self.graph_spots = (
            tuple(
                (
                    (self.spots.generation, number - 1),
                    float(self.rendered.temperatures_celsius[y, x]),
                )
                for number, (x, y) in self.spots.active
            )
            if self.show_graph
            else ()
        )

    def handle_clicks(self: DesktopSession) -> bool:
        """
        Handle clicks for the current desktop session.
        """
        quit_requested = False
        for click_x, click_y in self.picker.consume_context_clicks():
            if (
                self.api.image_position_at(
                    click_x,
                    click_y,
                    self.rendered.image.shape,
                    self.viewport_size,
                    self.layout.height,
                )
                is not None
            ):
                try:
                    self.api.LOG.info("Opening measuring-spot menu")
                    self.spots_panel.open(self.spots_state())
                except OSError as exc:
                    self.notify(f"Could not open spot menu: {exc}")
        for click_x, click_y in self.picker.consume_clicks():
            if (
                self.show_graph
                and self.graph_layout
                and self.graph_layout.graph_control_at(
                    click_x, click_y, self.api.graph_reset_rect, self.event_viewport
                )
            ):
                self.graph_interval_editor.text = None
                if self.graph_config_state()["locked"]:
                    self.notify("Stop graph logging before clearing chart data.")
                else:
                    self.graphs.clear_history()
                    self.notify("Chart data cleared")
                continue
            if (
                self.show_graph
                and self.graph_layout
                and self.graph_layout.graph_control_at(
                    click_x, click_y, self.api.graph_config_rect, self.event_viewport
                )
            ):
                self.graph_interval_editor.text = None
                try:
                    self.graph_panel.open(self.graph_config_state())
                except OSError as exc:
                    self.notify(f"Could not open graph configuration: {exc}")
                continue
            if (
                self.show_graph
                and self.graph_layout
                and self.graph_layout.graph_control_at(
                    click_x, click_y, self.api.graph_interval_rect, self.event_viewport
                )
            ):
                if self.graphs.logging or self.pending_save_kind == "graph_log":
                    self.notify("Stop graph logging before changing the update interval.")
                else:
                    self.graph_interval_editor.begin()
                    self.notify("Update interval in seconds: Enter to apply, Esc to cancel.")
                continue
            self.graph_interval_editor.text = None
            if (
                self.show_graph
                and self.graph_layout
                and self.graph_layout.graph_control_at(
                    click_x, click_y, self.api.graph_log_button_rect, self.event_viewport
                )
            ):
                self.toggle_graph_logging()
                continue
            action = self.api.toolbar_action_at(
                click_x,
                click_y,
                self.rendered.image.shape[1],
                viewport_size=self.viewport_size,
                canvas_height=self.display.shape[0],
            )
            if action == "capture":
                self.open_capture()
            elif action == "rotate":
                self.rotate_view()
            elif action == "spots":
                self.toggle_spots()
            elif action == "view":
                self.open_view()
            elif action == "unit":
                self.toggle_temperature_unit()
            elif action == "graph":
                self.toggle_graph()
            elif action == "help":
                self.show_instructions = not self.show_instructions
            elif action == "quit":
                quit_requested = True
            elif action is None and self.reflected_calibration.active:
                continue
            elif action is None and self.emissivity_calibration.active:
                if self.emissivity_calibration.selecting and self.rendered.measurements_valid:
                    position = self.api.image_position_at(
                        click_x,
                        click_y,
                        self.rendered.image.shape,
                        self.viewport_size,
                        self.layout.height,
                    )
                    if position is not None:
                        pixel = (
                            position[0] // self.renderer.scale,
                            position[1] // self.renderer.scale,
                        )
                        self.emissivity_calibration.select_point(
                            pixel,
                            self.rendered.temperatures_celsius.shape[::-1],
                            float(
                                self.api.raw_temperatures(
                                    self.rendered.raw_counts[pixel[1], pixel[0]], offset=50
                                )
                            ),
                        )
                        self.notify(self.emissivity_calibration.message)
            elif action is None and self.distance_calibration.selecting:
                continue
            elif action is None and self.spots.placing:
                if self.spots_state()["locked"]:
                    self.notify(
                        "Stop logging or close calibration before changing measuring spots."
                    )
                    continue
                position = self.api.image_position_at(
                    click_x,
                    click_y,
                    self.rendered.image.shape,
                    self.viewport_size,
                    self.layout.height,
                )
                self.spots.add(
                    (position[0] // self.renderer.scale, position[1] // self.renderer.scale)
                    if position is not None
                    else None
                )
        if quit_requested:
            return False
        if (
            self.spots.saved_state(
                self.renderer.rotation,
                self.renderer.mirror_horizontal,
                self.renderer.mirror_vertical,
            )
            != self.last_saved_spots
        ):
            self.persist_settings()
        self.view_panel.update(self.view_state())
        self.spots_panel.update(self.spots_state())
        self.graph_panel.update(self.graph_config_state())
        return True

    def present_and_handle_keys(self: DesktopSession) -> bool:
        """
        Present and handle keys for the current desktop session.
        """
        if self.show_graph:
            self.diagnostics.stage("graphs")
            if self.graph_layout is None:
                self.graph_layout = self.api.GraphWindowLayout.fit(
                    (self.display.shape[1], self.display.shape[0]),
                    self.api.window_resize_size(self.api.WINDOW_NAME) or self.last_window_size,
                )
            graph_layout = self.graph_layout
            assert graph_layout is not None
            graph_size = graph_layout.graph_size
            self.graphs.submit(
                self.api.GraphSnapshot(
                    stats=(
                        self.rendered.stats.minimum,
                        self.rendered.stats.average,
                        self.rendered.stats.maximum,
                        self.rendered.stats.center,
                    ),
                    spots=self.graph_spots,
                    spot_names=tuple(
                        ((self.spots.generation, number - 1), self.spots.name(number))
                        for number, _point in self.spots.active
                    ),
                    size=graph_size,
                    unit=self.renderer.temperature_unit,
                    measurements_valid=self.rendered.measurements_valid,
                )
            )
            self.graph_image = self.graphs.image(graph_size, resize=True).copy()
            self.api.draw_graph_logging_control(
                self.graph_image,
                self.graphs.logging,
                self.pending_save_kind == "graph_log",
                interval=self.graph_interval,
                edit_text=self.graph_interval_editor.text,
            )
            self.display = graph_layout.compose(self.display, self.graph_image)
        else:
            self.graphs.pause()
        self.diagnostics.stage("imshow")
        self.api.cv2.imshow(self.api.WINDOW_NAME, self.display)
        self.diagnostics.stage("waitKey")
        key = self.api.cv2.waitKey(1) & 0xFF
        self.diagnostics.stage("window_geometry")
        current_size = self.api.window_resize_size(self.api.WINDOW_NAME)
        if current_size is not None:
            self.last_window_size = current_size
        try:
            if (
                self.api.cv2.getWindowProperty(self.api.WINDOW_NAME, self.api.cv2.WND_PROP_VISIBLE)
                == 0
            ):
                return False
        except self.api.cv2.error:
            if self.last_window_size is not None:
                return False
        if self.graph_interval_editor.text is not None:
            try:
                updated_interval = self.graph_interval_editor.key(key)
                if updated_interval is not None:
                    if self.graphs.logging or self.pending_save_kind == "graph_log":
                        self.notify("Stop graph logging before changing the update interval.")
                    else:
                        self.graphs.set_interval(updated_interval)
                        self.graph_interval = updated_interval
                        self.graph_interval_editor.value = updated_interval
                        self.persist_settings()
                        self.notify(f"Graph update interval: {self.graph_interval:g} seconds.")
            except ValueError:
                self.notify("Enter an update interval between 0.1 and 60 seconds.")
            return True
        self.diagnostics.stage("keyboard_controls")
        if key in (ord("q"), 27):
            return False
        if key == ord("o"):
            self.rotate_view()
        elif key == ord("p"):
            self.toggle_spots()
        elif key == ord(" "):
            self.show_instructions = not self.show_instructions
        elif key == ord("g"):
            self.toggle_graph()
        elif key == ord("l"):
            self.toggle_graph_logging()
        elif key == ord("f"):
            self.toggle_temperature_unit()
        elif key == ord("v"):
            self.open_view()
        elif key == ord("s"):
            self.request_save()
        elif key == ord("c"):
            self.open_capture()
        return True
