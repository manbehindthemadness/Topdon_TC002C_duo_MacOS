"""
Save-dialog, graph-log, and recording transitions for a desktop session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class CaptureController(SessionState):
    """
    Own desktop capture operations and transitions.
    """

    def toggle_graph_logging(self: DesktopSession) -> None:
        """
        Toggle graph logging.
        """
        if self.reflected_calibration.active:
            self.notify("Close reflected-temperature calibration before starting logging.")
            return
        if self.emissivity_calibration.active:
            self.notify("Close emissivity calibration before starting temperature logging.")
            return
        if self.graphs.logging:
            try:
                path = self.graphs.stop_logging()
                if path is not None:
                    self.api.restore_user_ownership([path])
                    self.notify(f"Saved temperature log: {path.name}")
            except OSError as exc:
                self.notify(f"Could not finish temperature log: {exc}")
        elif self.pending_save_kind == "graph_log":
            self.save_dialog.close()
            self.pending_save_kind = None
            self.notify("Temperature logging cancelled")
        elif self.save_dialog.is_open:
            self.notify("Finish the current Save dialog before starting temperature logging.")
        elif self.show_graph:
            try:
                if self.save_dialog.open(self.args.output, suffix=".csv", kind="temperatures"):
                    self.pending_save_kind = "graph_log"
                    self.spot_drag.cancel()
            except OSError as exc:
                self.notify(f"Could not choose a temperature log file: {exc}")

    def set_timelapse_fpm(self: DesktopSession, value: int) -> None:
        """
        Set timelapse fpm.
        """
        if not self.recorder.is_recording and self.pending_save_kind not in ("video", "timelapse"):
            self.timelapse_fpm = min(max(value, 1), self.api.TIMELAPSE_MAX_FPM)

    def stop_recording(self: DesktopSession) -> None:
        """
        Stop recording.
        """
        try:
            path = self.recorder.stop()
        except (OSError, self.api.cv2.error) as exc:
            self.notify(f"Could not finalize recording: {exc}")
            return
        if path is not None:
            self.api.restore_user_ownership([path])
            self.notify(f"Saved {path.name} ({self.recorder.frames_written} frames)")
            self.api.LOG.info("Recording saved to %s", path)

    def request_save(self: DesktopSession, kind: str = "image") -> None:
        """
        Request save.
        """
        if kind in ("video", "timelapse"):
            if self.recorder.is_recording:
                if self.recorder.mode == kind:
                    self.stop_recording()
                return
            if self.pending_save_kind == kind:
                self.save_dialog.close()
                self.pending_save_kind = None
                self.notify("Recording cancelled")
                return
        if self.save_dialog.is_open:
            self.notify("Finish or cancel the open Save dialog first")
            return
        try:
            opened = (
                self.save_dialog.open(self.args.output)
                if kind == "image"
                else self.save_dialog.open(self.args.output, suffix=".mp4", kind=kind)
            )
            if opened:
                self.pending_save_kind = kind
        except OSError as exc:
            self.notify(f"Unable to open Save dialog: {exc}")

    def capture_state(self: DesktopSession) -> dict:
        """
        Capture state.
        """
        return {
            "recording_mode": self.recorder.mode,
            "pending_recording": self.pending_save_kind
            if self.pending_save_kind in ("video", "timelapse")
            else None,
            "capture_cursor": self.capture_cursor,
            "capture_graphs": self.capture_graphs,
            "frames_per_minute": self.timelapse_fpm,
            "max_fpm": self.api.TIMELAPSE_MAX_FPM,
            "status": self.status,
        }

    def open_capture(self: DesktopSession) -> None:
        """
        Open capture.
        """
        try:
            self.capture_panel.open(self.capture_state())
        except OSError as exc:
            self.notify(f"Could not open Capture: {exc}")

    def poll_capture_commands(self: DesktopSession) -> None:
        """
        Poll capture commands for the current desktop session.
        """
        for command in self.capture_panel.poll():
            previous_capture_preferences = (
                self.capture_cursor,
                self.capture_graphs,
                self.timelapse_fpm,
            )
            action = command.get("action")
            if action == "error":
                self.notify(f"Capture window failed: {command.get('message', '')}")
            elif action == "rate":
                self.set_timelapse_fpm(int(command["value"]))
            elif action == "graphs":
                if not self.recorder.is_recording and self.pending_save_kind not in (
                    "video",
                    "timelapse",
                ):
                    self.capture_graphs = bool(command["value"])
            elif action == "cursor":
                self.capture_cursor = bool(command["value"])
            elif action == "image":
                self.request_save()
            elif action in ("video", "timelapse"):
                self.set_timelapse_fpm(int(command["frames_per_minute"]))
                self.capture_cursor = bool(command["capture_cursor"])
                if not self.recorder.is_recording and self.pending_save_kind not in (
                    "video",
                    "timelapse",
                ):
                    self.capture_graphs = bool(command.get("capture_graphs", self.capture_graphs))
                self.request_save(action)
            if (
                self.capture_cursor,
                self.capture_graphs,
                self.timelapse_fpm,
            ) != previous_capture_preferences:
                self.persist_settings()

    def finish_capture(self: DesktopSession) -> None:
        """
        Finish capture for the current desktop session.
        """
        logging_error = self.graphs.take_logging_error()
        if logging_error:
            self.notify(logging_error)
        save_finished, save_path = self.save_dialog.poll()
        if save_finished:
            kind, self.pending_save_kind = self.pending_save_kind, None
            if save_path is not None:
                try:
                    if kind == "graph_log":
                        if self.hardware.tone_busy:
                            raise ValueError(
                                "Wait for the camera tone update before starting logging"
                            )
                        self.emissivity_calibration.cancel()
                        path = self.graphs.start_logging(save_path)
                        self.spot_drag.cancel()
                        if self.distance_calibration.selecting:
                            self.distance_calibration.cancel()
                        self.api.restore_user_ownership([path])
                        self.notify(f"Logging temperatures to {path.name}")
                    elif kind in ("video", "timelapse"):
                        self.recording_graphs = self.capture_graphs and self.show_graph
                        self.recorder.start(
                            save_path,
                            (
                                self.rendered.image.shape[0] + self.api.READOUT_HEIGHT,
                                self.rendered.image.shape[1] * (2 if self.recording_graphs else 1),
                                3,
                            ),
                            kind,
                            frames_per_minute=self.timelapse_fpm,
                        )
                        recording_path = self.recorder.path
                        assert recording_path is not None
                        self.notify(f"Recording to {recording_path.name}")
                    else:
                        saved = self.api.save_capture(
                            self.rendered,
                            save_path.parent,
                            self.renderer.ambient_celsius,
                            self.renderer.rotation,
                            self.last_selected,
                            self.renderer.temperature_unit,
                            base_path=save_path,
                            graph_image=(
                                self.graphs.image(
                                    (
                                        self.rendered.image.shape[1],
                                        self.rendered.image.shape[0] + self.api.READOUT_HEIGHT,
                                    ),
                                    resize=True,
                                )
                                if self.capture_graphs and self.show_graph
                                else None
                            ),
                        )
                        self.notify(f"Saved {saved[0].name}")
                except (OSError, ValueError, self.api.cv2.error) as exc:
                    self.notify(f"Capture failed: {exc}")
            else:
                self.notify("Save cancelled")
        # Captures omit application controls; the graph pane is optional.
        # The cursor sampler remains visible locally even when capture is off.
        if self.recorder.is_recording:
            recording_view = (
                self.rendered.image
                if self.emissivity_calibration.active or self.reflected_calibration.active
                else self.api.draw_sample_spots(
                    self.display if self.capture_cursor else self.rendered.image,
                    self.rendered,
                    self.spots,
                    self.renderer.scale,
                    self.renderer.temperature_unit,
                )
            )
            try:
                recording_image = self.api.draw_temperature_readout(
                    recording_view,
                    self.rendered.stats,
                    self.renderer.ambient_celsius,
                    self.renderer.temperature_unit,
                )
                if self.recording_graphs:
                    self.graph_image = (
                        self.graphs.image(
                            (recording_image.shape[1], recording_image.shape[0]), resize=True
                        )
                        if self.show_graph
                        else self.api.np.zeros_like(recording_image)
                    )
                    recording_image = self.api.np.concatenate(
                        (recording_image, self.graph_image), axis=1
                    )
                self.recorder.write(recording_image)
            except (OSError, self.api.cv2.error) as exc:
                self.stop_recording()
                self.notify(f"Recording failed: {exc}")
