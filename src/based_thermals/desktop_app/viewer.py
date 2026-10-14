"""
Qt viewer state and commands, separated from image rendering and camera controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..graphs import history_duration, validate_graph_interval
from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class ViewerController(SessionState):
    """
    Bridge native controls to the existing guarded session operations.
    """

    def viewer_state(self: DesktopSession) -> dict:
        """
        Return toolbar activity, graph settings and measurement locks.
        """
        self.reconcile_measurement_drawing()
        return {
            "rotation": self.renderer.rotation,
            "viewer_layout": self.viewer_layout,
            "measurement_drawing": self.measurement_drawing.state(),
            "unit": self.renderer.temperature_unit,
            "show_graph": self.show_graph,
            "show_instructions": self.show_instructions,
            "placing_spots": self.spots.placing,
            "recording": bool(self.recorder.mode or self.pending_save_kind in ("video", "timelapse")),
            "logging": self.graphs.logging,
            "log_pending": self.pending_save_kind == "graph_log",
            "graph_locked": self.graphs.logging or self.pending_save_kind == "graph_log",
            "graph_config_locked": self.graph_config_state()["locked"],
            "settings_locked": self.graphs.logging or self.emissivity_calibration.running
            or self.reflected_calibration.running,
            "spots_locked": self.spots_state()["locked"],
            "radiometry": self.camera_profile.radiometry,
            "graph_interval": self.graph_interval,
            "history": "History | " + (
                history_duration(self.graph_settings["range_seconds"])
                if self.graph_settings["range_mode"] == "fixed" else "Entire session"
            ),
            "status": self.status,
            "viewport_draw_ms": self.viewport_draw_ms,
        }

    def poll_viewer_commands(self: DesktopSession) -> bool:
        """
        Consume native input without loading Qt or performing camera work in its process.
        """
        if self.viewer is None:
            return True
        keys = {
            "rotate": "o", "unit": "f", "view": "v",
            "capture": "c", "graph": "g", "help": " ", "graph_log": "l",
        }
        for command in self.viewer.poll():
            action = command.get("action")
            try:
                if action == "quit":
                    return False
                if action == "error":
                    self.notify(f"Viewer failed: {command.get('message', '')}")
                    return False
                if action == "pointer":
                    self.viewer_pointer(command)
                elif action == "spots":
                    self.spots_panel.open(self.spots_state())
                elif action == "orientation":
                    self.set_viewer_orientation(command)
                elif action in ("draw_finish", "draw_cancel", "draw_undo"):
                    self.measurement_drawing_action(action)
                elif isinstance(action, str) and action in keys:
                    self.viewer_keys.append(ord(keys[action]))
                elif action == "key":
                    key = command.get("key")
                    if key == ord("q"):
                        return False
                    if isinstance(key, int) and not isinstance(key, bool) and key in map(ord, "qop glfvsc"):
                        self.viewer_keys.append(key)
                elif action == "graph_config":
                    self.graph_panel.open(self.graph_config_state())
                elif action == "graph_reset":
                    if self.graph_config_state()["locked"]:
                        raise ValueError("Stop graph logging before clearing chart data")
                    self.graphs.clear_history()
                    self.notify("Chart data cleared")
                elif action == "graph_interval":
                    if self.graphs.logging or self.pending_save_kind == "graph_log":
                        raise ValueError("Stop graph logging before changing the update interval")
                    interval = validate_graph_interval(command.get("value"))
                    self.graphs.set_interval(interval)
                    self.graph_interval = interval
                    self.graph_interval_editor.value = interval
                    self.persist_settings()
                    self.notify(f"Graph update interval: {interval:g} seconds.")
            except (OSError, ValueError, TypeError) as exc:
                self.notify(f"Viewer setting rejected: {exc}")
        return not self.viewer.closed

    def set_viewer_orientation(self: DesktopSession, command: dict) -> None:
        """
        Validate popup changes and reuse audited rotation and native-coordinate updates.
        """
        rotation, layout = command.get("rotation"), command.get("layout")
        if type(rotation) is not int or rotation not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180 or 270 degrees")
        if not isinstance(layout, str) or layout not in ("portrait", "landscape"):
            raise ValueError("Layout must be portrait or landscape")
        layout = cast(str, layout)
        if self.viewer_state()["settings_locked"] or self.pending_save_kind == "graph_log":
            raise ValueError("Stop logging or calibration measurement before changing orientation")
        for _step in range(((rotation - self.renderer.rotation) % 360) // 90):
            self.rotate_view()
        if layout == self.viewer_layout:
            return
        self.cancel_measurement_drawing()
        self.spot_drag.cancel()
        self.picker.consume_drag_events()
        self.picker.consume_clicks()
        self.picker.x = self.picker.y = None
        size = self.current_window_size() or self.last_window_size or self.requested_window_size
        self.viewer_layout = layout
        self.graph_layout = None
        if self.show_graph and size is not None:
            requested = (size[1], size[0])
            self.requested_window_size = self.last_window_size = requested
            if self.viewer is not None:
                self.viewer.resize(requested)
            else:
                self.api.cv2.resizeWindow(self.api.WINDOW_NAME, *requested)
        self.persist_settings()
        self.notify(f"Preview layout: {layout}")

    def viewer_pointer(self: DesktopSession, command: dict) -> None:
        """
        Translate validated image-local events into the existing picker/drag contract.
        """
        kind, button = command.get("kind"), command.get("button")
        if not isinstance(kind, str) or not isinstance(button, str):
            return
        events = {
            ("move", "left"): self.api.cv2.EVENT_MOUSEMOVE,
            ("press", "left"): self.api.cv2.EVENT_LBUTTONDOWN,
            ("release", "left"): self.api.cv2.EVENT_LBUTTONUP,
            ("release", "right"): self.api.cv2.EVENT_RBUTTONUP,
        }
        event = events.get((kind, button))
        if event is None:
            return
        x, y, flags = command.get("x"), command.get("y"), command.get("flags", 0)
        if (
            not isinstance(x, int) or not isinstance(y, int)
            or not isinstance(flags, int) or any(isinstance(n, bool) for n in (x, y, flags))
        ):
            return
        if not (-1 <= x <= 16384 and -1 <= y <= 16384 and 0 <= flags <= 63):
            return
        if self.measurement_drawing_pointer(command):
            return
        if hasattr(self, "rendered") and not self.rendered.measurement_mapping_valid and button == "left":
            return
        pointer_event = cast(int, event)
        self.picker.callback(pointer_event, x, y, flags, None)

    def viewer_key(self: DesktopSession) -> int:
        """
        Return one pending shortcut each tick, preserving the existing key handlers.
        """
        key = self.viewer_keys.pop(0) if self.viewer_keys else 255
        return key
