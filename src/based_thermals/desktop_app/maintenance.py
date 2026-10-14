"""
Maintenance request validation and cleanup after all viewer writers have closed.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, cast

from ..maintenance import MaintenancePaths, MaintenancePlan
from ..maintenance.logging import save_debug
from .state import SessionState

if TYPE_CHECKING:
    from .session import DesktopSession


class MaintenanceController(SessionState):
    """
    Schedule confirmed cleanup without allowing the Qt process to choose arbitrary paths.
    """

    def open_maintenance(self: DesktopSession) -> None:
        """
        Refresh remembered output locations before presenting the reviewable cleanup scope.
        """
        self.maintenance_paths = MaintenancePaths.discover(self.args.output)
        self.maintenance_panel.open(self.maintenance_state())

    def maintenance_state(self: DesktopSession) -> dict:
        """
        Expose cleanup locations and existing recording/logging/calibration locks.
        """
        locked = (
            self.graphs.logging
            or self.emissivity_calibration.active
            or self.reflected_calibration.active
            or self.distance_calibration.selecting
            or bool(self.recorder.mode)
            or self.pending_save_kind is not None
            or self.hardware.tone_busy
            or self.support_export.busy
        )
        return {
            "locked": locked,
            "paths": self.maintenance_paths.descriptions(),
            "debug_logs": self.debug_logs,
            "export_busy": self.support_export.busy,
            "export_status": self.support_status,
        }

    def set_debug_logs(self: DesktopSession, enabled: object) -> None:
        """
        Persist a strict checkbox value and switch diagnostic output without restarting capture.
        """
        if type(enabled) is not bool:
            raise ValueError("Debug logs must be enabled or disabled")
        enabled = cast(bool, enabled)
        if self.support_export.busy:
            raise ValueError("Wait for the support export to finish")
        if enabled:
            self.support_logging.start(True)
        path = self.args.diagnostics
        if enabled and path is None:
            path = self.support_logging.directory / (
                "viewer-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl"
            )
        self.diagnostics.configure(path)
        save_debug(enabled)
        self.debug_logs = enabled
        self.support_logging.set_debug(enabled or self.args.verbose)
        self.api.LOG.info("Debug logging %s", "enabled" if enabled else "disabled")

    def export_support(self: DesktopSession, command: dict) -> None:
        """
        Copy current settings and schedule an archive; retain saved originals as evidence.
        """
        kind, filename = command.get("kind"), command.get("path")
        if kind not in ("settings", "all") or not isinstance(filename, str) or not filename:
            raise ValueError("Choose settings or all and a ZIP filename")
        destination = Path(filename)
        if not destination.is_absolute() or destination.suffix.lower() != ".zip":
            raise ValueError("Choose an absolute .zip archive filename")
        if self.support_export.busy:
            raise ValueError("A support export is already running")
        paths = MaintenancePaths.discover(self.args.output)
        snapshot = deepcopy(
            {
                "settings": self.settings_snapshot(),
                "session": {
                    "camera": self.camera_profile.id,
                    "camera_profile": self.camera_profile.as_dict(),
                    "apple": self.apple_capability,
                    "nvidia": self.nvidia_capability,
                    "pipeline_error": self.pipeline_error,
                    "status": self.status,
                    "debug_logs": self.debug_logs,
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(self.args).items()
                    },
                },
            }
        )
        self.support_export.start(
            destination,
            paths,
            snapshot,
            all_data=kind == "all",
            diagnostics=self.args.diagnostics,
            flush_logs=self.support_logging.flush,
        )
        self.support_status = "Creating support archive…"
        self.notify(self.support_status)

    def poll_maintenance_commands(self: DesktopSession) -> bool:
        """
        Stop the frame loop only after a valid, confirmed, unlocked maintenance request.
        """
        result = self.support_export.poll()
        if result is not None:
            self.support_status = result
            self.notify(result)
            self.api.LOG.info("%s", result)
        if self.maintenance_panel.is_open:
            self.maintenance_panel.update(self.maintenance_state())
        for command in self.maintenance_panel.poll():
            try:
                if command.get("action") == "error":
                    self.notify(f"Maintenance window failed: {command.get('message', '')}")
                elif command.get("action") == "debug_logs":
                    self.set_debug_logs(command.get("enabled"))
                elif command.get("action") == "export":
                    self.export_support(command)
                elif command.get("action") == "cleanup":
                    if self.maintenance_state()["locked"]:
                        raise ValueError(
                            "Stop recording/logging or finish calibration before maintenance"
                        )
                    self.maintenance_plan = MaintenancePlan.from_command(
                        command, self.maintenance_paths
                    )
                    self.notify("Closing the viewer for maintenance…")
                    return False
            except (OSError, ValueError, TypeError) as exc:
                self.support_status = f"Maintenance rejected: {exc}"
                self.notify(self.support_status)
        return True

    def finish_maintenance(self: DesktopSession) -> int:
        """
        Run cleanup after shutdown, refusing deletion while installers or inference still run.
        """
        if self.maintenance_plan is None:
            return 0
        try:
            self.api.MODEL_DOWNLOADS.stop()
            thread = getattr(self.pipeline_worker, "thread", None)
            if isinstance(thread, Thread):
                thread.join(timeout=35)
                if thread.is_alive():
                    raise TimeoutError(
                        "Inference worker is still active; maintenance was not performed"
                    )
            count, size = self.maintenance_plan.execute()
            self.api.LOG.info(
                "Maintenance complete: removed %d files (%.1f MB). Start the app again for fresh initialization.",
                count,
                size / 1_000_000,
            )
            return 0
        except (OSError, ValueError) as exc:
            self.api.LOG.error("Maintenance incomplete: %s", exc)
            return 1
