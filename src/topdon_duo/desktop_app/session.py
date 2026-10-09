from __future__ import annotations

from typing import Any, cast

from .capture import CaptureController
from .controls import ControlsController
from .presentation import PresentationController
from .settings import SettingsController


class DesktopSession(
    SettingsController, CaptureController, ControlsController, PresentationController
):
    """
    Coordinate one viewer session while controllers own individual responsibilities.
    """

    def __init__(self, api: Any, argv: list[str] | None = None) -> None:
        """
        Initialize dependencies and restore validated session preferences.
        """
        super().__init__(api=api)
        self.args = self.api.parse_args(argv)
        self.saved_settings = self.api.load_settings()
        self.remembered_hardware = self.saved_settings.get("hardware", {}).copy()
        self.advanced_auto = self.saved_settings.get("advanced_auto", True)
        self.auto_calibrate = self.saved_settings.get("auto_calibrate", False)
        self.remembered_fixed_range = self.saved_settings.get("fixed_range", False)
        self.remembered_processing_preset = self.saved_settings.get("processing_preset", "balanced")
        self.remembered_gamma = self.saved_settings.get("camera_gamma", 50)
        self.remembered_boost = self.saved_settings.get("camera_boost", 0)
        self.calibration_available = False
        self.startup_calibration_pending = True
        self.show_graph = self.saved_settings.get("show_graph", False)
        self.graph_interval = cast(
            float, self.saved_settings.get("graph_interval", self.api.GRAPH_INTERVAL)
        )
        self.graph_interval_editor = self.api.GraphIntervalEditor(self.graph_interval)
        self.graph_settings = cast(
            dict[str, Any],
            self.saved_settings.get("graph_settings", self.api.GRAPH_DEFAULTS.copy()),
        )
        self.requested_window_size = None
        self.saved_window_size = (
            None if self.args.reset_window_size else self.api.load_main_window_size()
        )
        self.last_window_size = None
        self.api.logging.basicConfig(
            level=self.api.logging.DEBUG if self.args.verbose else self.api.logging.INFO,
            format="%(levelname)s %(message)s",
        )
        self.apple_capability = self.api.apple_acceleration()
        self.api.LOG.info("Apple acceleration: %s", self.apple_capability["reason"])
        self.renderer = self.api.ThermalRenderer(
            scale=self.args.scale,
            ambient_celsius=None,
            rotation=self.args.rotate
            if self.args.rotate is not None
            else self.saved_settings.get("rotation", 0),
            image_source=self.args.image_source or "preview",
        )
        for name in ("raw_temperature_low", "raw_temperature_high"):
            setattr(
                self.renderer,
                name,
                self.saved_settings.get("display", {}).get(name, self.api.VIEW_DEFAULTS[name]),
            )
        for name, value in self.saved_settings.get("display", {}).items():
            self.renderer.set_view_setting(name, value)
        if self.args.image_source is not None:
            self.renderer.set_view_setting("image_source", self.args.image_source)
        self.pipeline = self.saved_settings.get("pipeline") or self.api.migrate_pipeline(
            self.saved_settings
        )
        if self.args.image_source is not None:
            self.pipeline["software"][0]["params"]["source"] = (
                "raw" if self.args.image_source == "analyze" else self.args.image_source
            )
            if self.args.image_source == "analyze" and not any(
                n["type"] == "range" and not n["bypass"] for n in self.pipeline["software"]
            ):
                self.pipeline["software"].insert(
                    1,
                    self.api.node(
                        "software",
                        "range",
                        low=self.renderer.raw_temperature_low,
                        high=self.renderer.raw_temperature_high,
                    ),
                )
        self.renderer.set_pipeline(self.pipeline)
        self.pipeline_serial = 0
        self.pipeline_revision = 0
        self.pipeline_elapsed_ms = 0.0
        self.pipeline_error = ""
        self.pipeline_worker = None
        self.tone_previous_pipeline = None
        self.renderer.native_temperatures = True
        if self.args.ambient is not None:
            self.api.LOG.warning(
                "--ambient is ignored; set hardware ambient temperature in Camera."
            )
        self.camera = self.api.TC002CDuoCamera()
        self.camera.usb_queue_depth = self.args.usb_queue_depth
        self.display_awake = self.api.DisplayAwake()
        self.picker = self.api.MousePicker()
        self.spots = self.api.SampleSpots()
        if "spots" in self.saved_settings:
            self.spots.restore_saved_state(
                self.saved_settings["spots"],
                self.renderer.rotation,
                self.renderer.mirror_horizontal,
                self.renderer.mirror_vertical,
            )
        self.last_saved_spots = self.spots.saved_state(
            self.renderer.rotation, self.renderer.mirror_horizontal, self.renderer.mirror_vertical
        )
        self.ambient_input_celsius = self.saved_settings.get("ambient_input_celsius")
        self.spot_drag = self.api.SpotDrag()
        self.distance_calibration = self.api.DistanceCalibrator(
            self.saved_settings.get("distance_calibration")
        )
        self.pointer_monitor = self.api.PointerMonitor(self.api.WINDOW_NAME)
        self.save_dialog = (
            self.api.LinuxSaveDialog()
            if self.api.sys.platform.startswith("linux")
            else self.api.MacSaveDialog()
        )
        self.recorder = self.api.VideoRecorder()
        self.capture_panel = self.api.CapturePanel()
        self.view_panel = self.api.ViewPanel()
        self.graph_panel = self.api.GraphPanel()
        self.spots_panel = self.api.SpotsPanel()
        self.hardware = self.api.HardwareControls(self.camera)
        self.pipeline_hardware = self.api.PipelineHardware(self.hardware)
        self.emissivity_calibration = self.api.EmissivityCalibrator(
            self.hardware, self.saved_settings.get("emissivity_calibration")
        )
        self.reflected_calibration = self.api.ReflectedCalibrator(
            self.hardware, self.saved_settings.get("reflected_calibration")
        )
        self.graphs = self.api.GraphWorker()
        self.graphs.set_interval(self.graph_interval)
        self.graphs.configure(self.graph_settings)
        self.actual_image_source = self.renderer.image_source
        self.pending_save_kind: str | None = None
        self.capture_cursor = self.saved_settings.get("capture_cursor", False)
        self.capture_graphs = self.saved_settings.get("capture_graphs", False)
        self.recording_graphs = False
        self.timelapse_fpm = cast(
            int,
            (
                self.args.timelapse_fpm
                if self.args.timelapse_fpm is not None
                else self.saved_settings.get("timelapse_fpm", self.api.TIMELAPSE_DEFAULT_FPM)
            ),
        )
        self.status_message = "Ready"
        self.status_until = 0.0
        self.camera_operation = ""
        self.camera_operation_until = 0.0
        self.camera_operation_busy = False
        self.status = "Ready"
        self.show_instructions = False
        self.last_selected: tuple[int, int] | None = None

    def run(self) -> int:
        """
        Acquire frames, dispatch controllers, and always restore camera state.
        """
        print(
            "Mouse: inspect a pixel | p: toggle spot placement | s: save image data | "
            "c: Capture controls | o: rotate | f: metric/imperial | v: Camera controls | "
            "g: show/hide graph | l: log temperatures | Space: controls | q/Esc: quit"
        )
        try:
            self.diagnostics = self.api.ViewerDiagnostics(self.args.diagnostics)
        except OSError as exc:
            self.api.LOG.error("Could not open diagnostic log: %s", exc)
            return 2
        self.frame_pump = None
        try:
            self.diagnostics.stage("camera_open")
            self.camera.stream_observer = self.diagnostics.stream if self.args.diagnostics else None
            self.camera.rejected_frame_observer = (
                self.diagnostics.rejected_frame if self.args.diagnostics else None
            )
            self.camera.open()
            # Submit bulk reads immediately after UVC COMMIT. In particular, macOS
            # can stop delivering this mode if the SDK/settings handshake occupies
            # the control endpoint before the first stream requests are pending.
            self.diagnostics.stage("stream_start")
            self.frame_pump = self.api.CameraFramePump(self.camera)
            self.diagnostics.stage("hardware_setup")
            self.display_awake.start()
            self.hardware_setup_pending = not self.initialize_hardware()
            self.hardware_retry_deadline = None
            self.next_hardware_retry = 0.0
            self.api.cv2.namedWindow(
                self.api.WINDOW_NAME, self.api.cv2.WINDOW_NORMAL | self.api.cv2.WINDOW_GUI_NORMAL
            )
            self.api.cv2.setMouseCallback(self.api.WINDOW_NAME, self.picker.callback)
            self.api.set_black_window_backgrounds(self.api.WINDOW_NAME)
            self.initial_window_size_set = False
            self.pipeline_worker = self.api.PipelineWorker(
                apple_available=self.apple_capability["available"]
            )
            self.last_frame = None
            self.last_frame_at = None
            for frame in self.diagnostics.frames(self.frame_pump):
                self.frame = frame
                if not self.tick():
                    break
        except self.api.CameraError as exc:
            self.api.LOG.error("Unable to run desktop viewer: %s", exc)
            return 2
        finally:
            self.shutdown()
        return 0

    def tick(self) -> bool:
        """
        Process the latest frame and return whether the viewer should keep running.
        """
        received = self.receive_frame()
        if received is not None:
            return received
        self.poll_spot_and_graph_commands()
        self.poll_camera_commands()
        self.poll_capture_commands()
        self.render_frame()
        self.finish_capture()
        self.draw_status()
        if not self.handle_clicks():
            return False
        result = self.present_and_handle_keys()
        return result

    def shutdown(self: DesktopSession) -> None:
        """
        Shutdown for the current desktop session.
        """
        self.diagnostics.stage("shutdown")
        if self.pipeline_worker is not None:
            self.pipeline_worker.close()
        try:
            self.reflected_calibration.cancel()
        except self.api.CameraError as exc:
            self.api.LOG.error("Could not restore reflector measurement settings: %s", exc)
        try:
            self.emissivity_calibration.cancel()
        except self.api.CameraError as exc:
            self.api.LOG.error("Could not restore pre-calibration emissivity: %s", exc)
        self.persist_settings()
        if self.last_window_size is not None:
            try:
                self.api.save_main_window_size(self.last_window_size)
            except OSError as exc:
                self.api.LOG.warning("Could not save main window size: %s", exc)
        try:
            self.graphs.close()
        except OSError as exc:
            self.api.LOG.error("Could not finalize temperature log: %s", exc)
        self.stop_recording()
        self.capture_panel.close()
        self.view_panel.close()
        self.graph_panel.close()
        self.spots_panel.close()
        self.pointer_monitor.close()
        self.save_dialog.close()
        try:
            self.hardware.restore()
        except (self.api.CameraError, ValueError) as exc:
            self.api.LOG.error("Could not restore camera settings: %s", exc)
        try:
            self.hardware.restore_auto_calibrate()
        except self.api.CameraError as exc:
            self.api.LOG.error("Could not restore automatic camera calibration: %s", exc)
        self.display_awake.close()
        if self.frame_pump is not None:
            self.frame_pump.close()
        self.camera.close()
        self.api.cv2.destroyAllWindows()
        self.diagnostics.close()
