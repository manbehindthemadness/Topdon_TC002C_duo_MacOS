"""
Shared desktop state declared explicitly for controller introspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

    import numpy as np

    from .. import desktop as api_types


@dataclass
class SessionState:
    """
    Fields shared between desktop controllers; initialized by DesktopSession.
    """

    api: Any = None
    actual_image_source: str = ""
    advanced_auto: bool = False
    ambient_input_celsius: float | None = None
    apple_capability: dict[str, Any] = field(default_factory=dict)
    args: argparse.Namespace = field(init=False)
    auto_calibrate: bool = False
    calibration_available: bool = False
    camera: api_types.TC002CDuoCamera = field(init=False)
    camera_operation: str = ""
    camera_operation_busy: bool = False
    camera_operation_until: float = 0.0
    capture_cursor: bool = False
    capture_graphs: bool = False
    capture_panel: api_types.CapturePanel = field(init=False)
    diagnostics: api_types.ViewerDiagnostics = field(init=False)
    display: np.ndarray = field(init=False)
    display_awake: api_types.DisplayAwake = field(init=False)
    distance_calibration: api_types.DistanceCalibrator = field(init=False)
    emissivity_calibration: api_types.EmissivityCalibrator = field(init=False)
    event_viewport: tuple[int, int] | None = None
    frame: bytes | None = None
    frame_pump: api_types.CameraFramePump | None = None
    fresh_frame: bool = False
    graph_image: np.ndarray = field(init=False)
    graph_interval: float = 0.0
    graph_interval_editor: api_types.GraphIntervalEditor = field(init=False)
    graph_layout: api_types.GraphWindowLayout | None = None
    graph_panel: api_types.GraphPanel = field(init=False)
    graph_settings: dict[str, Any] = field(default_factory=dict)
    graph_spots: tuple = ()
    graphs: api_types.GraphWorker = field(init=False)
    hardware: api_types.HardwareControls = field(init=False)
    hardware_retry_deadline: float | None = None
    hardware_setup_pending: bool = False
    initial_window_size_set: bool = False
    last_frame: bytes | None = None
    last_frame_at: float | None = None
    last_saved_spots: dict[str, Any] = field(default_factory=dict)
    last_selected: tuple[int, int] | None = None
    last_window_size: tuple[int, int] | None = None
    layout: api_types.ToolbarLayout = field(init=False)
    next_hardware_retry: float = 0.0
    pending_save_kind: str | None = None
    picker: api_types.MousePicker = field(init=False)
    pipeline: dict[str, Any] = field(default_factory=dict)
    pipeline_elapsed_ms: float = 0.0
    pipeline_error: str = ""
    pipeline_hardware: api_types.PipelineHardware = field(init=False)
    pipeline_revision: int = 0
    pipeline_serial: int = 0
    pipeline_worker: api_types.PipelineWorker | None = None
    pointer_monitor: api_types.PointerMonitor = field(init=False)
    recorder: api_types.VideoRecorder = field(init=False)
    recording_graphs: bool = False
    reflected_calibration: api_types.ReflectedCalibrator = field(init=False)
    remembered_boost: int = 0
    remembered_fixed_range: bool = False
    remembered_gamma: int = 0
    remembered_hardware: dict[str, Any] = field(default_factory=dict)
    remembered_processing_preset: str = ""
    rendered: api_types.RenderedThermalFrame = field(init=False)
    renderer: api_types.ThermalRenderer = field(init=False)
    requested_window_size: tuple[int, int] | None = None
    save_dialog: api_types.MacSaveDialog = field(init=False)
    saved_settings: dict[str, Any] = field(default_factory=dict)
    saved_window_size: tuple[int, int] | None = None
    show_graph: bool = False
    show_instructions: bool = False
    spot_drag: api_types.SpotDrag = field(init=False)
    spots: api_types.SampleSpots = field(init=False)
    spots_panel: api_types.SpotsPanel = field(init=False)
    startup_calibration_pending: bool = False
    status: str = ""
    status_message: str = ""
    status_until: float = 0.0
    timelapse_fpm: int = 0
    tone_previous_pipeline: dict[str, Any] | None = None
    view_panel: api_types.ViewPanel = field(init=False)
    viewport_size: tuple[int, int] | None = None
