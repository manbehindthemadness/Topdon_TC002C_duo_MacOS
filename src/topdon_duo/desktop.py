"""Native OpenCV desktop viewer with per-pixel inspection and radiometric saves."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .apple_acceleration import apple_acceleration
from .camera import (
    DEFAULT_USB_QUEUE_DEPTH,
    FRAME_RATE,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    CameraError,
    TC002CDuoCamera,
    raw_temperatures,
)
from .camera_backends import DUO_PROFILE, CameraProfile, create_camera
from .capture_panel import CapturePanel
from .dialog_preferences import load_dialog_directory, remember_dialog_directory
from .display_awake import DisplayAwake
from .distance_calibration import DistanceCalibrator
from .emissivity_calibration import EmissivityCalibrator
from .frame_pump import CameraFramePump
from .graph_panel import GraphPanel
from .graph_settings import GRAPH_DEFAULTS, validate_graph_settings
from .graphs import (
    GRAPH_INTERVAL,
    GraphIntervalEditor,
    GraphSnapshot,
    GraphWorker,
    draw_graph_logging_control,
    graph_config_rect,
    graph_interval_rect,
    graph_log_button_rect,
    graph_reset_rect,
)
from .hardware_controls import (
    HARDWARE_CONTROLS,
    HardwareControls,
    HardwareProtocolError,
    camera_operation_title,
)
from .model_downloads import MODEL_DOWNLOADS
from .nvidia_acceleration import nvidia_acceleration
from .pipeline import (
    collapse_previews,
    default_pipeline,
    execution_dependencies,
    geometry,
    migrate_pipeline,
    node,
    software_tabs,
    thermal_source,
    validate_pipeline,
)
from .pipeline_hardware import PIPELINE_FIELDS, PipelineHardware, desired_hardware
from .pipeline_processing import PipelineWorker
from .pointer import PointerMonitor
from .recording import VideoRecorder
from .reflected_calibration import ReflectedCalibrator
from .render import (
    READOUT_HEIGHT,
    RenderedThermalFrame,
    TemperatureStats,
    ThermalRenderer,
    draw_temperature_readout,
)
from .settings_preferences import load_settings, save_settings
from .spot_preferences import validate_spots
from .spots_panel import SpotsPanel
from .view_panel import ViewPanel
from .view_settings import VIEW_DEFAULTS
from .viewer_diagnostics import ViewerDiagnostics
from .window_preferences import load_main_window_size, save_main_window_size
from .window_style import set_black_window_backgrounds, window_resize_size

LOG = logging.getLogger(__name__)
WINDOW_NAME = "TOPDON TC002C Duo"
TOOLBAR_BUTTON_HEIGHT = 24
TOOLBAR_PADDING = 4
TOOLBAR_FONT_SCALE = 0.36
TIMELAPSE_DEFAULT_FPM = 60
TIMELAPSE_MAX_FPM = FRAME_RATE * 60
SAVE_DIALOG_SCRIPT = """
on run argv
    tell current application to activate
    set defaultName to item 1 of argv
    set defaultFolder to item 2 of argv
    if defaultFolder is "" then
        set chosenFile to choose file name with prompt "Save thermal capture" default name defaultName
    else
        set chosenFile to choose file name with prompt "Save thermal capture" default name defaultName default location (POSIX file defaultFolder)
    end if
    return POSIX path of chosenFile
end run
"""


from .desktop_app.captures import restore_user_ownership, save_capture
from .desktop_app.cli import parse_args
from .desktop_app.dialogs import LinuxSaveDialog, MacSaveDialog
from .desktop_app.layout import (
    GraphWindowLayout,
    ToolbarLayout,
    draw_toolbar,
    graph_control_at,
    graph_interval_at,
    graph_logging_button_at,
    image_position_at,
    mouse_viewport_size,
    toolbar_action_at,
    toolbar_layout,
)
from .desktop_app.overlays import (
    draw_contrasting_overlay,
    draw_control_instructions,
    draw_distance_selection,
    draw_emissivity_point,
    draw_label_leader,
    draw_picker,
    draw_reflector_target,
    draw_sample_spots,
    place_temperature_label,
    spot_label_layout,
)
from .desktop_app.session import DesktopSession
from .desktop_app.spots import MousePicker, SampleSpots, SpotDrag, spot_hit


def main(argv: list[str] | None = None) -> int:
    """
    Run the desktop viewer with its public dependency boundary.
    """
    result = DesktopSession(sys.modules[__name__], argv).run()
    return result


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "DEFAULT_USB_QUEUE_DEPTH",
    "DUO_PROFILE",
    "FRAME_RATE",
    "GRAPH_DEFAULTS",
    "GRAPH_INTERVAL",
    "HARDWARE_CONTROLS",
    "LOG",
    "MODEL_DOWNLOADS",
    "READOUT_HEIGHT",
    "SAVE_DIALOG_SCRIPT",
    "SENSOR_HEIGHT",
    "SENSOR_WIDTH",
    "TIMELAPSE_DEFAULT_FPM",
    "TIMELAPSE_MAX_FPM",
    "TOOLBAR_BUTTON_HEIGHT",
    "TOOLBAR_FONT_SCALE",
    "TOOLBAR_PADDING",
    "VIEW_DEFAULTS",
    "WINDOW_NAME",
    "CameraError",
    "CameraFramePump",
    "CameraProfile",
    "CapturePanel",
    "DesktopSession",
    "DisplayAwake",
    "DistanceCalibrator",
    "EmissivityCalibrator",
    "GraphIntervalEditor",
    "GraphPanel",
    "GraphSnapshot",
    "GraphWindowLayout",
    "GraphWorker",
    "HardwareControls",
    "HardwareProtocolError",
    "LinuxSaveDialog",
    "MacSaveDialog",
    "MousePicker",
    "Path",
    "PipelineHardware",
    "PipelineWorker",
    "PointerMonitor",
    "ReflectedCalibrator",
    "RenderedThermalFrame",
    "SampleSpots",
    "SpotDrag",
    "SpotsPanel",
    "TC002CDuoCamera",
    "TemperatureStats",
    "ThermalRenderer",
    "ToolbarLayout",
    "VideoRecorder",
    "ViewPanel",
    "ViewerDiagnostics",
    "apple_acceleration",
    "argparse",
    "asdict",
    "camera_operation_title",
    "collapse_previews",
    "create_camera",
    "cv2",
    "dataclass",
    "datetime",
    "default_pipeline",
    "desired_hardware",
    "draw_contrasting_overlay",
    "draw_control_instructions",
    "draw_distance_selection",
    "draw_emissivity_point",
    "draw_graph_logging_control",
    "draw_label_leader",
    "draw_picker",
    "draw_reflector_target",
    "draw_sample_spots",
    "draw_temperature_readout",
    "draw_toolbar",
    "execution_dependencies",
    "field",
    "geometry",
    "graph_config_rect",
    "graph_control_at",
    "graph_interval_at",
    "graph_interval_rect",
    "graph_log_button_rect",
    "graph_logging_button_at",
    "graph_reset_rect",
    "image_position_at",
    "json",
    "load_dialog_directory",
    "load_main_window_size",
    "load_settings",
    "logging",
    "main",
    "migrate_pipeline",
    "mouse_viewport_size",
    "node",
    "np",
    "nvidia_acceleration",
    "os",
    "parse_args",
    "pipeline_hardware_fields",
    "place_temperature_label",
    "raw_temperatures",
    "remember_dialog_directory",
    "replace",
    "restore_user_ownership",
    "save_capture",
    "save_main_window_size",
    "save_settings",
    "set_black_window_backgrounds",
    "software_tabs",
    "spot_hit",
    "spot_label_layout",
    "subprocess",
    "sys",
    "thermal_source",
    "time",
    "toolbar_action_at",
    "toolbar_layout",
    "validate_graph_settings",
    "validate_pipeline",
    "validate_spots",
    "window_resize_size",
]

# Compatibility aliases for callers using the former desktop helper names.
_restore_user_ownership = restore_user_ownership
_draw_contrasting_overlay = draw_contrasting_overlay
_draw_label_leader = draw_label_leader
_place_temperature_label = place_temperature_label
_spot_label_layout = spot_label_layout
_spot_hit = spot_hit
pipeline_hardware_fields = PIPELINE_FIELDS
