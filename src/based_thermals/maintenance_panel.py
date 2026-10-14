"""
Nonblocking bridge to the isolated Maintenance dialog.
"""

from .capture_panel import CapturePanel


class MaintenancePanel(CapturePanel):
    """
    Keep Qt maintenance controls separate from capture and inference processes.
    """

    window_module = "based_thermals.maintenance_window"
    window_label = "Maintenance"
