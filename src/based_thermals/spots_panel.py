"""
Nonblocking JSON bridge to the Qt measurement configuration popup.
"""

from .capture_panel import CapturePanel


class SpotsPanel(CapturePanel):
    window_module = "based_thermals.measurements_window"
    window_label = "Measurements"
