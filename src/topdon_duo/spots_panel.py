"""
Nonblocking JSON bridge to the Qt measurement configuration popup.
"""

from .capture_panel import CapturePanel


class SpotsPanel(CapturePanel):
    window_module = "topdon_duo.measurements_window"
    window_label = "Measurements"
