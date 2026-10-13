"""Nonblocking bridge to the display settings window."""

from .capture_panel import CapturePanel


class ViewPanel(CapturePanel):
    window_module = "based_thermals.view_window"
    window_label = "Imaging"
