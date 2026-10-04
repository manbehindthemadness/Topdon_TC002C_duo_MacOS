"""Nonblocking bridge to the display settings window."""

from .capture_panel import CapturePanel


class ViewPanel(CapturePanel):
    window_module = "topdon_duo.view_window"
    window_label = "View"
