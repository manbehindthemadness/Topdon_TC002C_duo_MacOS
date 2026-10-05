"""Nonblocking bridge to the measuring-spot context menu."""

from .capture_panel import CapturePanel


class SpotsPanel(CapturePanel):
    window_module = "topdon_duo.spots_window"
    window_label = "Measuring spots"
