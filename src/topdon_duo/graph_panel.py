"""Nonblocking bridge to graph configuration."""

from .capture_panel import CapturePanel


class GraphPanel(CapturePanel):
    window_module = "topdon_duo.graph_window"
    window_label = "Graph configuration"
