"""Nonblocking bridge to graph configuration."""

from .capture_panel import CapturePanel


class GraphPanel(CapturePanel):
    window_module = "based_thermals.graph_window"
    window_label = "Graph configuration"
