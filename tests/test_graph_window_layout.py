import numpy as np
import pytest

from topdon_duo.desktop import GraphWindowLayout, image_position_at
from topdon_duo.graphs import graph_interval_rect, graph_log_button_rect


@pytest.mark.parametrize("window", [(2400, 1200), (2400, 700), (1000, 1400), (600, 400)])
def test_graphs_fill_window_and_camera_retains_aspect(window):
    layout = GraphWindowLayout.fit((768, 650), window)
    camera_width, camera_height = layout.camera_size
    assert layout.canvas_size == window
    assert camera_width <= window[0] // 2 and camera_height <= window[1]
    assert abs(camera_width / camera_height - 768 / 650) < 0.005
    assert layout.graph_size == (window[0] - camera_width, window[1])
    camera = np.full((650, 768, 3), 100, np.uint8)
    graph = np.full((window[1], layout.graph_size[0], 3), 200, np.uint8)
    canvas = layout.compose(camera, graph)
    assert canvas.shape == (window[1], window[0], 3)
    assert np.all(canvas[:camera_height, :camera_width] == 100)
    assert np.all(canvas[camera_height:, :camera_width] == 0)
    assert np.all(canvas[:, camera_width:] == 200)


@pytest.mark.parametrize("event_scale", [0.5, 1, 1.5])
def test_image_and_graph_controls_map_separately_with_letterboxing(event_scale):
    layout = GraphWindowLayout.fit((768, 650), (2400, 700))
    viewport = tuple(value * event_scale for value in layout.canvas_size)
    camera_viewport = layout.camera_viewport(viewport)
    x = 240 * layout.camera_size[0] / 768 * event_scale
    y = 254 * layout.camera_size[1] / 650 * event_scale
    assert image_position_at(x, y, (576, 768, 3), camera_viewport, 74) == (240, 180)
    # Graph clicks cannot become image measurements or spots.
    assert image_position_at(2390 * event_scale, y, (576, 768, 3), camera_viewport, 74) is None
    for rectangle in (graph_log_button_rect, graph_interval_rect):
        x0, y0, x1, y1 = rectangle(layout.graph_size[0])
        x = (layout.camera_size[0] + (x0 + x1) / 2) * event_scale
        y = (y0 + y1) / 2 * event_scale
        assert layout.graph_control_at(x, y, rectangle, viewport)
        assert not layout.graph_control_at(20, 20, rectangle, viewport)
