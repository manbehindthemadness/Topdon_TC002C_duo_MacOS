from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from topdon_duo import desktop

from .frames import make_frame


@pytest.fixture(name="viewer")
def viewer_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    # These interaction/recording tests use fixed 768px toolbar coordinates.
    # Exercise the supported legacy scale explicitly, not the startup default.
    """
    Viewer.
    """
    parse_args = desktop.parse_args
    monkeypatch.setattr(desktop, "parse_args", lambda argv: parse_args(["--scale", "3", *argv]))
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    hardware = Mock(original={"loaded": True}, enabled=set(), error="", preview_active=False)
    hardware.processing_preset = "balanced"
    hardware.gamma = 50
    hardware.boost = 0
    hardware.tone_busy = False
    hardware._tone_sent = 0
    hardware.fixed_range = False
    from topdon_duo.hardware_controls import HARDWARE_CONTROLS

    hardware.state.return_value = {
        name: {"value": spec.minimum, "enabled": False, "available": True}
        for name, spec in HARDWARE_CONTROLS.items()
    }
    hardware.state.return_value["ambient"]["value"] = 30.0
    from copy import deepcopy

    hardware.state.side_effect = lambda: deepcopy(hardware.state.return_value)

    def set_hardware(name: Any, value: Any, enabled: Any) -> None:
        """
        Set hardware.
        """
        hardware.state.return_value[name].update(value=value, enabled=enabled)
        if enabled:
            hardware.enabled.add(name)
        else:
            hardware.enabled.discard(name)

    hardware.set.side_effect = set_hardware
    baseline = {name: value["value"] for name, value in hardware.state.return_value.items()}

    def restore_hardware() -> None:
        """
        Restore hardware.
        """
        for name, value in hardware.state.return_value.items():
            value.update(value=baseline[name], enabled=False)
        hardware.enabled.clear()
        hardware.fixed_range = False
        hardware.gamma, hardware.boost, hardware.processing_preset = 50, 0, "balanced"

    hardware.restore.side_effect = restore_hardware
    hardware.set_processing_preset.side_effect = lambda value: setattr(
        hardware, "processing_preset", value
    )
    hardware.set_fixed_range.side_effect = lambda value: setattr(hardware, "fixed_range", value)
    hardware.restore_fixed_range.side_effect = lambda: setattr(hardware, "fixed_range", False)
    hardware.set_tone.side_effect = lambda gamma, boost: (
        setattr(hardware, "gamma", gamma),
        setattr(hardware, "boost", boost),
    )
    monkeypatch.setattr(desktop, "HardwareControls", lambda _camera: hardware)
    graphs = Mock(logging=False)
    graphs.take_logging_error.return_value = None

    def start_logging(path: Any) -> Any:
        """
        Start logging.
        """
        graphs.logging = True
        return path.with_suffix(".csv")

    def stop_logging() -> Any:
        """
        Stop logging.
        """
        graphs.logging = False
        return dialog.selected.with_suffix(".csv")

    graphs.start_logging.side_effect = start_logging
    graphs.stop_logging.side_effect = stop_logging
    graphs.image.side_effect = lambda size, **_kwargs: np.zeros((size[1], size[0], 3), np.uint8)
    monkeypatch.setattr(desktop, "GraphWorker", lambda: graphs)
    awake = Mock()
    awake.check.return_value = None
    monkeypatch.setattr(desktop, "DisplayAwake", lambda: awake)
    camera = Mock()
    camera.frames.return_value = [make_frame()] * 30
    monkeypatch.setattr(desktop, "TC002CDuoCamera", lambda: camera)

    class FramePump:
        def __init__(self, source: Any) -> None:
            """
            Init.
            """
            self._source = source

        def __iter__(self) -> Any:
            """
            Iter.
            """
            return iter(self._source.frames())

        def close(self) -> None:
            """
            Close.
            """

    monkeypatch.setattr(desktop, "CameraFramePump", FramePump)
    monkeypatch.setattr(desktop, "mouse_viewport_size", lambda: None)
    monkeypatch.setattr(desktop, "window_resize_size", lambda _name: None)
    pointer = Mock()
    pointer.over_image.return_value = True
    monkeypatch.setattr(desktop, "PointerMonitor", lambda _name: pointer)
    panel = Mock()
    panel.events = []
    panel.state = {"capture_cursor": False, "frames_per_minute": 60}

    def poll_panel() -> Any:
        """
        Poll panel.
        """
        events, panel.events = panel.events, []
        return events

    panel.poll.side_effect = poll_panel
    panel.update.side_effect = lambda state: setattr(panel, "state", state.copy())
    monkeypatch.setattr(desktop, "CapturePanel", lambda: panel)
    for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "setTrackbarPos"):
        monkeypatch.setattr(desktop.cv2, name, Mock())
    callbacks = {}
    monkeypatch.setattr(
        desktop.cv2,
        "createTrackbar",
        lambda name, _window, _value, _maximum, callback: callbacks.update({name: callback}),
    )
    set_mouse = Mock()
    monkeypatch.setattr(desktop.cv2, "setMouseCallback", set_mouse)
    displayed = []
    monkeypatch.setattr(desktop.cv2, "imshow", lambda _name, image: displayed.append(image.copy()))
    clock = [0.0]
    monkeypatch.setattr(desktop.time, "monotonic", lambda: clock[0])
    writer = Mock()
    writer.isOpened.return_value = True
    writer.frames = []
    writer.write.side_effect = lambda image: writer.frames.append(image.copy())
    monkeypatch.setattr(desktop.cv2, "VideoWriter", Mock(return_value=writer))

    class Dialog:
        def __init__(self) -> None:
            """
            Init.
            """
            self.is_open = False
            self.polls = 0
            self.selected = tmp_path / "test recording.mp4"
            self.open_calls = []
            self.close_calls = 0

        def open(self, directory: Any, **options: Any) -> Any:
            """
            Open.
            """
            self.open_calls.append((directory, options))
            self.is_open = True
            self.polls = 0
            return True

        def poll(self) -> Any:
            """
            Poll.
            """
            if not self.is_open:
                return False, None
            self.polls += 1
            if self.polls == 1:
                return False, None
            self.is_open = False
            return True, self.selected

        def close(self) -> None:
            """
            Close.
            """
            self.close_calls += 1
            self.is_open = False

    dialog = Dialog()
    monkeypatch.setattr(desktop, "LinuxSaveDialog", lambda: dialog)
    draw_toolbar = Mock(wraps=desktop.draw_toolbar)
    monkeypatch.setattr(desktop, "draw_toolbar", draw_toolbar)

    def click_control(action: Any) -> None:
        """
        Click control.
        """
        layout = desktop.toolbar_layout(768)
        callback = set_mouse.call_args.args[1]
        if action in ("video", "timelapse"):
            panel.events.append(
                {
                    "action": action,
                    "frames_per_minute": panel.state["frames_per_minute"],
                    "capture_cursor": panel.state["capture_cursor"],
                }
            )
        elif action == "capture_cursor":
            panel.events.append({"action": "cursor", "value": not panel.state["capture_cursor"]})
        elif action == "image":
            panel.events.append({"action": "image"})
        else:
            x0, y0, x1, y1 = layout.buttons[action]
            callback(cv2.EVENT_LBUTTONUP, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        callback(cv2.EVENT_MOUSEMOVE, 333, 301 + layout.height, 0, None)

    def key_events(actions: Any) -> Any:
        """
        Key events.
        """
        actions = iter(actions)

        def wait_key(_delay: Any) -> Any:
            """
            Wait key.
            """
            action = next(actions)
            if action in ("video", "timelapse", "image"):
                click_control(action)
                return -1
            return ord(action) if isinstance(action, str) else action

        return wait_key

    return SimpleNamespace(
        graphs=graphs,
        hardware=hardware,
        camera=camera,
        writer=writer,
        dialog=dialog,
        clock=clock,
        callbacks=callbacks,
        set_mouse=set_mouse,
        displayed=displayed,
        click_control=click_control,
        draw_toolbar=draw_toolbar,
        panel=panel,
        key_events=key_events,
    )
