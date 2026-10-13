"""
Render real application widgets with synthetic camera data, without USB or downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def demo_pipeline() -> dict[str, Any]:
    """
    Build a CPU-only thermal composition with a grayscale upper-left inset.
    """
    from based_thermals.pipeline import default_pipeline, node, validate_pipeline

    document = default_pipeline()
    document["hardware"] = []
    document["software"] = [
        node("software", "source", source="raw"),
        node("software", "colors", palette="inferno"),
        node("software", "combine", tab="B", mode="replace", opacity=1.0),
        node("software", "output"),
    ]
    document["branches"] = {"B": [
        node("software", "source", source="raw"),
        node("software", "colors", palette="white_hot"),
        node("software", "transform", resize_mode="canvas", width=25, height=25,
             position_unit="percent", position_x=0, position_y=0, method="bicubic"),
    ]}
    for label, nodes in {"A": document["software"], **document["branches"]}.items():
        for index, item in enumerate(nodes):
            item["id"] = f"documentation-{label}-{index}"
    validated = validate_pipeline(document)
    return validated


def render_frames(directory: Path) -> None:
    """
    Use the virtual backend, renderer, region kernels and graph painter for example pixels.
    """
    from dataclasses import replace
    from unittest.mock import patch

    import cv2

    from based_thermals.camera_backends.virtual import VirtualDualCamera
    from based_thermals.desktop_app.captures import save_capture
    from based_thermals.desktop_app.layout import GraphWindowLayout
    from based_thermals.desktop_app.overlays import draw_sample_spots
    from based_thermals.desktop_app.spots import SampleSpots
    from based_thermals.graphs import GraphSnapshot, render_graphs
    from based_thermals.measurements.schema import default_region
    from based_thermals.pipeline import node
    from based_thermals.pipeline_processing import PipelineProcessor
    from based_thermals.render import ThermalRenderer, draw_temperature_readout

    camera = VirtualDualCamera()
    size = camera.profile.native_size
    spots = SampleSpots(native_size=size)
    for name, center, geometry in (
        ("Warm region", (40, 30), {"kind": "rectangle", "points": [[24, 15], [61, 44]]}),
        ("Reference", (12, 40), {"kind": "spot", "radius": 3}),
        ("Exclude peak", (40, 30), {"kind": "oval", "points": [[35, 23], [45, 37]],
                                  "exclude": True}),
    ):
        region = {**default_region(), **geometry, "labels": ["name", "min", "average", "max"]}
        number = spots.create_region(center, region, size)
        spots.rename(number, name)
    document = demo_pipeline()
    plain = {**document, "software": [
        node("software", "source", source="raw"),
        node("software", "colors", palette="inferno"),
        node("software", "output"),
    ], "branches": {}}
    # Avoid discovery of physical accelerators: this documentation exercise uses
    # ordinary CPU pixel operations and never loads a model or runtime installer.
    with patch("based_thermals.render.PipelineProcessor", lambda: PipelineProcessor(False, False)):
        renderer = ThermalRenderer(scale=6, smoothing=1.0, ambient_celsius=None, image_source="raw")
    renderer.pipeline = plain
    master = []
    histories: dict[tuple[int, int], list] = {}
    end_time = round(20 * math.pi * 25) / 25
    try:
        rendered = renderer.render_detailed(camera.make_frame(0))
        for step in range(121):
            now = end_time - 60 + step / 2
            rendered = renderer.render_detailed(camera.make_frame(round(now * 25)))
            stats = rendered.stats
            master.append((now, (stats.minimum, stats.average, stats.maximum, stats.center)))
            for key, values in spots.graph_readings(rendered.temperatures_celsius):
                row = values if isinstance(values, tuple) else (values,)
                histories.setdefault(key, []).append((now, row))
        image = draw_sample_spots(rendered.image, rendered, spots, 6)
        save_capture(
            replace(rendered, image=image), directory, None, 0,
            selected_pixel=(40, 30), base_path=directory / "synthetic-capture",
        )
        image = draw_temperature_readout(image, rendered.stats, None)
        names = tuple(((spots.generation, number - 1), spots.name(number))
                      for number, _center in spots.active)
        for orientation, canvas_size in (("landscape", (1200, 480)), ("portrait", (720, 1040))):
            layout = GraphWindowLayout.fit((image.shape[1], image.shape[0]), canvas_size, orientation)
            snapshot = GraphSnapshot(
                master[-1][1], spots.graph_readings(rendered.temperatures_celsius),
                layout.graph_size, spot_names=names, history_range=60,
            )
            graphs = render_graphs(snapshot, master, histories, end_time)
            if not cv2.imwrite(str(directory / f"frame-{orientation}.png"), layout.compose(image, graphs)):
                raise OSError("Could not write example frame")
        renderer.pipeline = document
        inset = renderer.render_detailed(camera.make_frame(round(end_time * 25)))
        if not cv2.imwrite(str(directory / "frame-inset.png"), inset.image):
            raise OSError("Could not write inset frame")
        metadata = {"sensor_size": list(size), "orientation": [0, False, False],
                    "spots": spots.region_states(), "pipeline": document}
        (directory / "demo.json").write_text(json.dumps(metadata), encoding="utf-8")
    finally:
        renderer.pipeline_processor.close()


def capture_widgets(directory: Path, output: Path) -> None:
    """
    Grab actual Qt widgets in an isolated process using the generated demo state.
    """
    from copy import deepcopy

    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget

    from based_thermals.capture_window import CaptureWindow
    from based_thermals.measurements_window import MeasurementsWindow
    from based_thermals.view_window import ViewWindow
    from based_thermals.viewer_window import ViewerWindow

    app = QApplication([])
    app.setStyle("Fusion")
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(directory / "qt-config"))
    metadata = json.loads((directory / "demo.json").read_text(encoding="utf-8"))

    def save(window: QWidget, name: str) -> None:
        """
        Lay out and capture the real window contents without editing the resulting PNG.
        """
        window.show()
        app.processEvents()
        QTest.qWait(100)
        if not window.grab().save(str(output / name)):
            raise OSError(f"Could not capture {name}")

    def ignore(_message: dict) -> None:
        """
        Keep documentation controls disconnected from cameras, recording and user settings.
        """

    viewer = ViewerWindow(ignore)
    for orientation, size in (("landscape", (1200, 535)), ("portrait", (720, 1120))):
        viewer.canvas.image = QImage(str(directory / f"frame-{orientation}.png"))
        viewer.update_state({
            "unit": "C", "show_graph": True, "viewer_layout": orientation,
            "graph_interval": 0.5, "history": "History | 1.0 min", "radiometry": True,
            "status": "Virtual dual camera | SYNTHETIC data | CPU documentation example",
        })
        viewer.resize(*size)
        save(viewer, f"viewer-{orientation}.png")
    viewer.canvas.image = QImage(str(directory / "frame-inset.png"))
    viewer.update_state({"unit": "C", "show_graph": False, "viewer_layout": "landscape",
                         "status": "Synthetic raw inset | Tab B: Transform 25% | A: Combine Replace"})
    viewer.resize(960, 785)
    save(viewer, "raw-inset.png")
    viewer.hide()
    capture = CaptureWindow(ignore)
    capture.update_state({"frames_per_minute": 60, "capture_graphs": True,
                          "status": "Ready — 60 frames/minute; graphs included"})
    save(capture, "capture-controls.png")
    capture.hide()
    measurements = MeasurementsWindow(ignore)
    measurements.update_state({**metadata, "locked": False})
    measurements.kind.setCurrentIndex(measurements.kind.findData("rectangle"))
    save(measurements, "measurements-editor.png")
    measurements.hide()
    imaging = ViewWindow(ignore)
    state = {"pipeline": metadata["pipeline"], "pipeline_serial": 1,
             "apple_acceleration": {"available": False, "reason": "CPU-only documentation example"},
             "nvidia_acceleration": {"available": False, "reason": "CPU-only documentation example"}}
    document = deepcopy(metadata["pipeline"])
    for item in document["software"]:
        item["expanded"] = item["type"] in {"source", "combine"}
    imaging.update_state({**state, "pipeline": document})
    editor = imaging.pipeline_editor
    imaging.show()
    app.processEvents()
    editor.resize(840, 760)
    for listing in editor.stacks.values():
        listing.fit_contents()
    save(editor, "pipeline-combine.png")
    for item in editor.document["branches"]["B"]:
        item["expanded"] = item["type"] == "transform"
    editor.tab = "B"
    editor.rebuild()
    editor.resize(840, 1100)
    app.processEvents()
    for listing in editor.stacks.values():
        listing.fit_contents()
    save(editor, "pipeline-transform.png")
    # The parent owns this widget; hide rather than change the application's
    # permanent ownership/lifecycle just for the documentation screenshot.
    editor.hide()
    imaging.close()


def main(argv: list[str] | None = None) -> int:
    """
    Generate reproducible screenshots with isolated preferences and separate rendering/Qt phases.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    parser.add_argument("--examples", type=Path, default=Path("docs/examples"))
    parser.add_argument("--phase", choices=("frames", "widgets"), help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.phase:
        if args.work is None:
            parser.error("Internal screenshot phases need a working directory")
        if args.phase == "frames":
            render_frames(args.work)
        else:
            capture_widgets(args.work, args.output)
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="based-thermals-docs-") as temporary:
        work = Path(temporary)
        environment = dict(os.environ)
        environment.update({"QT_QPA_PLATFORM": "offscreen", "BASED_THERMALS_RUNTIME_SETUP": "0",
                            "XDG_CONFIG_HOME": str(work / "config"),
                            "XDG_CACHE_HOME": str(work / "cache")})
        environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        environment.pop("QT_QPA_FONTDIR", None)
        for phase in ("frames", "widgets"):
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
                 "--work", str(work), "--output", str(output)],
                env=environment, check=True, timeout=60,
            )
        examples = args.examples.resolve()
        examples.mkdir(parents=True, exist_ok=True)
        pipeline = json.loads((work / "demo.json").read_text(encoding="utf-8"))["pipeline"]
        (examples / "raw-thermal-inset.pipeline.json").write_text(
            json.dumps(pipeline, indent=2) + "\n", encoding="utf-8",
        )
        for suffix in (".png", ".npz", ".json"):
            source = work / f"synthetic-capture{suffix}"
            shutil.copyfile(source, examples / source.name)
    print(f"Saved application screenshots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
