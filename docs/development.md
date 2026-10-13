# Development and diagnostics

Run commands from the repository root. See [setup](setup.md) for system libraries and USB permissions.

Commands below assume uv is on PATH. For the Ubuntu environment-local bootstrap,
use `.venv/bin/uv sync --inexact --dev` and `.venv/bin/uv run …` instead.

## Development

Application-owned Python files, including tests, must stay at or below 800
physical lines. Split larger modules by responsibility into package folders;
`tests/test_repository_layout.py` enforces the limit during the normal test run.
Project skills stay version-controlled under `.agents/skills/`.

The public `desktop.py`, `pipeline.py`, `pipeline_editor.py`, and
`pipeline_processing.py` modules retain the entry points and imports used by
callers. Their implementation lives in these packages:

- `camera_backends/duo/`: Duo capture, frame assembly, queued USB transfers,
  hardware controls and the macOS authorized USB helper. These transport details
  are specific to the Duo. `camera_backends/virtual_dual/` owns the synthetic camera;
  contracts, registration and generic control ownership remain in `camera_backends/`.
  Root camera/control/USB modules preserve public imports as thin facades.
- `desktop_app/`: a session coordinator with settings, capture, hardware-control,
  and presentation controllers; shared typed session state; toolbar/graph layout,
  spot handling, overlays, dialogs, and capture helpers.
- `pipeline_model/`: node catalog, validation, migration, and branch dependencies.
- `pipeline_ui/`: the Qt editor, widgets, and runtime state updates.
- `viewer_ui/` and `viewer_window.py`: the Qt main window, toolbar, image canvas,
  pointer mapping and fullscreen behavior. `viewer_panel.py` bridges the session
  using JSON controls and one acknowledged memory-mapped display buffer; frames
  are dropped while Qt is still copying the previous frame rather than queued.
- `processing/`: image operations, per-branch processing and model caches,
  dependency scheduling, and the asynchronous worker.

Controllers use public renderer and hardware APIs to restart measurement
averaging, inspect cached state, and report upload progress. Keep camera reads
and audited hardware setup order intact when changing these boundaries. Qt
viewer and popup processes communicate through JSON separately from the capture
and processing process. No Qt 6 binding is loaded alongside OpenCV's Qt binding.
Shared test fixtures live in `tests/support/`; recording, pipeline, and ONNX
tests are grouped by responsibility so each test module follows the same limit.

Optional external ONNX weight checks are separate from normal pytest collection:

```bash
.venv/bin/python experiments/experiment_onnx_visual.py --models realesr-general-x4v3
# Check all installed catalog weights, plus V1 shape overrides and optional Core ML:
.venv/bin/python experiments/experiment_onnx_visual.py --coreml
```

These checks require locally installed, checksum-valid weights and do not download
models. Missing weights are reported as skipped. Core ML checks require macOS and
an available ONNX Runtime Core ML provider; its compiler may create runtime caches.
See [experiment requirements](../experiments/README.md).

To investigate a Linux viewer stall without reopening its camera, monitor the
running viewer's PID:

```bash
.venv/bin/python tools/watch_ui.py PID --output diagnostics/ui-watch.jsonl
```

The monitor writes one JSON record per second with thread CPU activity, scheduler
waits, memory use, open CSV file sizes, and periodic X11 display power and window
geometry. It stops when the viewer exits or after one hour (`--duration` changes
the limit). CSV buffering and normal waiting can look like inactivity; these are
diagnostic clues rather than measurement counters. Kernel stack availability is
recorded at startup and may be restricted by Linux permissions. The monitor does
not acquire the camera, change power settings, or interrupt the viewer.

For precise diagnostics, launch the desktop viewer with
`--diagnostics diagnostics/viewer.jsonl`. This records received-frame counts,
measurement-validity transitions, and slow stages such as camera acquisition,
rendering and GUI event processing. A watchdog captures all Python thread stacks
in `diagnostics/viewer.stacks.log` when a stage runs for at least two seconds.
This requires starting the viewer with the option; it cannot be attached to an
existing session. A native call holding Python's interpreter lock can delay the
watchdog, so the external monitor remains useful alongside it.
The trace also records USB packet/byte counts, timeouts and frame rejection
reasons, including the most recent rejected length. This distinguishes missing
USB traffic from traffic that does not assemble into a complete thermal frame.

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```
