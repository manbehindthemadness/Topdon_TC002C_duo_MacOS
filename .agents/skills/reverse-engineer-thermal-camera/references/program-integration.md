# Integrating another camera into this program

Paths below describe the repository as inspected on 2026-10-07. Re-read the
current files before implementation; earlier research documents describe old UI
and capture behavior. This is a map, not a claim that a generic adapter exists.

## Current responsibilities

| Files | Responsibility and relevant invariants |
| --- | --- |
| `src/topdon_duo/camera.py` | `TC002CDuoCamera`, UVC negotiation, frame assembly, queued/synchronous reads, Duo decode, temperature conversion and validity detection |
| `queued_usb.py`, `frame_pump.py` | Continuous acquisition; queued bulk transfers, cancellation and latest-complete-frame consumption independent of UI |
| `hardware_controls.py`, `tone_curves.py` | Target-specific blocks, SDK initialization, saved originals, status polling, presets, shutter and incremental LUT upload/restore |
| `pipeline_hardware.py` | Applies owned fields in audited order, preserves calibration/correction fields, skips USB for software/reorder/collapse edits |
| `render.py` | Native-grid measurements, invalid-frame handling, smoothing and geometry; display enhancement is separate |
| `pipeline.py`, `pipeline_processing.py` | Versioned A–D pipelines, sources, filters/features, model processing, Combine masks, per-branch threads and preview timing |
| `desktop.py`, `web.py` | Camera construction, acquisition/render integration and user-facing capability/control state |
| `view_window.py`, `pipeline_editor.py` | Standalone measurement/calibration settings, hardware/software nodes and control availability |
| `settings_preferences.py` | Validated persistence and migration; unit presentation must not change stored wire/calibration values |
| `distance_calibration.py`, `emissivity_calibration.py`, `reflected_calibration.py` | Host calibration tools; preserve saved calibrations and use native measurement pixels |
| `graphs.py`, `graph_window.py`, `recording.py` | Measurement sampling, named spots/logging, recording and graph capture |
| `viewer_diagnostics.py`, `tools/watch_ui.py` | Opt-in stage timing, stream health and thread/stall observations |
| `tools/inspect_telemetry.py` | Offline Duo-frame explorer; hardcodes Duo offsets and is not a generic parser |

The current Duo decoder returns telemetry uint16 words, a 192×256 native count
plane and an 8-bit preview luminance plane. The pipeline has additional Duo
preview extraction and geometry assumptions. Shared rendering, calibration,
graphs, GUI coordinates and recordings also refer to fixed native dimensions.
Simply changing VID/PID or `SENSOR_WIDTH` is not new-camera support.

## Choose the smallest honest support boundary

Prove raw capture and decoding before exposing controls. If the device only offers
a visual stream, implement preview-only support and mark radiometry unavailable;
never convert palette colors into supposedly accurate measurements. If it offers
radiometry but no preview, software colors can supply the display. Unknown commands
remain unavailable rather than guessed checkboxes.

Inventory direct imports of Duo constants/functions before designing a provider:

```bash
rg -n 'TC002CDuoCamera|decode_duo_frame|raw_temperatures|measurement_frame_status|SENSOR_|HEADER_U16|IMAGE_OFFSET|FRAME_MAGIC|FRAME_BYTES' src tests
```

Introduce an explicit target backend/device profile where necessary, preserving
existing Duo behavior. A provider should own discovery/open/close, negotiated
transport, frame decoding/validity, native dimensions, preview format, conversion
and supported control implementations. Keep raw evidence alongside decoded data.
Expose firmware/layout compatibility and measured capability bounds, not merely
advertised fields. Thread the selected provider through desktop/web, measurements
and pipelines; do not translate a new camera's packets into fake Duo headers.

Do not refactor unrelated GUI/pipeline features to make an initial read-only
backend work. Document the supported model/firmware and unsupported features.
A saved setting or calibration belongs to a device/profile: a different native
resolution, optical geometry or protocol must not inherit incompatible spots,
calibrations or control overrides silently.

## Capture and concurrency

Maintain one device owner and serialize selectors/mailbox request–reply sequences
so commands cannot consume each other's responses. Preserve the continuous capture
thread and latest-frame behavior. Use queued reads only for a proven bulk transport;
add appropriate isochronous/native transport support for a different endpoint.
Cancel/drain transfers before freeing buffers; libusb can deliver callbacks on a
control-transfer thread. A timeout is not permission to free pending transfer RAM.

Perform expensive image processing off the acquisition/UI threads. Graph sampling
and logging consume valid native measurements independently of display filters.
Inactive/unconnected pipeline branches and closed previews must not keep workers
running. Rotation runs after the software pipeline; native spots must map through
mirrors, rotation and source-dependent display scaling to the same sensor pixel.

Use incremental control operations for slow LUT/table uploads, with status,
progress, cancellation and restore. A detail/preset change may require settling
or stream refresh; signal the operation instead of appearing hung. Unknown protocol
layouts fail closed. Apply readiness/handshake state per physical USB device and
invalidate it after replacement/reconnect; allow only a bounded evidenced retry.

Stall investigations must separate missing USB transfers, frame rejection,
invalid/frozen radiometry, control blocking, processing latency and GUI repaint.
Record stage/counter/queue observations before blaming focus, Git activity or
monitor sleep. The Duo queue-depth/defaults are not validated for a new camera.

## Measurement integrity and control ownership

Keep the native measurement plane distinct from uint8 display normalization,
palettes, interpolation, sharpening, AI and contour fills. Upscaled pixels must
not generate new measurement spots or improve claimed temperature resolution.
Firmware range/gain switching and host fixed display bounds are different controls.
A percentile-normalized raw display can auto-adjust entirely in software.

Use validated conversion/corrections for the specific target. Keep wire/storage
units distinct from Celsius/Fahrenheit and centimeters/inches presentation. Honor
sensor invalid/saturation codes and native shutter/FFC/freeze flags; do not copy the
Duo header offset 32 or 0/65535 rule to a different encoding without evidence.
Hold the last valid display if appropriate, but do not present repeated held
measurements as new CSV samples. Preserve gaps/validity semantics for analysis.

Advertise hardware effects accurately: preview-only controls are inactive when
no active branch/input/mask uses preview; measurement corrections work on verified
radiometry in either source. Source/pipeline edits must not reset calibrations.
Logging/calibration locks, wheel-input suppression and saved settings remain
consistent with existing application behavior.

Save original device state before overrides. Restore only owned supported fields,
preserving unowned bytes and user calibration data. Bypass/removal/exit and failed
operations need audited cleanup. Scene/bank/preset changes can rebuild a tone LUT;
preserve other curve stages and reapply an owned gamma/boost after the rebuild.
Do not overwrite actual UI inputs with quantized wire readback if it misrepresents
the user's selected value; keep internal and presented values separately.

## Validation before declaring support

Add fixtures with known provenance under the requested source tree, and mock USB
for parsing/serializer/status tests. Test meaningful behavior: short/partial/error
payloads, recovery at frame boundaries, endian/stride/plane separation, numerical
conversion, saturation/freeze, startup/reconnect/layout rejection, response-queue
ownership and cancellation. Control tests need payload preservation, fresh
readback, failure cleanup and unsupported-feature behavior.

Run the affected acquisition/control/measurement/persistence/pipeline tests and
required lint. Existing commands use `.venv/bin/pytest` and `.venv/bin/ruff`; inspect
`pyproject.toml` before assuming tooling. Expand to the full suite before a broad
camera-provider change is declared complete. Tests may require offline Qt setup;
reuse `tests/test_capture_panel.py::popup_environment` rather than inheriting
OpenCV's incompatible Qt plugin environment.

A bounded live check then verifies fresh connection, continuous frames,
measurement validity, current control capabilities and restoration. Where safe
and authorized, cover cold-start/replug, open–close–open and settings reload;
a successful same-session toggle is not power-cycle support. Compare at least one
normal frame and one documented calibration event against saved evidence. Report
actual precision/compatibility limits; do not equate transport success with
thermometry calibration or manufacturer certification.
