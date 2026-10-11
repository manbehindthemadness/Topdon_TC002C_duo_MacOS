# Camera backend integration

The viewer defaults to the existing TOPDON TC002C Duo implementation. Other cameras
can be integrated through `src/topdon_duo/camera_backends/`; this foundation does
not provide protocols or verified support for another physical camera.

On macOS, normal-user Duo factories select the private USB helper facade described
in [macOS USB authorization](macos-usb.md). The original Duo USB protocol and queued
capture run in the authorized helper; decoding, controls ownership, processing and
persistence stay in the viewer. Linux, injected factories and other registered
backends retain their direct backend behavior.

Register an application-owned factory with `register_backend(name, factory)` before
invoking the desktop or web entry point, then select it with `--camera name`.
Registration does not execute the factory or discover devices. Unregistered names
fail explicitly. There is no automatic plugin discovery or VID/PID substitution.

## Backend and frame contract

Implement `CameraBackend`: declare a `CameraProfile` before session initialization,
open the device, yield complete `CameraFrame` objects, provide `create_controls()`,
stop acquisition with `stop_stream()`, and close/drain resources with `close()`.
`timeout_ms` bounds acquisition shutdown. Opening must verify that the physical
device, firmware, layout and optics match the declared profile before overrides
are replayed. A backend owns transport, synchronization, decoding, calibration
validity, invalid/saturation flags and the temperature converter.

The profile declares native width/height, maximum frame rate and verified preview
and radiometry availability. `calibration_key` must distinguish incompatible
firmware/layout/optical calibrations; use `device_id` when settings must be isolated
per physical camera. The firmware field is descriptive; it does not automatically
change compatibility. Profile identity changes clear held frames and averaging.
The capture pump rejects a frame whose profile differs from the selected camera.

Frames keep native numeric counts separate from grayscale or BGR preview pixels.
Arrays are copied and made read-only before branch dispatch. One decoded frame is
shared by all processing branches. A converter returns finite Celsius values at
the same native shape; it must interpret the actual target encoding. Display
filters, palettes, resizing, rotation and AI never replace the native count plane.
Source bytes, timestamps, sequence numbers and reported readings can accompany
the decoded frame for evidence. Reported readings also appear in capture metadata.

A preview-only camera supplies preview pixels without counts or a converter.
Its display works even for a completely black frame; native temperatures remain
unavailable, raw-source selection is disabled, and measurement logging is refused.
Camera-reported spot temperatures can still be displayed as separate telemetry.
If a radiometric profile temporarily omits its native plane, hold its last valid
image and readings, mark the frame invalid for logging/calibration and restart the
measurement average on recovery. Before the first valid plane, readings remain
unavailable. A profile change always clears that held state.
Radiometric cameras may omit the preview and use software colors. The existing
Duo byte-frame API, native `raw / 64 - 50` interpretation and web ambient anchoring
are preserved by the Duo bridge; those rules do not apply to another backend.

## Capabilities, corrections and spots

Declare each verified control with `ControlSpec`: title, numeric bounds/step/unit,
boolean or choice options, default, support/reason, effect and scope. The Qt process
uses this metadata without querying hardware. Unsupported saved nodes retain
their parameters and show an unavailable explanation. Backend dependency checks
run before hardware writes, independently of document validation.
Saved values incompatible with the selected control remain in the document and
show a warning. The editor displays a compatible default for explicit replacement;
state updates never silently coerce the saved value. Input precision follows the
declared native step and minimum through display-unit changes.

`effect="preview"`, `scope="device"` fields are available in the generic **Camera
control** pipeline node. Arbitrary backend names and target-specific ranges need
no catalog changes. Existing Duo nodes retain their established schema and rules;
generic backends use Camera control instead of inheriting Duo presets, noise,
detail or tone commands. Measurement corrections remain outside the pipeline.

`effect="measurement"`, `scope="device"` fields appear in standalone camera
controls. This can include distance, relative humidity, ambient temperature,
emissivity or transmission when verified on the target. `scope="spot"` fields
appear under a separate camera spot selector using the backend's declared
`spot_ids`. Each camera spot has its own correction baseline and override; it
does not become a whole-device correction or a viewer sampling spot.

Use Celsius and meters internally for temperature and distance controls with
`unit="°C"` and `unit="m"`; the existing UI handles Fahrenheit and distance
presentation. Other unit labels are presented literally. Preserve the control's
declared step precision. Device readback units and quantization must be translated
by the backend's read/write callbacks before reaching this interface.

`ReportedReading` carries a name, value, unit, validity, optional camera spot ID
and optional native sensor pixel. Ambient temperature, humidity, distance and
spot temperatures can therefore be reported without masquerading as corrections
or replacing native-grid measurements. Temperature telemetry should remain
Celsius and is labelled with its source units. Readings are displayed separately
and included in captures and web status; no telemetry-to-calibration mapping is
assumed.

`DeviceControls` snapshots all supported global and declared spot settings before
the first override. Its callbacks implement target-compatible reads and writes,
raising `CameraError` for transport/device failures. Writes require readback and
attempt rollback on failure. Restore changes only owned fields and retains
ownership for failed restorations. A failed restoration does not prevent independent
device and spot settings from being restored; failed fields retain ownership for retry.
Restore in the viewer clears persisted global and spot overrides. After restoration,
`invalidate_device()` discards baselines
so reconnect/replacement requires fresh reads. A backend must independently
invalidate protocol readiness and reject incompatible replacements. A session
that reconnects must also call `PipelineHardware.invalidate_applied_state()` and
restart measurement averaging before replaying settings against the new baseline.

Optional Duo operations and host calibration tools remain hidden or unavailable
on a generic backend. A new camera's advertised distance or ambient field does
not establish compatibility with the Duo host calibration algorithm.

## Persistence and verification

Duo preferences keep their original location and migration. Other profiles use
`camera-profiles/<hash>.json` beneath the preference directory. The hash covers
backend ID, dimensions, calibration key and device ID. Display choices can be
shared; camera overrides, spots, pipelines and calibrations are isolated. Unknown
primitive saved fields can be retained for future support without being applied.
Pipeline storage/import/export checks structure; activation checks the selected
backend's dependencies. Duo Fixed Detail restrictions remain enforced on Duo.

Before declaring a new backend supported, use the repository's camera research
skill to establish protocol evidence and fixture provenance. Test native encoding,
different geometry, preview-only frames, invalid-frame hold/gaps, capability bounds,
device and spot ownership, failed writes/restores, reconnect compatibility,
settings isolation and offscreen UI locks. Then perform authorized live capture
and restoration checks against the actual supported firmware. Existing Duo USB
diagnostics and telemetry tools remain Duo-specific.
