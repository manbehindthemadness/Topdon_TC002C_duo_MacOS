# Thermal and visible-light camera systems

Use this reference when a device has an optical camera alongside a thermal sensor,
or a vendor advertises visible-light, RGB, PIP or fused output. A processed thermal
display, grayscale/BGR versions of one display, two color palettes, and a thermal
device branded “Duo” do not establish two independent cameras.

## Establish the actual streams

Record each sensor/stream's role, USB device/interface/endpoint or SDK channel,
pixel format, dimensions, stride, rate, timestamp/sequence domain and validity.
Determine whether the sensors can stream concurrently or whether mode selection
replaces the sole output. A vendor fused image is a separate processed product;
it does not prove availability of its original optical and thermal planes.

Identify planes against both thermal and optical scene changes using saved raw
evidence and producer code. Determine whether shutter/FFC freezes only thermal
data, both displays or a vendor fusion stream. Preserve original bytes and
provenance separately for each stream. Coordinate all interfaces under one
backend owner; independent USB handles do not authorize competing reply consumers.

## Current program boundary

The application has **experimental dual-camera support, untested on physical
hardware**, and an offline `virtual-dual` backend. `CameraFrame` retains native
`raw_counts` and thermal `preview`/`preview_bgr`; `visible` is a separate immutable
`VisibleFrame`. It carries optical BGR pixels, independent timestamps/sequences,
status and optional homography/provenance. `CameraProfile.visible` advertises a
second camera. Keep both clocks in the same monotonic domain and declare the
maximum tolerated age. Source, Combine input and masks accept `visible` alongside
thermal `raw` and `preview`; **Thermal upsample** retains the `preview` key.

Never store optical pixels in `preview_bgr` alongside thermal `preview`. Thermal
FFC holds native measurements and thermal imagery, while current optical pixels
continue. Missing/stale optical video fails explicitly without invalidating thermal
sampling. Unregistered optical images permit display-only processing; spatial
measurements/placement require fresh known timing and full native-grid coverage
from an explicit optical-to-thermal transform. Capture JSON records stream timing,
status and registration provenance. Synthetic registration is not evidence for a lens.

Version-4 documents optionally store `visible_hardware` separately from thermal
`hardware`. The editor shows this stack only for a second camera. Implement
`create_visible_controls()` only for audited settings, returning a distinct
`DeviceControls` owner with the matching profile. Each camera has its own baseline,
capabilities and pipeline owner; same-name controls remain independent. Optical
settings activate through connected optical source/input/mask dependencies.
Original reads, rollback and restoration must address the correct camera/channel.

See [camera-backends.md](../../../../docs/camera-backends.md) for the current public
contracts and offline launch command. The virtual backend exercises unequal sizes
and rates, faults and synthetic control isolation; it does not prove a physical
transport, synchronization or lens geometry.

## Integration requirements once streams are verified

- Keep thermal counts, thermal display and optical imagery separately identified.
  Describe per-stream geometry, availability, acquisition time/sequence and
  stale/missing state. Preserve old frame constructors and saved `preview` keys.
- Add optical source/input/mask choices through document validation, dependencies,
  UI capabilities and processing. Only advertise demonstrated streams; a missing
  optical stream must not silently become thermal output. Preview-only thermal
  devices remain supported without implying optical capability.
- Define bounded synchronization and newest-frame handling for different rates
  and timestamp domains. Optical loss must not stop verified thermal sampling;
  invalid thermal data must preserve graph/CSV gaps even if optical video is fresh.
  Apply independent freshness policy to vendor fused images.
- Establish optical-to-thermal geometry before overlaying temperature labels or
  sampling through optical coordinates. A resize or matching dimensions only
  aligns array sizes. Document lens distortion, field of view, crop, rotation,
  parallax and calibration scope. Allow display-only optical processing when
  registration is unknown; do not attach misleading measurement labels.
- Audit hardware control ownership per stream, including source/mask dependency
  rules, original settings, rollback and reconnect. Scope registration and optics
  calibration to the actual device/profile. Preserve thermal conversion and
  native measurement resolution through blends and optional fusion.
- Include stream identities, timing, validity and registration provenance in
  capture/export metadata where those streams are used. Preserve worker-local
  ownership, model cleanup and JSON-only Qt communication.

Before declaring dual-camera support, test unequal resolutions/rates, out-of-order
or partial stream data, missing/stale optical frames, thermal FFC/invalid gaps,
source/input/mask selection, branch scheduling, locks, reconnect and recorded
metadata. Verify registered coordinates with independent fixture landmarks across
rotation/mirroring, then run bounded authorized checks on the actual firmware.
