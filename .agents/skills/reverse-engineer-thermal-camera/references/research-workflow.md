# Discovery and offline protocol audit

## Identify without assuming a protocol

Save the OS, USB backend/version, VID/PID, interface/alternate settings, endpoint
directions/types/packet sizes, manufacturer/model/serial strings and advertised
UVC formats. On Linux, `lsusb`, descriptor dumps, kernel logs and `v4l2-ctl`
can help; on macOS use `system_profiler SPUSBDataType`/IORegistry and available
libusb tooling. Use whichever is present; a native SDK-only/non-UVC device needs
its own transport investigation. Lack of a USB analyzer does not end the search.

Do not detach another application's driver or reset a shared USB bus merely to
collect identity. Close/coordinate the owning application when capture must claim
an interface. Record disconnects and re-enumerated identity, including after a
cold power cycle. A camera can click and still lack a ready control layout.

Create a per-device research folder with a UTC run identifier, identity JSON,
unaltered descriptors, original settings where readable, raw frames and a journal.
Store serials and scene captures in local ignored evidence; publish only material
within the user's requested scope. A hash identifies a firmware/app artifact;
a marketing model name alone does not.

## Find primary sources and the actual vendor path

Prefer official product/manual/download pages, SDK headers and vendor sample
code. Use public or user-provided APKs/firmware bundles for static inspection;
retain download URL, date/version, checksum and license/source provenance.
Community implementations are leads to validate against primary artifacts.
Do not guess that the branded camera and a similarly named module share layouts.

Inspect archive members before extracting to a separate directory under `/usr/src`.
Do not run the vendor app, install firmware or execute unknown native binaries
as part of static inspection. Trace DEX/JNI calls and native symbols/strings to
packet serializers. SDK structures include padding, host pointers and versioned
fields: wire offsets must come from the serializer or saved packets, not `sizeof`.

Useful approaches: `rg`, `strings`, archive listings, DEX decompilers, `readelf`,
`nm`, architecture-appropriate `objdump`, or available disassembly tools. Establish
endianness/ISA, image section offsets, load base and globals from evidence. Build
an analysis-only image if needed; preserve the original and record relocations.
The Duo firmware was RV32 with mapped sections; a new camera may be unrelated.

Search themes rather than one expected spelling: thermometry, radiometric/raw,
AGC/histogram/gamma/GBCE, boost/detail, shutter/FFC/NUC, emissivity, reflection,
transmission, region/metering, frame/counter/frozen, capability/version and SR/AI.
Trace call sites to determine whether a phone setting changes a hardware command,
a host processing stage, or both. An advertised SR feature can be host-side.

## Reassemble complete frames

Start with descriptor negotiation and a bounded capture using the documented
stream path. Save negotiated format/index/interval and all alternate settings.
An odd advertised image dimension can be a transport container, not sensor size.
Determine whether the endpoint is bulk or isochronous; do not reuse a bulk reader
for an isochronous protocol without implementing the required transfer handling.

For UVC, separate each payload header from data, honor FID/EOF/error indications,
reject partial/oversize frames and resynchronize at actual boundaries. A timeout
may carry usable bytes; assess backend behavior. A completed USB transfer is not
necessarily a complete image. Track received bytes, completed/rejected frames and
rejection reasons. Preserve bounded rejected-frame examples for offline decoding.

Save complete frame bytes at the acquisition boundary, before palette mapping,
8-bit conversion, smoothing or upsampling. Determine total length clusters,
magic/version fields, plane offsets/strides, padding, endianness and channel format
with several frames. Verify each candidate plane against controlled warm/cold
objects and device orientation. Distinguish 8/16-bit raw counts, calibrated fixed
point/float temperatures, grayscale, YUY2 and other preview formats.

Do not scale all words as temperatures or treat color-preview luminance as
radiometry. Packed chroma is not a second temperature byte. Keep the native
sensor grid and the manufacturer preview dimensions separately identified.

## Validate radiometry and telemetry

Test candidate integer/float encodings for plausibility, monotonicity, scale,
resolution, saturation and response to known targets. Validate offset/gain against
SDK conversions and reference measurements at more than one temperature. Scene
percentiles or ambient anchoring can make an image plausible without validating
absolute temperature. If calibrated encoding is unknown, label the plane as raw
signal and do not invent accurate Celsius readings.

Change one documented correction (ambient, distance, emissivity, reflected
temperature, transmission) only after the reversible-trial prerequisites. Compare
raw counts, converted temperatures, header min/mean/max and preview separately.
A unit enum can change an overlay while raw data remains Celsius. Validate actual
units; keep host unit display separate from wire/storage units.

Compare telemetry at aligned and unaligned offsets across normal capture, scene
changes, documented corrections and a shutter event. Record candidates for
sequence/time, dimensions, validity/FFC, gain/range, statistics, corrections and
CRC. Check candidates against producer code and repeated controlled observations.
Constant fields can be defaults; changing fields can be timestamps or rolling
buffers. Coincidence with an event does not establish a reliable status flag.

## Trace the complete control operation

For each candidate, map:

- Request type, request/value/index, interface/unit, selector/opcode, endian fields,
  payload/reply lengths, checksum and framing.
- Selection/handshake/read/write sequence, asynchronous status and settling times.
- Handler validation and all callees; every RAM/register/cache/LUT/bank touched.
- Whether a getter really reads current state, consumes a reply, returns a constant
  ACK, or changes state. Determine whether more than one reply must be drained.
- Capability ranges, firmware gates, persistent versus volatile effects and the
  exact restoration sequence, including refresh/latch/recomposition operations.

A function named Get can dispatch a write, and an operation byte of 1 is not
universally read-only. Trace handlers before sending undocumented queries. If
readback is stale or unsupported, document that limitation rather than accepting
previous selected data as a new response.

A short settings block after initialization can become a large legacy block on
cold boot. Identify a target-supported protocol/readiness handshake and verify
its side effects. Retry only known transient conditions with an explicit bound;
never pad/truncate an unknown layout to force an old serializer to work.

Use bounded instruction replay/emulation where practical to verify packet
parsing, range checks, gates, side effects and rollback for the exact firmware.
Replay normal initialization to populate state instead of replacing missing bank
values with guessed zeros. Trap unmodeled MMIO, loops and outside-image memory;
log all writes. A CPU replay proves a modeled path, not live FPGA effects or
firmware equivalence. Absence of a getter can make the setter unsuitable for
safe experimentation even if the handler is understood.
