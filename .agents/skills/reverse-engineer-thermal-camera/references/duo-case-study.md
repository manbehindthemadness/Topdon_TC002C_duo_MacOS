# TC002C Duo: useful precedent, never a new-camera protocol

These examples come from this repository's investigated Duo and matching vendor
artifacts. They are scoped to that device/firmware and to the recorded experiments.
Do not send these requests to another model simply because it is thermal, USB/UVC,
TOPDON-branded, or manufactured by a related module supplier.

Read current `src/topdon_duo/` for product behavior. Consult
[the discovery index](../../../../docs/hardware-discoveries.txt) and
[the chronological research log](../../../../docs/hardware-research.txt)
for packet examples, source artifacts and later corrections. Do not copy the old
Analyze-mode/toolbar UI descriptions into today's pipeline builder.

## Identity and data-plane evidence

The investigated device reported VID/PID `2bdf:0102`, a TM32 module identity,
`APP_209031_BUILD_20250917`, and `FPGA_010108_BUILD_20250414`. Similar HIKMICRO SDK
structures were useful leads, but do not establish an exact module datasheet.
The running firmware was not proven byte-identical to every packaged firmware.

The normal native grid is 256×192. The repository supports an assembled frame
with 2,320 little-endian uint16 telemetry words (4,640 bytes), followed by
49,152 uint16 temperature words and a native or larger preview. Magic at byte 0
is `0x70827773`. The native container is 201,248 bytes; a 512×384 YUY2 preview
variant is 496,160 bytes. Other research stream modes exist but are not universally
implemented. Negotiated descriptors and actual frame lengths govern decoding.

Validated native apparent temperature uses `raw / 64 - 50` °C in camera-conversion
mode. The renderer also retains an older ambient/percentile anchor fallback;
that software anchor is not calibration proof. Camera ambient, distance,
emissivity, reflected-temperature and optical-transmission settings affect native
measurements. Unit selection did not convert the captured plane out of Celsius.

The native preview is processed independently. SDK brightness/contrast/detail/noise,
presets, palette and tone curves affect it; interpolation, palettes and AI can
also be host processing. Raw display auto-scaling came from host percentile mapping.
The absence of preview AGC effects in a raw pipeline did not rule out camera AGC.
Phone TISR was traced in the official Android path to host Anime4K09; the generic
camera AI flag was ignored on the investigated firmware. The iOS path was not traced.

A DWORD at assembled-frame byte 32 marked frozen radiometry during shutter/FFC,
with repeated temperature planes. Production holds valid readings and skips
invalid sample points. This is a scoped status finding, not a generic telemetry
schema. A visible click, image jitter or changing header statistic alone is not
reliable invalid-pixel detection.

## Configuration protocol and cold-start trap

The audited SDK sequence uses extension unit 10/interface 0 (`wIndex=0x0A00`).
Select a configuration with `0x21 / SET_CUR(1) / wValue=0x0500` and two bytes
`[selector, command]`; obtain length via `0xA1 / GET_LEN(0x85)` with
`wValue=selector << 8`; read via `0xA1 / GET_CUR(0x81)` or write the complete
preserved payload through `0x21 / SET_CUR(1)`. GET_LEN requested four bytes but
returned two on this camera. Bound lengths and accept only exact known layouts.
Do not insert a GET_CUR between selection and the actual SET_CUR write.

Cold startup could return a legacy 512-byte layout for a 2-byte brightness block.
A validated selector-4 version GET_CUR returning `2.0\0` also initialized volatile
SDK-2.0 dispatch, after which block sizes became 2/2/79/80. Production caches this
only after success and per device, with one bounded handshake/reselection retry
for the evidenced legacy-layout case. Padding/truncating the unexpected block
would have hidden the initialization problem and risked invalid writes.

Important configuration blocks:

| Selector / command | Wire layout on this camera |
| --- | --- |
| 2 / 1, 2 / 2 | 2-byte brightness/contrast; value at byte 1, 0–100 |
| 2 / 5 | 79-byte enhancement block; palette byte 5, detail enable/amount 6/7, noise mode/levels 1–4 |
| 3 / 1 | 80-byte thermometry block; offsets below |

The initial research also captured stream/adjustment blocks. Read the actual
current code and per-run originals; this table is not a full snapshot catalog.
Thermometry fields are little-endian uint32: emissivity percent at 16, distance
centimeters at 21, reflected temperature `(C+100)*10` at 26 with enable byte 25,
transmission percent at 40, humidity `percent*10` at 69, ambient `(C+100)*100` at
76 with enable byte 75. Apparent floats in the UI can be scaled unsigned integers
on the wire. Preserve every unknown field and companion enable byte.
The center-overlay field accepted changes without a useful demonstrated image
effect and its user control was subsequently removed.

A command-state success response did not prove application: ignored AI/settings
still returned success. Fresh readback could lag; unsupported getters could return
stale selected data. Verify exact state and actual frame effects separately.

## USB mailbox and firmware audit lessons

Static inspection of a vendor APK firmware bundle recovered RV32 RISC-V CPU code
and strings. Follow serializers, dispatchers, globals and side-effect calls;
related Mini/Duo images had different FPGA projects and refresh behavior.
An identical forced-register quartet did not establish safe cross-device presets.

The audited raw UART mailbox uses `wIndex=0x0A00`, `wValue=0`, request types `0x41`
(write request packet) and `0xC1` (reply length/read via `0x85`/`0x81`). GET_CUR
consumes a queued reply. Legacy framing was `F0 | length | 36 | command | operation
| data | byte-sum checksum | FF`. “Operation 1 = getter” was valid only for individually
traced opcodes. Some command handlers acted regardless of the operation byte.
These descriptions are not an executable generic probe recipe.

Tone boost staged flags after a rebuild and emitted success followed by a
fall-through error. Both replies needed consumption, and applying/restoring twice
was proven on this specific route. Normal SDK gamma uploads did not give the
expected effect; a traced 256-entry composite LUT path did. Production uploads
257 bounded volatile commands incrementally with progress/cancel, and preserves
other curve stages when SDK/preset changes recompose the tone state.

“Fixed range” `f113` changed volatile camera processing but did not become a
validated temperature-bound-controlled AGC freeze. It forced registers and could
remove thermal shading; detail enhancement supplied visible edges. Product Fixed
detail mode therefore requires detail enhancement and audited preset/tone
conditions. Raw threshold units were not established as calibrated degrees.

The ISP export was a 3,956-byte stored preset with seven banks/public parameters,
not a complete active register/LUT/calibration backup. Physical ISP getters could
return zero for everything, which could not be used as original state. Bank-8
metering used missing parameters/sentinels; a read-looking ROI route also mutated
cache/geometry. CPU-loader replay exposed the issue before a live legacy-ROI probe.
Later direct metering trials restored known defaults but on/off ultimately looked
identical. The camera AGC freeze goal remains unverified; do not expose that failed
lead as a supported switch.

## Local evidence and transferable lessons

Large original artifacts are ignored under `diagnostics/telemetry-research/` and
`diagnostics/agc-research/{firmware-offline,deep-audit}/`. A fresh clone may not have
them. If a file is absent, recover an authorized original source or mark the
claim unverified; do not invent bytes from a log description. Vendor download URLs
and artifact names are recorded in the chronological log, not promised permanent.

Reapply these reasoning lessons to a new device, not these numerical constants:

- Exact live protocol/layout readiness matters more than brand or API naming.
- Separate raw measurement changes, camera preview mapping and host display mapping.
- ACK, actual state, effect, complete restore and production support differ.
- NVM presets, live caches, LUTs and physical registers require distinct accounting.
- Check queued multi-reply behavior and call order before labeling a feature ignored.
- Keep native references and invalid-frame evidence; visual improvement is not new
  radiometric or spatial precision.
- Preserve failed probes and side leads; later controlled observations can overturn
  enthusiastic first impressions.
