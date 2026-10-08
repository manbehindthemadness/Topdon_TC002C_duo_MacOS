# Controlled live trials and resumable evidence

## Evidence states

Keep these conclusions separate for every candidate:

| State | Minimum useful evidence |
| --- | --- |
| Lead | Documentation, string, symbol, similar-device path or candidate header field |
| Offline mapped | Exact target handler/serialization traced; modeled writes and gates recorded |
| Transport accepted | Packet/reply framing valid and status recorded |
| State applied | Fresh current-state readback or validated independent state evidence |
| Effect demonstrated | Repeatable matched-scene change in the intended raw/preview output |
| Restored | Fresh verification of all affected supported state plus healthy live capture |
| Production supported | Compatibility gates, lifecycle/rollback and regression tests pass |

These are distinct properties, not an automatic linear promotion. An operation
may have a visible effect without trustworthy readback, or return an ACK while
ignoring the setting. Mark “ignored,” “unproven,” “unsupported” or “inconclusive”
accurately. Record failed leads and unrelated discoveries so they are not retried
as undocumented successes in the next session.

## Before changing a live setting

Use the authorization and scope already established. The no-brick constraint
rules out speculative persistent actions; it does not require asking permission
for every documented reversible adjustment.

Establish identity/firmware scope, current readiness, one USB owner, healthy
capture and a bounded trial window. For an undocumented route, first trace
initialization, validation, writes, response behavior and restore offline.
Read and save the complete original supported configuration blocks, not just
named sliders. Copy them into immutable per-run evidence files. Include selected
banks, feature-enable flags and tone/cache state where the setter affects them.

Audit the restore against the actual starting configuration, including existing
user gamma, preset, boost or calibration overrides. Restoring factory defaults is
not restoring the original. A file-exported stored preset cannot prove active
registers, RAM caches, LUTs or calibration NVM were saved. If an original cannot
be read, require independently established exact volatile state and a proven
restoration procedure; otherwise do not send the setter. A physical getter that
always returns zero cannot establish a zero baseline.

Prepare the allowed mutation and cleanup code before running. Define maximum
commands/duration, expected responses/layouts, valid value bounds and criteria
for stopping. Prefer an existing audited setter with a small valid change. No
opcode/register fuzzing, unknown-bank sweeps or guessed “unlock” operation.

## Trial execution

Use baseline → candidate → restored baseline, repeating only if the first run
restores cleanly and resolves a meaningful uncertainty. Keep the camera mounted,
scene, gain/range, palette and other controls fixed. Log the actual control state
and timestamps; a stale GUI mode label is not packet evidence.

For brightness/contrast stability, deliberately bring a warm/cold object into one
edge while measuring a stationary background patch. Compare automatic and
candidate states under matched perturbations, and avoid clipping/saturation.
A static scene alone cannot prove AGC freeze. Track fixed patch mean/luminance
range and corresponding native pixel temperatures; unchanged temperatures with
changing colors can indicate display mapping rather than radiometric changes.

For corrections, measure stationary targets and allow settling. Keep scene drift,
shutter events and invalid frames out of claimed differences. Compare candidate
with both neighboring baseline phases. Preserve native references when evaluating
SR/sharpening: improved apparent edges or model-generated texture are not measured
sensor resolution or accurate new detail.

Capture protocol packets/replies and frame bytes with a common monotonic time
basis. Label stages only after the actual command result, record timeout/latency
and retain invalid samples as evidence. Do not silently heal invalid measurements
in the authoritative log; interpolation for later analysis needs explicit labels.

Always perform cleanup through `try/finally` or equivalent, including normal
completion, cancellation, a failed command and capture exceptions. Verify fresh
readbacks where valid; preserve unowned bytes and restore all affected volatile
state in audited order. Reopen/renegotiate a stream only through a supported path.
Verify liveness via counters/timestamps and new scene response, not just changing
telemetry or one plausible frame. Frozen planes can still look correct.

If rollback fails, stop further mutations, save the failure and identify the
remaining state uncertainty. Do not substitute a factory reset or power cycle as
unproven recovery. If an already-established safe recovery requires a physical
replug, ask for that concrete action and verify again afterward.

## Artifact layout and journal

Use `diagnostics/<device-slug>/<UTC-run-id>/` under the existing `/usr/src` tree.
A practical run records identity, provenance/checksums, raw descriptors, packet
logs, original/restored binary blocks, original/restored manifests, frames,
statistics and a journal. Avoid committing serials, scenes and vendor binaries
unless specifically part of the requested deliverable and appropriately licensed.

For every trial, record:

- Question and target plane; exact device/firmware and host code revision.
- Source URL/artifact hash, relevant handler/symbol/address and confidence.
- Original full payload path, selected/cached state and compatibility checks.
- Request bytes, replies/checksums, command count, value and bounded duration.
- Scene/test actions and actual phase timestamps; interruptions or confounds.
- Raw/converted/preview metrics and observed validity or freeze indicators.
- Restore bytes/actions, readback coverage, state not covered and final liveness.
- Conclusion, failed hypothesis/side discoveries and next safe discriminating test.

At a session boundary, write a short handoff with the last completed phase, exact
current device state, owner/PID if relevant, remaining restore work, artifacts and
what physical action is pending. Another session must not continue halfway through
a volatile setter using an assumed baseline. Never erase a contradicted conclusion;
append the correction with evidence and update the discovery index.

The bundled manifest helper compares saved files only. Hash equality supports
recorded-block restoration; combine it with actual device readback and liveness
checks. It cannot discover or back up hidden calibration state.
