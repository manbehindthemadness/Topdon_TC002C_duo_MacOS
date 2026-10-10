---
name: reverse-engineer-thermal-camera
description: Reverse engineer a USB thermal camera's capture, radiometry, telemetry, and reversible controls, then integrate verified capabilities into this thermal-viewer program. Use for a new camera, undocumented camera protocol, or vendor SDK/firmware investigation; ordinary software filters and pipeline UI edits do not need this skill.
---

# Reverse engineer a thermal camera

Discover enough of the target camera's transport, image format, measurement
encoding and controls to use it in this program. Produce reproducible evidence,
a capability map, a restored device and tested integration. A useful read-only
capture backend is a valid first milestone; lack of safe controls must not block it.

## Start with the actual device and current repository

- Check `/usr/src` and reuse this source tree. Keep clones, SDKs, build trees and
  research artifacts under `/usr/src/<project-name>`; use `diagnostics/` for local
  captures/binaries. Standard XDG runtime settings may stay in their usual place.
- Inspect `git status`, repository instructions, existing research and current
  code. Protect unrelated edits, saved captures/pipelines and user files. No
  commit, push, public upload or settings reset is implied by research approval.
- Establish the connected model, VID/PID, firmware, OS/backend, current stream
  and whether a viewer or phone app owns it. Record uncertainty instead of filling
  gaps from a sibling camera. Ask only for missing facts or physical actions needed
  for the next experiment; continue independent offline work while waiting.
- Read [program-integration.md](references/program-integration.md) to locate the
  capture, control, radiometry and pipeline boundaries. This program currently
  has explicit backend/frame/capability contracts. The Duo transport and host
  calibration remain target-specific; a new camera still requires its own verified backend.

## Keep the camera recoverable

The project's hardware research constraint is **no brick-risk experiments**.
This skill does not authorize firmware upload, factory reset, flash/NVM writes,
bootloader commands, calibration-coefficient/bad-pixel-table writes, burn-protection
changes, or unbounded opcode/register fuzzing. Inspect those paths offline only.

Before a reversible live setting trial, establish its exact target-compatible
packet/layout, affected state, bounds and rollback. Save original full settings
blocks and any additional volatile state the operation changes. Defaults from an
app, another device or firmware are not the original state. An ACK, zero-valued
register getter or stored ISP export alone is not a sufficient rollback baseline.
Trace suspicious getters: some read-looking commands alter caches or registers.

Use one owner for USB streaming and consumptive reply queues. Do not launch an
uncoordinated probe against a running viewer. Preserve continuous USB drainage
while applying controls; a long LUT upload needs incremental work, progress and
cancellation. Stop setting trials after an unsupported layout, unreadable original,
unexpected side effect, failed restore or unexplained stream loss. Continue offline
analysis; do not escalate into resets, magic writes or repeated speculative retries.

These rules constrain live trials, not ordinary read-only analysis or code edits.
Use existing user authorization for evidenced reversible steps; do not invent
approval checkpoints. If a physical action or new authorization is actually needed,
state the concrete reason and the evidence missing.

## Progress through evidence, not command names

1. **Identify and capture.** Read [research-workflow.md](references/research-workflow.md)
   for descriptors, vendor sources, frame reassembly, telemetry and radiometry.
   Save original bytes before interpretations or enhancements.
2. **Audit controls offline.** Follow vendor app/SDK call sites into serializers and
   matching firmware handlers. Establish getter effects, layouts, response queues,
   range checks and restoration before using a candidate on the camera.
3. **Run bounded trials.** Read [experiments-and-evidence.md](references/experiments-and-evidence.md)
   before live changes. Use baseline → one change → restore, matched scenes, fresh
   readback and measurements of both preview and radiometric planes. Record ACK,
   application, effect and restoration as separate results.
4. **Integrate only the demonstrated capabilities.** Follow
   [program-integration.md](references/program-integration.md). Keep unsupported
   settings unavailable, measurements independent of display processing, and the
   existing Duo backend working. Test byte-level behavior with recorded fixtures
   and mocked USB before bounded hardware checks.
5. **Leave a resumable handoff.** Record successes, failed probes and unrelated
   leads with exact evidence paths and device/firmware scope. State what was
   restored, what remains unknown, and the next safe experiment.

Read [duo-case-study.md](references/duo-case-study.md) when a finding resembles
our existing camera, or to avoid repeating its failed leads. Its constants and
packets are **examples for the investigated Duo**, never defaults for a new camera.
For original evidence, search [hardware-discoveries.txt](../../../docs/hardware-discoveries.txt)
and [hardware-research.txt](../../../docs/hardware-research.txt). Those files are
chronological: later entries can overturn earlier hypotheses, and current source
code governs current UI behavior. Missing gitignored artifacts are missing evidence.

## Offline helper

`scripts/evidence_manifest.py` snapshots hashes/sizes of already-saved evidence
files and compares two manifests. It never opens a camera, issues USB requests or
restores settings. A match proves equality of recorded files, not device state,
liveness or a complete calibration backup. Run it on copied, quiescent evidence
with the same relative filenames before and after a trial:

```bash
python3 .agents/skills/reverse-engineer-thermal-camera/scripts/evidence_manifest.py snapshot \
  diagnostics/new-camera/run-001/original --output diagnostics/new-camera/run-001/original-manifest.json
python3 .agents/skills/reverse-engineer-thermal-camera/scripts/evidence_manifest.py snapshot \
  diagnostics/new-camera/run-001/restored --output diagnostics/new-camera/run-001/restored-manifest.json
python3 .agents/skills/reverse-engineer-thermal-camera/scripts/evidence_manifest.py compare \
  diagnostics/new-camera/run-001/original-manifest.json diagnostics/new-camera/run-001/restored-manifest.json
```

Create the evidence directories during the actual investigation. The helper
refuses empty snapshots, symlinks, changing files and overwriting an output.
Comparison exits 0 for matching files, 1 for differences, and 2 for invalid input.
