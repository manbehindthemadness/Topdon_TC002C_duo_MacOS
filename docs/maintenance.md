# Maintenance and first-install testing

[Documentation home](../README.md) · [Imaging](imaging-guide.md)

Show the graph pane (`G`), then click **Maintenance** at the far right of its
controls. In portrait layout, the button moves with the graph toolbar below the
image. On smaller windows, use the toolbar overflow menu.

Maintenance lets you select cleanup items or use **Factory reset and close**.
Each action shows a confirmation with its scope and locations; **Cancel** is the
default. Close other viewer instances first. Recording, CSV logging, pending Save
dialogs, calibration and camera tone updates prevent cleanup.

The viewer closes before deleting files. This releases inference sessions,
measurement/graph caches, imported custom-node instances and frame buffers, and
prevents open dialogs from saving their old settings back into the cleared store.
Active model downloads are cancelled and joined; converter processes are stopped.
If an installer or inference worker cannot stop, deletion is refused. The terminal
reports completed cleanup or failures; partial filesystem failures are reported
explicitly. Restart the viewer yourself afterward.

## Cleanup choices

| Item | Removed | Retained |
| ---- | ------- | -------- |
| Downloaded model cache | Recognized catalog/custom downloads and checksum files | Bundled models, registered original local models, converted exports for this targeted action, unknown files and linked files |
| Inference runtime caches | App-generated runtime generations, setup state and failure/retry markers | The installed Python environment and system CUDA/cuDNN/driver installations |
| Settings | Viewer/camera settings, saved presets and the working pipeline, measurement configuration, calibration preferences, preset association and window/file-dialog settings | Original custom-node folders, backend source files, shipped presets and unrelated files/folders |
| Session artifacts | Support/debug logs, camera baseline snapshots, `:memory:.ses` and its journal files, project test/lint caches | Camera firmware, backend source and IDE configuration |
| Python bytecode | Application/check-out `.pyc`, `.pyo` and `__pycache__` artifacts | Python source, custom-node directory structure, `.agents`, `.idea`, `standards` and virtual-environment dependencies |
| Generated outputs | Default-named captures/recordings/CSV, diagnostics traces and `.pipeline.json` exports in the listed output locations | Unrelated files, source folders and linked files |

**Warning:** Clearing settings permanently deletes saved custom pipeline entries,
embedded custom-node configurations, measurements and calibration preferences.
It does not delete the original node packages or user-added camera backends. Export
important working pipelines before clearing settings; keep backups outside the
output locations selected for cleanup. Settings cleanup does not modify camera
firmware or remove installed source code.

## Factory reset

**Factory reset and close** selects every category above. It additionally removes
recognized converted model exports/provenance and the local-model registration
index, while protecting the registered original files. Saved custom pipelines and
their embedded node entries are lost. Generated outputs in the listed locations
are also deleted after an explicit warning.

Paths are derived by the parent application, including configured model/runtime
locations, the current checkout/working directory and remembered Save/Export
folders. Output cleanup recognizes the application's default timestamp filenames,
trace files in `diagnostics`, and `.pipeline.json` exports. It does not search the
whole computer: renamed captures, unknown files and historical exports elsewhere
are preserved. Directory symlinks are not traversed during cleanup. No virtual
environment, source package, custom-node folder, added camera backend, IDE
configuration or bundled preset/model is removed.

On the next launch, absent preferences produce the fresh-install defaults: the
Duo starts with **Detail Enhanced CPU Upscale**; other cameras use their own
profile defaults. Runtime provisioning and on-demand model installation run again
when needed. In-memory session state is gone because the viewer process has exited.
This recreates application first-install conditions; it does not reinstall Python,
reset the GPU driver, empty global uv/pip caches or change camera firmware.

## Support exports and debug logs

Use **Debug logs** to enable detailed application logging plus USB stream,
viewer timing, heartbeat and stall/thread-stack diagnostics. The checkbox applies
immediately and persists for subsequent launches. Disable it after reproducing
the problem; existing logs remain available to export. An explicit command-line
`--diagnostics` path remains active when the checkbox is off, and `--verbose`
continues to request verbose application logging.

Ordinary application messages/errors are retained even with Debug logs off.
File logging runs through a background queue into
`$XDG_STATE_HOME/based-thermals/debug-logs` (default
`~/.local/state/based-thermals/debug-logs`, also used on macOS). The application log
rotates at 4 MB with three backups; opt-in diagnostic traces/stacks stay until
session-artifact cleanup or factory reset. Native library messages printed only
to terminal stderr are not captured by Python's application logger; attach a
terminal log separately when those messages matter.

**Export settings…** creates a ZIP containing saved viewer/camera preferences,
custom pipeline presets, window/dialog preferences, maintenance preferences and
the local-model registration index. `current-settings.json` separately captures
the active pipeline, measurements and current controls, including edits not yet
saved on exit. Saved originals are copied without parsing/replacing them, so
malformed JSON is preserved for investigation.

**Export all…** adds available application logs, diagnostics traces/stacks and
rejected-frame debug samples, temperature CSV logs in known output/remembered
log locations, camera baseline/session artifacts, runtime setup state, and
platform/Python/package versions with the current camera and acceleration
capability report. It includes an explicit `--diagnostics` log location. Active
logs are snapshots and may acquire more records afterward. The archive does not
scan the whole machine or collect environment variables, model weights, runtime
binaries, captures/videos or external custom-node/backend source folders.
Settings may themselves contain embedded custom-node code and local file paths;
review the archive before sharing it.

Both export actions use a Save dialog and run compression in a background worker
without closing the viewer. Cleanup and further exports are disabled until the
archive finishes. A failed archive write preserves an existing destination.
`manifest.json` lists copied file locations and any skipped linked files or
partial read errors; the popup reports those warnings. Exports are local files;
users can send the ZIP themselves. Session-artifact cleanup removes retained
support logs, while settings cleanup also resets the Debug logs preference.
