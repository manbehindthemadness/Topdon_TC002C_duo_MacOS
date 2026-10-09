# Pipeline exports

These individual pipeline files can be imported from Camera settings. The same
settings are included in `src/topdon_duo/predefined_pipeline_presets.json`, which
ships with the package and populates the preset dropdown on a fresh installation.

The following exports capture the saved macOS pipelines on 2026-10-09, preserving
node identities, expansion and bypass states, hardware controls, all four branches,
and model/backend preferences:

- `Yautja GPU.pipeline.json`
- `Reaper Night GPU.pipeline.json`
- `Reaper Day GPU.pipeline.json`
- `Detail Enhanced CPU Upscale.pipeline.json`

The active pipeline matched the saved Yautja GPU at capture time. The older
`Redneck Combat.pipeline.json` remains the original reference export. Tests compare
the bundled presets with these captured exports. Local settings files and unrelated
viewer preferences are not part of the exports.
