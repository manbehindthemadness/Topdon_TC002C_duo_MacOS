# Custom node packages

Add **Custom** through the software stack's right-click menu, then expand it and
choose **Open module folder…**. Select the package folder itself. The included
`examples/custom_nodes/local_contrast` folder is a working starting point:

```text
local_contrast/
  __init__.py
  contrast.py
  config.json
```

The package's `__init__.py` must expose `process(image, config)`. Helpers and nested
packages work through relative imports, such as `from .contrast import adjust_contrast`.
Installed dependencies can be imported normally; the node does not install them.

```python
import numpy as np

from .contrast import adjust_contrast


def process(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Adjust the current display image using JSON settings.
    """
    return adjust_contrast(image, float(config.get("contrast", 1.2)))
```

The node receives a copy of the current H×W×3 BGR display image, normally with
float32 intensities in the 0–255 range, plus a fresh configuration dictionary.
Return a numeric NumPy array with three channels, positive width and height,
finite values, and at most 4,000,000 pixels. Output is clipped to 0–255 and converted
to float32. The node runs in stack order and can be repeated or bypassed.
It processes display pixels; it receives no native radiometric or measurement arrays.
An unloaded Custom node passes its input through.

For text that should match measurement labels, use
`topdon_duo.custom_nodes.labels.defer_labels(labels, image.shape[:2])`, where
`labels` is a list of `(text, (x, y))` pixel anchors inside the current image.
It returns `True` when the worker queues the labels, or `False` outside the worker
so standalone code can draw its own text. Queued text is drawn after viewer rotation
and resizing with measurement typography and contrasting outlines. Anchors follow
subsequent mirror nodes and connected image branches; glyphs stay upright and keep
their display size. Preview thumbnails also draw labels after downsampling.
These are display annotations, independent of subsequent pixel filters and blend
colors; an input used only as a mask does not contribute its annotations.

Define `CONFIG_JSON` directly in the root `__init__.py` to bundle settings without
a separate file. It can be a literal JSON string or a literal Python dictionary.
It takes precedence over the optional `config.json` file; otherwise settings
default to `{}`. The editor hides **Import JSON configuration…** when this variable
is present. **Edit JSON configuration…** remains available for advanced edits.
Use the callback's `config` argument for current settings; the original declaration
retains its defaults. Configuration must contain finite JSON values and fit within
64 KiB. Edits reach `process` on its next execution without reimporting the package.

## Embedded defaults and custom controls

Use a plain settings object for JSON-only editing, or a document containing
`defaults` and `controls` to generate node controls:

```python
CONFIG_JSON = r'''
{
  "defaults": {"gain": 1.2, "enabled": true},
  "controls": [
    {"key": "gain", "label": "Gain", "type": "number",
     "min": 0, "max": 3, "step": 0.1},
    {"key": "enabled", "label": "Enabled", "type": "boolean"}
  ]
}
'''
```

The same document format works in `config.json` and the JSON import dialog. Each
control needs a unique `key`, a `label`, a supported `type`, and a matching value
in `defaults`. The processing callback receives the settings dictionary, without
the control definitions. Current values and control definitions travel with
saved/exported pipelines. Reopening the folder reloads its defaults and controls.

| Type | Widget | Additional required fields |
| --- | --- | --- |
| `number` | Decimal number field | `min`, `max`, `step` |
| `integer` | Integer number field | Integer `min`, `max`, `step` |
| `boolean` | Checkbox | None |
| `choice` | Dropdown | `options`: a nonempty list of unique strings |
| `text` | Text field, applied when editing finishes | None |
| `color` | Color dropdown and **Choose…** picker | Value: RGB `#RRGGBB` or `dynamic` |
| `model` | Existing ONNX model dropdown and **Browse…** | Value: local path or `""` for automatic download |

Up to 64 controls are supported. Numeric bounds/values must be finite, within
±1,000,000,000; steps must be at least 0.000000001 and no larger than the range.
Defaults/current values must match their types and ranges. Control definitions
fit within 64 KiB. Generated controls honor pipeline locks, and JSON edits refresh
their displayed values. Controls update configuration; they do not invoke Python
callbacks in the UI process.

Model controls discover ONNX files in the shared cache, bundled models and the
path library created by Browse (`custom-models.json` in the model cache). Browsing
registers/selects an existing file without copying it or importing a model. The
dropdown refreshes on opening and preserves missing saved paths visibly. Model
compatibility is checked by the package during processing. Color controls store
normalized RGB hex values or `dynamic`; the package decides how to render them.
Classless YOLO reuses the measurement contrast compositor for dynamic borders
and labels, and uses inverted pixels for its optional dynamic fill.

`CONFIG_JSON` and `MODEL_SOURCE` must be direct, unique module-level literal
assignments (annotations are allowed). Computed expressions such as `json.loads`
or function calls are rejected for these declarations. The editor reads their AST
without importing/executing the package.

## Additional examples and builder skill

[Classless YOLO](../examples/custom_nodes/Classless%20YOLO/README.md) adapts
Inspector's Mobile Object Localizer stage into an optional Custom package.
It draws confidence-only boxes using external ONNX weights and the viewer's
existing dependencies. Its README explains model setup and the supported export.
Select the `examples/custom_nodes/Classless YOLO` folder to load it; its pinned
model downloads automatically, or you can supply a local model-file override.

## Shared model downloader

Custom packages can use the viewer's background downloader:

```python
from topdon_duo.custom_nodes.models import ModelResolver

models = ModelResolver()

# Inside process(image, config):
path = models.resolve(config["model"])
if path is None:
    return image
# Initialize/cache inference with str(path), then process the frame.
```

Create one resolver per package, then call `resolve` during processing. It returns
a checksum-verified `Path` when ready, or `None` while downloading. Return the
input image during that wait. Downloads share the existing desktop progress/error
display and cache, and matching requests from separate nodes share one job.
Downloads do not run while browsing folders or importing pipeline documents.

The source can be an existing catalog name such as `"espcn"`, or a JSON object:

```json
{
  "name": "My model",
  "url": "https://example.com/model.onnx",
  "sha256": "<64 lowercase hexadecimal characters>"
}
```

Supply the actual pinned SHA-256 before use. Optional `max_bytes` limits both
download and extracted-file size (default 70,000,000; maximum 512,000,000).
For tar.gz sources, also supply `archive_member` (a relative regular-file path)
and `archive_sha256`. Both the archive and extracted model are verified; no
other members are unpacked. Classless YOLO's `MODEL_SOURCE` is a working example.
Custom sources do not change the built-in model catalog.

Embed that catalog name or source dictionary as `MODEL_SOURCE` in `__init__.py`:

```python
MODEL_SOURCE = "espcn"

# Inside process, using a resolver retained in module globals:
path = models.resolve(config.get("model", MODEL_SOURCE))
```

Folder loading copies the source into the initial configuration's `model` key
unless defaults already supply one. Embedded sources hide **Import model source…**.
Without this declaration, an optional `model.json` file supplies the default;
the import button can load a source JSON file into `config["model"]`. Keep folder
selection available to change/reload the package. A source declaration only supplies
metadata; the package must call the resolver during processing to request weights.

Cached weights are reused offline. `TOPDON_MODEL_DIR` overrides the standard
macOS/Linux model-cache directory. Failed downloads raise a pipeline error on
subsequent calls rather than retrying every frame. Bypass/re-enable or reload the
node to create a fresh resolver and retry. Catalog models requiring a local Torch
export retain that requirement; the downloader does not perform those exports.
Record model licenses/provenance and document automatic downloads in your package.

Use the project-local [$create-custom-node skill](../.agents/skills/create-custom-node/SKILL.md)
to build future packages, for example: “Use $create-custom-node to turn this image
filter into a Custom node with JSON controls.” The skill covers the implemented
callback contract, modular package layout, configuration, model provenance and
verification. It is kept in this repository's `.agents/skills/` directory.

Python and JSON files are embedded in saved presets and pipeline exports, so the
original folder is no longer required. Select the folder again to load changes
from disk, including its configuration defaults; files are not watched automatically.
Hidden files and `__pycache__` are excluded, other file types are not bundled,
and symbolic links are refused. Packages need a root `__init__.py` and are limited
to 128 files and a 2 MiB serialized bundle. JSON resources may be read relative
to `__file__`; each loaded package has its own temporary directory.

Package loading in the editor and pipeline document validation check paths,
Python syntax and JSON without running the code. Imports and callbacks execute
in the processing worker. Each node has independent module globals, retained
between frames; changing its code, bypassing/removing it, disconnecting its branch,
or closing the viewer unloads the package and removes its temporary directory.
Reported Python errors and invalid outputs use the existing pipeline-error behavior,
keeping the last valid viewer frame. Configuration controls honor pipeline locks.

Custom Python runs with the application's permissions and installed interpreter.
Load packages you trust, including code embedded in imported pipelines. Execution
is not sandboxed or time-limited: blocking calls, infinite loops, native crashes
and process termination cannot be contained by the callback's exception handling.
