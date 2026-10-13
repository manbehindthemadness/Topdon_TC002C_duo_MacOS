"""
Remove recognized downloaded weights while preserving local and bundled models.
"""

import json
import re
from pathlib import Path

from .onnx_models import MODELS, model_path


def clear_downloaded_models() -> tuple[int, int]:
    """
    Return removed file count and bytes from the shared model cache only.

    The download manager must exclude concurrent installs while calling this.
    Registered local files, local exports, unknown files and symlinks are retained.
    """
    root = model_path("espcn").parent
    if not root.exists():
        return 0, 0
    registry = root / "custom-models.json"
    protected: set[Path] = set()
    if registry.exists():
        if registry.stat().st_size > 64 * 1024:
            raise ValueError("Cannot safely clear cache: local model index exceeds 64 KiB")
        values = json.loads(registry.read_text(encoding="utf-8"))
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not Path(value).is_absolute() for value in values
        ):
            raise ValueError("Cannot safely clear cache: invalid local model index")
        protected = {Path(value).resolve() for value in values}
    names = [re.escape(name) for name, spec in MODELS.items() if not spec.get("local_export")]
    pattern = re.compile(
        rf"(?:({'|'.join(names)})(?:-[0-9a-f]{{12}})?|custom-[0-9a-f]{{24}}-[0-9a-f]{{12}})\.onnx"
    )
    count, removed_bytes = 0, 0
    for path in sorted(root.iterdir()):
        if (not pattern.fullmatch(path.name) or path.is_symlink() or not path.is_file()
                or path.resolve() in protected):
            continue
        for candidate in (path, path.with_suffix(".sha256")):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            size = candidate.stat().st_size
            candidate.unlink()
            count += 1
            removed_bytes += size
    return count, removed_bytes
