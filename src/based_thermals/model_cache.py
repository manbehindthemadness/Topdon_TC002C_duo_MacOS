"""
Remove recognized downloaded weights while preserving local and bundled models.
"""

import json
import re
from pathlib import Path

from .onnx_models import MODELS, model_path


def clear_downloaded_models(
    *,
    root: Path | None = None,
    include_exports: bool = False,
    clear_registry: bool = False,
) -> tuple[int, int]:
    """
    Return removed file count and bytes from the shared model cache only.

    The download manager must exclude concurrent installs while calling this.
    Registered local files, unknown files and symlinks are retained. Converted
    exports are included only for a settings/model factory reset; its registry
    is removed after registered originals have been protected during deletion.
    """
    root = model_path("espcn").parent if root is None else root
    bundled = Path(__file__).resolve().parent / "models"
    if root.is_symlink() or root.resolve().is_relative_to(bundled):
        raise ValueError("Cannot clear a linked or bundled model directory")
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
    names = [
        re.escape(name)
        for name, spec in MODELS.items()
        if include_exports or not spec.get("local_export")
    ]
    pattern = re.compile(
        rf"(?:({'|'.join(names)})(?:-[0-9a-f]{{12}})?|custom-[0-9a-f]{{24}}-[0-9a-f]{{12}})\.onnx"
    )
    count, removed_bytes = 0, 0
    for path in sorted(root.iterdir()):
        if (
            not pattern.fullmatch(path.name)
            or path.is_symlink()
            or not path.is_file()
            or path.resolve() in protected
        ):
            continue
        candidates = [path, path.with_suffix(".sha256")]
        if include_exports:
            candidates.append(path.with_suffix(".json"))
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            size = candidate.stat().st_size
            candidate.unlink()
            count += 1
            removed_bytes += size
    if include_exports:
        for name, spec in MODELS.items():
            lock = root / f".{name}.export.lock"
            if spec.get("local_export") and lock.is_file() and not lock.is_symlink():
                removed_bytes += lock.stat().st_size
                lock.unlink()
                count += 1
    if clear_registry and registry.is_file() and not registry.is_symlink():
        removed_bytes += registry.stat().st_size
        registry.unlink()
        count += 1
    elif clear_registry and registry.is_symlink():
        registry.unlink()
        count += 1
    return count, removed_bytes
