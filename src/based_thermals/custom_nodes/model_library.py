"""
Discover local ONNX models and retain paths explicitly added through Browse.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from ..onnx_models import model_path
from .bundle import MAX_CONFIG_BYTES

LOG = logging.getLogger(__name__)


def library_path() -> Path:
    """
    Keep the custom-model path index beside the shared downloaded-model cache.
    """
    return model_path("espcn").parent / "custom-models.json"


def registered_models() -> list[str]:
    """
    Read the bounded path index without failing UI startup on stale/corrupt metadata.
    """
    path = library_path()
    if not path.exists():
        return []
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise ValueError("Custom model library exceeds 64 KiB")
        values = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(values, list) or len(values) > 128
                or any(not isinstance(value, str) or not Path(value).is_absolute() for value in values)):
            raise ValueError("Custom model library must contain absolute paths")
        return values
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        LOG.warning("Cannot read custom model library: %s", exc)
        return []


def existing_models(current: str = "") -> list[str]:
    """
    List cache/bundled models and registered files; retain the current missing selection.
    """
    paths = {path for value in registered_models() if (path := Path(value)).is_file()}
    roots = [library_path().parent, Path(__file__).resolve().parents[1] / "models"]
    for root in roots:
        paths.update(path for path in root.glob("*.onnx") if path.is_file())
    if current:
        paths.add(Path(current).expanduser())
    return sorted({str(path.absolute()) for path in paths}, key=str.casefold)


def named_model_choices(
    current: str = "", models: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """
    Filter discovery with declarative hints while retaining explicit local overrides.

    Cache checksum suffixes and original basenames identify display names only;
    the worker still verifies bytes and model contracts before using weights.
    """
    selected = str(Path(current).expanduser().absolute()) if current else ""
    choices = []
    for path in existing_models(current):
        file = Path(path)
        if models is None:
            label = f"{file.name} ({file.parent.name})"
        else:
            matches = [model["name"] for model in models
                       if file.name in model.get("filenames", [])
                       or ("sha256" in model
                           and file.name.endswith(f"-{model['sha256'][:12]}.onnx"))]
            if matches:
                label = matches[0]
            elif path == selected:
                label = f"Local override: {file.name}"
            else:
                continue
        if not file.is_file():
            label += " — missing"
        choices.append((label, path))
    choices.sort(key=lambda choice: (choice[0].casefold(), choice[1].casefold()))
    return choices


def add_model(path: Path) -> str:
    """
    Register an existing ONNX path atomically, without copying or loading its weights.
    """
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".onnx" or not path.is_file():
        raise ValueError("Select an existing ONNX model file")
    value = str(path)
    values = registered_models()
    if value in values:
        return value
    values.append(value)
    if len(values) > 128:
        raise ValueError("Custom model library supports at most 128 added paths")
    text = json.dumps(values, ensure_ascii=False)
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ValueError("Custom model library exceeds 64 KiB")
    target = library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, mode="w", encoding="utf-8",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        Path(stream.name).replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return value
