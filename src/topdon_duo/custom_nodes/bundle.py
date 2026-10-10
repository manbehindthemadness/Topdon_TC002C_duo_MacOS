"""
Read and validate portable Python packages without executing user code.
"""

from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

MAX_CONFIG_BYTES = 64 * 1024


def reject_constant(value: str) -> None:
    """
    Reject non-finite values accepted by Python's permissive JSON decoder.
    """
    raise ValueError(f"Invalid JSON constant: {value}")


def read_json_object(text: str, maximum: int | None = MAX_CONFIG_BYTES) -> dict[str, Any]:
    """
    Decode a JSON object with finite numbers and an optional size limit.
    """
    try:
        if not isinstance(text, str):
            raise TypeError("Custom JSON must be text")
        if maximum is not None and len(text.encode("utf-8")) > maximum:
            raise ValueError("Custom JSON exceeds its size limit")
        result = json.loads(text, parse_constant=reject_constant)
        if not isinstance(result, dict):
            raise TypeError("Custom JSON must be an object")
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"Invalid custom JSON: {exc}") from exc
    return result


@lru_cache(maxsize=16)
def checked_files(text: str) -> tuple[tuple[str, str], ...]:
    """
    Validate embedded paths and syntax before materializing a package.
    """
    files = read_json_object(text, None)
    if not files:
        return ()
    if "__init__.py" not in files:
        raise ValueError("Custom package needs __init__.py")
    for name, source in files.items():
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or "\x00" in name
            or path.is_absolute()
            or path.as_posix() != name
            or any(part in (".", "..") or part.startswith(".") for part in path.parts)
            or path.suffix not in (".py", ".json")
            or not isinstance(source, str)
        ):
            raise ValueError(f"Invalid custom package file: {name}")
        try:
            if path.suffix == ".py":
                ast.parse(source, filename=name)
            else:
                value = json.loads(source, parse_constant=reject_constant)
                json.dumps(value, allow_nan=False)
        except (ValueError, SyntaxError, RecursionError) as exc:
            raise ValueError(f"Invalid custom package file {name}: {exc}") from exc
    for name in files:
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents):
            raise ValueError(f"Conflicting custom package path: {name}")
    return tuple(files.items())


def package_files(text: str) -> dict[str, str]:
    """
    Copy cached validated files so callers cannot mutate the validated snapshot.
    """
    result = dict(checked_files(text))
    return result


def load_folder(folder: Path) -> dict[str, str]:
    """
    Snapshot Python/JSON package files, excluding caches and refusing symbolic links.
    """
    files: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Custom packages cannot contain symbolic links")
        if not path.is_file() or path.suffix not in (".py", ".json"):
            continue
        files[relative.as_posix()] = path.read_text(encoding="utf-8")
    if not files:
        raise ValueError("Select a Python package folder containing __init__.py")
    package = json.dumps(files, ensure_ascii=False)
    package_files(package)
    from .configuration import configuration
    from .metadata import embedded_metadata

    metadata = embedded_metadata(files["__init__.py"])
    config, controls = configuration(metadata.get("CONFIG_JSON", files.get("config.json", "{}")))
    model = metadata.get("MODEL_SOURCE", files.get("model.json"))
    if model is not None:
        from .models import model_source

        source = json.loads(model, parse_constant=reject_constant)
        model_source(source)
        config.setdefault("model", source)
    params = {"name": folder.name, "package": package, "config": json.dumps(config),
              "controls": json.dumps(controls)}
    validate_custom(params)
    return params


def validate_custom(params: dict[str, Any]) -> None:
    """
    Validate serialized custom-node parameters without importing the package.
    """
    if not isinstance(params["name"], str) or len(params["name"]) > 200:
        raise ValueError("Invalid custom module name")
    files = package_files(params["package"])
    if not files:
        params["package"] = "{}"
    from .configuration import read_controls
    from .metadata import embedded_metadata

    if files:
        embedded_metadata(files["__init__.py"])
    values = read_json_object(params["config"])
    read_controls(params.get("controls", "[]"), values)
