"""
Portable named pipeline bundles and compatibility with individual pipeline files.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...pipeline import validate_pipeline

BUNDLE_FORMAT = "topdon-duo-pipelines"


@dataclass
class IncomingPipeline:
    """
    A validated pipeline and the name and file presented in the import checklist.
    """

    name: str
    document: dict[str, Any]
    source: str


def unique_name(name: str, occupied: set[str]) -> str:
    """
    Suggest a distinct name without overwriting an existing or incoming preset.
    """
    candidate = name
    number = 2
    while candidate in occupied:
        candidate = f"{name} ({number})"
        number += 1
    return candidate


def read_pipelines(path: Path) -> list[IncomingPipeline]:
    """
    Validate a complete bundle or derive a preset name from a legacy file name.
    """
    saved = json.loads(path.read_text())
    if isinstance(saved, dict) and saved.get("format") == BUNDLE_FORMAT:
        if set(saved) != {"format", "version", "pipelines"} or type(saved["version"]) is not int:
            raise ValueError("Invalid pipeline bundle")
        documents = saved["pipelines"]
        if saved["version"] != 1 or not isinstance(documents, dict) or not documents:
            raise ValueError("Unsupported or empty pipeline bundle")
        if any(
            not isinstance(name, str) or not name.strip() or name != name.strip()
            for name in documents
        ):
            raise ValueError("Pipeline names must be nonempty and have no surrounding spaces")
    else:
        name = path.name.removesuffix(".json").removesuffix(".pipeline").strip()
        documents = {name or "Imported pipeline": saved}
    return [
        IncomingPipeline(name, validate_pipeline(document), path.name)
        for name, document in documents.items()
    ]


def write_pipelines(path: Path, documents: dict[str, dict[str, Any]], *, single: bool) -> None:
    """
    Atomically export selected documents, retaining the legacy single-file format.
    """
    validated = {name: validate_pipeline(document) for name, document in documents.items()}
    if not validated:
        raise ValueError("Select at least one pipeline")
    payload = (
        next(iter(validated.values()))
        if single and len(validated) == 1
        else {"format": BUNDLE_FORMAT, "version": 1, "pipelines": validated}
    )
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(serialized)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
