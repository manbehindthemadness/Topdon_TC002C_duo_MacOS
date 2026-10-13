"""
Atomic JSON replacement shared by local preference writers.
"""

import json
import tempfile
from pathlib import Path


def save_json(path: Path, value: object, *, indent: int | None = None) -> None:
    """
    Replace a JSON file using a private sibling temporary file for each save.

    Serialization and write failures preserve the previous file. Overlapping
    writers publish complete documents; the last replacement wins.
    """
    text = json.dumps(value, indent=indent, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            created_file = Path(stream.name)
            temporary = created_file
            stream.write(text)
        created_file.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
