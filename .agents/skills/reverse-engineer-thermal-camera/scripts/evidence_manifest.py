#!/usr/bin/env python3
"""Hash saved evidence and compare snapshots. No hardware or network access.

A matching manifest proves recorded-file equality, not active camera restoration.
Exit codes: 0 success/match, 1 differences, 2 invalid input or I/O failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
SCOPE = "Offline evidence files only; does not verify active device state or liveness."


def file_record(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ) or count != after.st_size:
        raise ValueError(f"Evidence changed while hashing: {path}")
    return {"bytes": count, "sha256": digest.hexdigest()}


def snapshot(root: Path, output: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Snapshot source must be an existing, non-symlink directory")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    root = root.resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not evidence snapshots: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Unsupported evidence file: {path}")
        files[path.relative_to(root).as_posix()] = file_record(path)
    if not files:
        raise ValueError("Refusing an empty evidence snapshot")
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "created_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "files": files,
    }


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise ValueError(f"Invalid or empty evidence manifest: {path}")
    for name, record in manifest["files"].items():
        relative = PurePosixPath(name)
        if (
            not name
            or name == "."
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or not isinstance(record, dict)
            or type(record.get("bytes")) is not int
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise ValueError(f"Invalid file record in {path}: {name}")
    return manifest


def compare(original: dict, restored: dict) -> dict:
    before, after = original["files"], restored["files"]
    missing = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    changed = [
        {"file": name, "original": before[name], "restored": after[name]}
        for name in sorted(before.keys() & after.keys())
        if (before[name]["bytes"], before[name]["sha256"])
        != (after[name]["bytes"], after[name]["sha256"])
    ]
    return {
        "scope": SCOPE,
        "matching_recorded_files": not (missing or added or changed),
        "original_file_count": len(before),
        "restored_file_count": len(after),
        "missing_files": missing,
        "added_files": added,
        "changed_files": changed,
    }


def write_exclusive(path: Path, data: dict) -> None:
    """Publish a complete report without replacing a prior baseline."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".evidence-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Linking publishes atomically and fails if output already exists.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    take = commands.add_parser("snapshot", help="Hash a directory of saved, quiescent evidence")
    take.add_argument("directory", type=Path)
    take.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("compare", help="Compare relative filenames, sizes and hashes")
    check.add_argument("original", type=Path)
    check.add_argument("restored", type=Path)
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            manifest = snapshot(args.directory, args.output)
            write_exclusive(args.output, manifest)
            print(f"Recorded {len(manifest['files'])} files: {args.output}")
            return 0
        result = compare(load_manifest(args.original), load_manifest(args.restored))
        if args.output is not None:
            write_exclusive(args.output, result)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["matching_recorded_files"] else 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"Evidence error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
